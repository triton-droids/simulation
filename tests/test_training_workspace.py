import json
from pathlib import Path
import pytest
from robocub.user_cli.training_recipe import build_recipe,save_recipe,training_argv
from robocub.user_cli.policy_library import policies,checkpoints
from robocub.tools.cluster_research import input_asset_archive
ROOT=Path(__file__).resolve().parents[1]


def values():
    return dict(num_envs='512',seed='101',num_timesteps='81920',batch_size='32',unroll_length='20',num_minibatches='16',num_evals='2',learning_rate='.0003',discounting='.97',entropy_cost='.005',hours='1',cpus='3',memory_gb='16')


def test_export_and_inputs(tmp_path):
    entry=policies(ROOT)[0];row=checkpoints(ROOT,entry)[0]
    recipe=build_recipe(ROOT,entry,row,values(),'atlantis','Continue weights')
    assert recipe['checkpoint'].endswith('/1003520')
    assert recipe['input_assets'] and not recipe['exact_resume']
    assert input_asset_archive(ROOT,recipe)
    saved=save_recipe(tmp_path,recipe)
    plan=json.loads((tmp_path/saved['plan']).read_text())
    from robocub.scripts.run_g1_queue import validate_plan
    validate_plan(plan,tmp_path)
    argv=plan['jobs'][0]['stages'][0]['argv']
    assert '--resume' in argv and '++agent.num_envs=512' in argv
    assert not saved['enabled']
    assert (tmp_path/saved['plan']).with_name('registration.json').exists()


def test_validation_and_new_training():
    entry=policies(ROOT)[0]
    for key,value in [('num_envs','333'),('hours','0'),('learning_rate','nan'),('seed','-1')]:
        settings=values();settings[key]=value
        with pytest.raises(ValueError):build_recipe(ROOT,entry,None,settings,'local','New training')
    recipe=build_recipe(ROOT,entry,None,values(),'local','New training')
    assert recipe['input_assets']==[]
    assert '--resume' not in training_argv(recipe,'Artifacts/Results/test/train')


def test_asset_corruption_rejected():
    entry=policies(ROOT)[0];row=checkpoints(ROOT,entry)[0]
    recipe=build_recipe(ROOT,entry,row,values(),'atlantis','Continue weights')
    recipe['input_assets'][0]['sha256']='wrong'
    with pytest.raises(ValueError,match='hash changed'):input_asset_archive(ROOT,recipe)


def test_workspace_browse_and_paused(monkeypatch):
    from robocub.user_cli.workspace import Workspace
    app=Workspace()
    try:
        app.root.update()
        assert len(app.policy_tree.get_children())==4
        assert app.train_button.instate(['disabled'])
        assert app.loaded_checkpoint['step']==1003520
        app.policy_tree.selection_set('policy1');app.choose_tree()
        assert app.current_checkpoint['total']==100352000
        app.open_results();app.root.update()
        assert app.tabs.select()==str(app.results_tab)
        assert app.result_files
        assert 'local_rb07r_101' in app.result_location.get()
    finally:app.close()

def test_export_composes_saved_hydra_configuration():
    from hydra import compose,initialize
    import robocub.config
    entry=policies(ROOT)[0]
    recipe=build_recipe(ROOT,entry,None,values(),'local','New training')
    args=training_argv(recipe,'Artifacts/Results/workspace_test/train')
    with initialize(version_base=None,config_path=None):
        config=compose(config_name='config',overrides=[x.replace('{root}',ROOT.as_posix()) for x in args if '=' in x])
    assert config.agent.num_envs==512 and config.agent.seed==101
    assert config.sim.reward_scales.tracking_lin_vel==recipe['config']['sim']['reward_scales']['tracking_lin_vel']


def test_embedded_recording_decodes_without_external_player():
    from robocub.user_cli.workspace import Workspace
    app=Workspace()
    try:
        app.root.update();app.recordings()
        videos=[i for i,path in enumerate(app.result_files) if path.suffix=='.mp4']
        assert videos,'Provisioned recording index must contain video evidence'
        app.result_list.selection_clear(0,'end');app.result_list.selection_set(videos[0]);app.show_result();app.root.update()
        assert app.capture is not None and hasattr(app,'preview_image')
        app.seek_video(2);app.toggle_video();app.root.update();app.toggle_video()
    finally:app.close()
