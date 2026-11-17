import json
from pathlib import Path
import pytest
from robocub.user_cli.policy_library import checkpoints, viewer_command
from robocub.user_cli.dashboard import Service


def test_checkpoint_timestamp_steps_and_explicit_viewer(tmp_path):
    run = tmp_path/'Policies/example'
    folder = run/'logs/checkpoints/20'
    folder.mkdir(parents=True)
    (folder/'policy').write_bytes(b'weights')
    (folder/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs':1700000000000000000}))
    (run/'resolved_config.json').write_text('{}')
    row, = checkpoints(tmp_path, dict(run_dir='Policies/example',ancestor_steps=100))
    assert row['total'] == 120
    assert row['saved'] == '2023-11-14 22:13:20 UTC'
    assert row['timestamp_source'] == 'Checkpoint commit'
    assert viewer_command(tmp_path,'python',row)[-4:] == ['--run-dir',str(run),'--checkpoint','20']
    (folder/'policy').unlink()
    with pytest.raises(ValueError, match='missing'): viewer_command(tmp_path,'python',row)


def test_missing_checkpoint_visible_and_escape_rejected(tmp_path):
    folder = tmp_path/'Policies/test/logs/checkpoints/0'
    folder.mkdir(parents=True)
    row, = checkpoints(tmp_path,dict(run_dir='Policies/test'))
    assert not row['ready']
    assert row['timestamp_source'].startswith('File modified')
    with pytest.raises(ValueError): checkpoints(tmp_path,dict(run_dir='../outside'))


def test_pause_blocks_launch_before_any_subprocess(tmp_path):
    marker = tmp_path/'Artifacts/Results/local_rb07_recovery/USER_PAUSED.txt'
    marker.parent.mkdir(parents=True)
    marker.touch()
    with pytest.raises(ValueError,match='paused'): Service(tmp_path).launch({})


def test_dashboard_selection_and_launch_wiring(monkeypatch):
    from robocub.user_cli.dashboard import Dashboard
    calls = []
    monkeypatch.setattr('robocub.user_cli.dashboard.subprocess.Popen',lambda command,**kw: calls.append(command))
    app = Dashboard()
    try:
        app.root.update()
        assert app.tabs.tab(app.tabs.select(),'text') == 'Policies & saved checkpoints'
        assert len(app.policy_entries) >= 4
        assert app.launch_button.instate(['disabled'])
        app.load_checkpoint()
        assert app.loaded_checkpoint['step'] == 1003520
        app.watch()
        assert calls[0][-2:] == ['--checkpoint','1003520']
        app.policy_combo.current(1); app.policy_changed()
        assert app.loaded_checkpoint is None
        app.load_checkpoint()
        assert app.loaded_checkpoint['total'] == 100352000
        app.watch()
        assert calls[1][-2:] == ['--checkpoint','5734400']
    finally:
        app.root.destroy()
