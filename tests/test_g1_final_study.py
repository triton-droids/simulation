import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from source.scripts.report_g1_final_study import build_report, validate_evaluation

ROOT = Path(__file__).resolve().parents[1]


class FinalStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads((ROOT / 'research/queues/final_f1_protocol.json').read_text())
        self.plan = json.loads((ROOT / self.protocol['queue_plan']).read_text())

    def write(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    def fixture(self):
        self.write(self.protocol['queue_plan'], self.plan)
        for family in self.protocol['seed_runs']:
            seed = family['seed']
            self.write(family['untrained_audit'], {'normalizer_count': 0})
            self.write(family['untrained_run'] + '/logs/checkpoints/0/policy', {'seed': seed})
            for i, training in enumerate(family['training']):
                run = training['run_dir']
                self.write(run + '/run_manifest.json', {'jax_backend':'gpu', 'ended_at_utc':'fixture',
                    'command':[a.replace('{root}',str(self.root)) for a in training['argv']]})
                self.write(run + '/resolved_config.json', {'agent':{'seed':seed,'num_timesteps':training['steps']}})
                self.write(run + f"/logs/checkpoints/{training['steps']}/policy", {})
                if i:self.write(run + '/restore_parity.json', {'exact_actor_normalizer_match':True,'leaves':17})
            self.write(f'results/final_f1/seed{seed}/command_prior/audit.json',
                       {'max_old_subspace_action_difference':0,'max_full_checkpoint_round_trip_difference':0})
            for entry in family['evaluations']:
                summary={'commands':self.protocol['commands'],'reset_seeds':self.protocol['reset_seeds'],
                         'steps_per_episode':500,'trained_checkpoint':1003520,'untrained_checkpoint':0,
                         'reference_controller_label':'untrained','reference_run_dir':str(self.root/family['untrained_run']),
                         'reset_randomized':entry['mode']=='randomized','observation_noise':False,
                         'videos':['trained_combined_seed6000.mp4']}
                self.write(entry['output_dir']+'/summary.json',summary)
                rows=[]
                for controller in ('trained','untrained','standing'):
                    for name,values in self.protocol['commands'].items():
                        for reset in self.protocol['reset_seeds']:
                            rows.append(dict(controller=controller,command_name=name,reset_seed=reset,requested_steps=500,
                                command_vx=values[0],command_vy=values[1],command_yaw_rate=values[2],all_finite=True,
                                episode_steps=500 if controller=='trained' else 60,minimum_pelvis_height=.75,
                                linear_velocity_vector_rmse=.1 if controller=='trained' else 1.,yaw_rate_rmse=.1,
                                single_support_fraction=.8,left_median_completed_air_seconds=.3,right_median_completed_air_seconds=.3))
                p=self.root/entry['output_dir']/'episodes.csv'
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

    def test_independent_recipe_and_budget(self):
        self.assertEqual(self.protocol['training_seeds'],[11,22,33])
        self.assertEqual(self.protocol['total_training_steps'],31180800)
        for family in self.protocol['seed_runs']:
            self.assertEqual(len(family['training']),6)
            seed=family['seed']
            for i,t in enumerate(family['training']):
                a=t['argv'];self.assertEqual(a[a.index('--seed')+1],str(seed))
                self.assertIn('hydra.run.dir={root}/'+t['run_dir'],a)
                if i:
                    self.assertIn(f'/seed{seed}/',a[a.index('--checkpoint')+1])
                    self.assertNotIn('gate4_corrective',a[a.index('--checkpoint')+1])
                else:self.assertNotIn('--resume',a)
            self.assertIn('agent.learning_rate=0.0001',family['training'][-1]['argv'])

    def test_full_report_and_failure_not_hidden(self):
        self.fixture()
        report=build_report(self.protocol,self.root)
        self.assertTrue(report['all_numeric_pass'])
        self.assertEqual(len(report['results']),6)
        self.assertEqual(report['status'],'needs_final_visual_review')
        p=self.root/self.protocol['seed_runs'][0]['evaluations'][0]['output_dir']/'episodes.csv'
        with p.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        rows[0]['episode_steps']='100'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.assertFalse(build_report(self.protocol,self.root)['all_numeric_pass'])

    def test_wrong_reference_and_protocol_rejected(self):
        self.fixture()
        p=self.root/self.protocol['seed_runs'][0]['evaluations'][0]['output_dir']/'summary.json'
        s=json.loads(p.read_text());s['reference_run_dir']=str(self.root/'wrong')
        p.write_text(json.dumps(s))
        with self.assertRaises(AssertionError):build_report(self.protocol,self.root)

    def test_shared_initializations_rejected(self):
        self.fixture()
        for family in self.protocol['seed_runs']:
            self.write(family['untrained_run']+'/logs/checkpoints/0/policy',{'same':True})
        with self.assertRaises(AssertionError):build_report(self.protocol,self.root)


if __name__=='__main__':unittest.main()
