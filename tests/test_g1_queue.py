"""Stdlib-only runner tests, including real subprocess failures on Linux."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from source.scripts.run_g1_queue import assess_gate, execute_stage, local_path, metrics_health, source_is_clean, validate_plan, screening_allows


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.monitor = dict(ignore_kl_through_step=100, consecutive_kl_limit=2, kl_limit=.2)

    def metrics(self, values, tail=''):
        p = self.root / 'metrics.jsonl'
        p.write_text(''.join(json.dumps({'event': 'metrics', 'step': step,
                     'metrics': {'training/kl_mean': value}}) + '\n' for step, value in values) + tail)
        return metrics_health(p, self.monitor)

    def test_screen_dependencies_fail_closed(self):
        job = {'requires_screening': ['screen']}
        for status in ['screening_rejected', 'error', 'needs_review', 'needs_visual_review']:
            self.assertFalse(screening_allows(job, [{'id': 'screen', 'status': status}]))
        self.assertFalse(screening_allows(job, []))
        self.assertTrue(screening_allows(job, [{'id': 'screen', 'status': 'screening_passed'}]))

    def test_screen_rejects_future_or_non_screen_dependency(self):
        plan=json.loads((Path(__file__).resolve().parents[1]/'research/queues/c16_balance.json').read_text())
        plan['jobs'][0]['requires_screening']=['missing']
        with self.assertRaises(ValueError): validate_plan(plan,self.root)

    def test_transient_and_partial_write(self):
        self.assertEqual(self.metrics([(0, 0), (100, .8), (200, .04)], '{'), (None, 200))

    def test_sustained_kl_even_if_recovered(self):
        self.assertIn('consecutive', self.metrics([(100,.8),(200,.3),(300,.25),(400,.05)])[0])

    def test_nonfinite_including_initial(self):
        self.assertIn('nonfinite', self.metrics([(0, float('nan'))])[0])

    def test_missing_monitored_key(self):
        p = self.root / 'metrics.jsonl'
        p.write_text(json.dumps({'event':'metrics','step':200,'metrics':{}})+'\n')
        self.assertIn('missing', metrics_health(p, self.monitor)[0])

    def rows(self, steps=(500,500,500,500), height=.7):
        rows = []
        for controller in ('trained','untrained','standing'):
            for i, (cmd,seed) in enumerate((('stand',3000),('stand',3001),('forward',3000),('forward',3001))):
                rows.append(dict(controller=controller,command_name=cmd,reset_seed=seed,
                                 requested_steps=500,episode_steps=steps[i],minimum_pelvis_height=height,
                                 linear_velocity_vector_rmse=.1,yaw_rate_rmse=.1,all_finite=True))
        return rows

    def assess(self, rows):
        p=self.root/'episodes.csv'
        with p.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        plan=json.loads((Path(__file__).resolve().parents[1]/'research/queues/c16_balance.json').read_text())
        return assess_gate(p,plan['jobs'][0]['gate'])

    def test_balance_pass_still_requires_visual_review(self):
        result=self.assess(self.rows())
        self.assertTrue(result['outcomes']['pass']['passed'])
        self.assertTrue(result['visual_review_required'])

    def test_extension_eligibility_is_not_pass(self):
        result=self.assess(self.rows((500,500,200,200)))
        self.assertFalse(result['outcomes']['pass']['passed'])
        self.assertTrue(result['outcomes']['extension_eligible']['passed'])

    def test_reject_missing_duplicate_and_nonfinite_evidence(self):
        rows=self.rows()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):self.assess(bad)
        rows[0]['minimum_pelvis_height']=float('nan')
        with self.assertRaises(ValueError):self.assess(rows)

    def test_height_boundary_is_strict(self):
        result=self.assess(self.rows(height=.6))
        self.assertFalse(result['outcomes']['pass']['passed'])
        self.assertFalse(result['outcomes']['extension_eligible']['passed'])

    def test_paths_cannot_escape(self):
        with self.assertRaises(ValueError):local_path(self.root,'../outside')

    def stage(self, code, **extra):
        stage=dict(id='test', argv=['-c',code],fresh_paths=['evidence'],required_paths=[],timeout_seconds=5)
        stage.update(extra)
        state={}
        execute_stage(self.root,stage,self.root,state,lambda:None,time.monotonic()+10,poll_seconds=.01)
        return state

    @unittest.skipUnless(sys.platform=='linux', 'Process-group supervision uses Linux/WSL')
    def test_child_failure_and_no_overwrite(self):
        with self.assertRaisesRegex(RuntimeError,'exited 3'):self.stage('raise SystemExit(3)')
        (self.root/'evidence').mkdir()
        with self.assertRaises(FileExistsError):self.stage('pass')

    @unittest.skipUnless(sys.platform=='linux', 'Process-group supervision uses Linux/WSL')
    def test_timeout_terminates_child(self):
        with self.assertRaises(TimeoutError):self.stage('import time;time.sleep(60)',timeout_seconds=.05)

    @unittest.skipUnless(sys.platform=='linux', 'Process-group supervision uses Linux/WSL')
    def test_zero_exit_without_artifact_is_error(self):
        with self.assertRaisesRegex(RuntimeError,'required artifact'):self.stage('pass',required_paths=['missing'])

    @unittest.skipUnless(sys.platform=='linux', 'Process-group supervision uses Linux/WSL')
    def test_successful_process_and_artifact(self):
        state=self.stage("from pathlib import Path;Path('evidence').mkdir()",required_paths=['evidence'])
        self.assertIsNone(state['child_pid'])

    @unittest.skipUnless(sys.platform=='linux', 'Full queue executes on Linux/WSL')
    def test_full_queue_records_result_and_refuses_second_launch(self):
        project=Path(__file__).resolve().parents[1]
        runner=self.root/'source/scripts/run_g1_queue.py'
        runner.parent.mkdir(parents=True)
        runner.write_text((project/'source/scripts/run_g1_queue.py').read_text())
        (self.root/'.gitignore').write_text('results/\n__pycache__/\n')
        (self.root/'crlf_fixture.txt').write_bytes(b'unchanged content\n')
        plan=json.loads((project/'research/queues/c16_balance.json').read_text())
        plan['output_dir']='results/queue'
        job=plan['jobs'][0]
        job['episodes_csv']='results/eval/episodes.csv'
        job['stages']=[dict(id='smoke', argv=['-c',
            "from pathlib import Path;import csv;Path('results/eval').mkdir();"
            "f=open('results/eval/episodes.csv','w');"
            f"rows={self.rows()!r};w=csv.DictWriter(f,fieldnames=rows[0].keys());"
            "w.writeheader();w.writerows(rows);f.close()"],
            fresh_paths=['results/eval'],required_paths=['results/eval/episodes.csv'],timeout_seconds=30)]
        plan['jobs'].append(dict(id='diagnostic', hypothesis='fixture', evidence='fixture',
            review_artifact='results/diagnostic.json', stages=[dict(id='diagnostic',
            argv=['-c', "from pathlib import Path;Path('results/diagnostic.json').write_text('{}')"],
            fresh_paths=['results/diagnostic.json'], required_paths=['results/diagnostic.json'], timeout_seconds=30)]))
        import copy
        screen=copy.deepcopy(job)
        screen['id']='screen';screen['screening_only']=True
        screen['episodes_csv']='results/screen/episodes.csv'
        screen['stages']=json.loads(json.dumps(screen['stages']).replace('results/eval','results/screen'))
        screen['gate']['outcomes']['pass']=[dict(name='forced rejection',aggregate='min',metric='episode_steps',min=501)]
        plan['jobs'].append(screen)
        blocked=copy.deepcopy(plan['jobs'][1]);blocked['id']='blocked'
        blocked['requires_screening']=['screen']
        blocked=json.loads(json.dumps(blocked).replace('results/diagnostic.json','results/blocked.json'))
        plan['jobs'].append(blocked)
        p=self.root/'plan.json';p.write_text(json.dumps(plan))
        for args in (['init','-q'],['add','.'],['-c','user.name=Queue Test','-c','user.email=test@example.invalid','commit','-qm','fixture']):
            subprocess.run(['git','-c','core.autocrlf=true']+args,cwd=self.root,check=True,capture_output=True)
        # Windows checkout line endings are not a semantic source change.
        (self.root/'crlf_fixture.txt').write_bytes(b'unchanged content\r\n')
        command=[sys.executable,str(runner),'--plan',str(p)]
        first=subprocess.run(command,cwd=self.root,capture_output=True,text=True,timeout=75)
        self.assertEqual(first.returncode,0,first.stderr)
        status_path=self.root/'results/queue/status.json'
        before=status_path.read_bytes()
        status=json.loads(before)
        self.assertEqual(status['status'],'needs_review')
        self.assertEqual(status['jobs'][0]['status'],'needs_visual_review')
        self.assertTrue(status['plan_sha256'])
        self.assertEqual(status['jobs'][2]['status'],'screening_rejected')
        self.assertFalse(status['jobs'][2]['assessment']['full_validation'])
        self.assertEqual(status['jobs'][3]['status'],'skipped_screening')
        self.assertFalse((self.root/'results/blocked.json').exists())
        self.assertEqual(status['jobs'][1]['status'], 'needs_review')
        self.assertFalse(status['jobs'][1]['assessment']['numeric_gate_evaluated'])
        second=subprocess.run(command,cwd=self.root,capture_output=True,text=True,timeout=10)
        self.assertNotEqual(second.returncode,0)
        self.assertEqual(status_path.read_bytes(),before)
        self.assertTrue(source_is_clean(self.root))
        (self.root/'crlf_fixture.txt').write_text('actually changed\n')
        self.assertFalse(source_is_clean(self.root))


if __name__=='__main__':
    unittest.main()
