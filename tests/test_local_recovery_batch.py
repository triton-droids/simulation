"""Budget, replay and shutdown safeguards for the local recovery supervisor."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from robocub.tools.local_recovery_batch import ROOT, stop_queue, validate_manifest


class RecoveryBatchTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT/'documentation/research/queues/local_rb07_recovery_batch.json').read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for run in self.manifest['runs']:
            path = self.root/run['plan']
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((ROOT/run['plan']).read_bytes())

    def test_frozen_plans(self):
        validate_manifest(self.manifest,self.root)

    def test_hash_and_budget_rejected(self):
        for change in ('hash', 'budget', 'duplicate'):
            manifest = copy.deepcopy(self.manifest)
            if change == 'hash':
                manifest['runs'][0]['sha256'] = '0'*64
            elif change == 'budget':
                manifest['max_wall_seconds'] = 62*3600
            else:
                manifest['runs'].append(manifest['runs'][0])
            with self.assertRaises(ValueError):
                validate_manifest(manifest,self.root)

    def test_existing_batch_rejected(self):
        self.manifest['output_dir'] = 'documentation/research'
        with self.assertRaisesRegex(ValueError, 'Existing batch'):
            validate_manifest(self.manifest,self.root)

    @unittest.skipUnless(sys.platform == 'linux', 'POSIX process groups')
    def test_fallback_kills_owned_stage_then_queue(self):
        child = Mock(pid=321)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('queue',45),0]
        from unittest.mock import MagicMock
        owned, foreign = MagicMock(), MagicMock()
        owned.name, foreign.name = '654', '987'
        owned.__truediv__.return_value.read_text.return_value = '654 (python) S 321 654'
        foreign.__truediv__.return_value.read_text.return_value = '987 (python) S 999 987'
        with patch.object(Path,'iterdir',return_value=iter([owned,foreign])), patch('os.getpgid',return_value=654), patch('os.killpg') as kill:
            stop_queue(child)
        self.assertEqual(kill.call_args_list, [unittest.mock.call(321,signal.SIGINT),unittest.mock.call(654,signal.SIGKILL),unittest.mock.call(321,signal.SIGKILL)])

    def test_gates_and_original_controls_preserved(self):
        for seed, remaining in ((101,306790400),(202,312524800),(303,312524800)):
            original = json.loads((ROOT/f'documentation/research/queues/cluster_rb07_{seed}.json').read_text())
            local = json.loads((ROOT/f'documentation/research/queues/local_rb07r_{seed}.json').read_text())
            for old,new in zip(original['jobs'][1:],local['jobs'][1:]):
                for key in ('gate','requires_screening','screening_only'):
                    self.assertEqual(old.get(key),new.get(key))
                for stage in new['stages']:
                    argv = stage['argv']
                    if '--reference-run-dir' in argv:
                        self.assertEqual(argv[argv.index('--reference-run-dir')+1], f'Artifacts/Results/local_rb07_parents/{seed}')
            argv = local['jobs'][0]['stages'][1]['argv']
            self.assertIn(f'agent.num_timesteps={remaining}',argv)
            self.assertNotIn('local_lp01_capacity',str(local))


if __name__ == '__main__':
    unittest.main()
