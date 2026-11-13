import io
import tarfile
import unittest
from robocub.tools.cluster_research import validate_archive


class SnapshotTests(unittest.TestCase):
    def archive(self, name, symlink=False):
        out=io.BytesIO()
        with tarfile.open(fileobj=out, mode='w') as t:
            item=tarfile.TarInfo(name)
            if symlink: item.type=tarfile.SYMTYPE; item.linkname='/outside'
            t.addfile(item)
        return out.getvalue()

    def test_source_allowed(self):
        validate_archive(self.archive('robocub/scripts/train.py'))

    def test_private_and_traversal_rejected(self):
        for name in ['.env','.ssh/id_rsa','.cache/model','../escape','/absolute']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_archive(self.archive(name))

    def test_symlink_rejected(self):
        with self.assertRaises(ValueError): validate_archive(self.archive('link',True))


class ResourceArgumentTests(unittest.TestCase):
    def test_duration_bounds_and_format(self):
        from robocub.tools.cluster_research import slurm_time
        self.assertEqual(slurm_time(8), '08:00:00')
        self.assertEqual(slurm_time(24), '24:00:00')
        for value in [0,25,1.5,True]:
            with self.assertRaises(ValueError): slurm_time(value)

    def test_duration_rejected_before_network(self):
        import subprocess, sys
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / 'robocub/tools/cluster_research.py'
        for value in ['0','25']:
            result = subprocess.run([sys.executable,str(script),'submit','--run','unused',
                '--slurm-hours',value],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('Slurm duration must be',result.stderr)

    def test_invalid_resources_rejected_before_network(self):
        import subprocess, sys
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / 'robocub/tools/cluster_research.py'
        for option, value in [('--cpus','0'),('--cpus','9'),('--memory-gb','7'),('--memory-gb','33')]:
            with self.subTest(option=option, value=value):
                result = subprocess.run([sys.executable,str(script),'status','--run','unused',option,value],capture_output=True,text=True)
                self.assertEqual(result.returncode,2)
                self.assertIn('Resources must be',result.stderr)


class CompactStatusTests(unittest.TestCase):
    def test_preserves_failure_and_scheduler_without_gate_details(self):
        from robocub.tools.cluster_research import compact_status
        result = compact_status('{"status":"error","error":"stage failed","jobs":[{"id":"train","status":"error","heartbeat_at":"now","assessment":{"large":"details"}}]}\nSLURM FAILED')
        self.assertEqual(result['error'], 'stage failed')
        self.assertEqual(result['jobs'][0]['heartbeat_at'], 'now')
        self.assertNotIn('assessment', result['jobs'][0])
        self.assertIn('FAILED', result['scheduler'])

    def test_missing_status_keeps_scheduler(self):
        from robocub.tools.cluster_research import compact_status
        self.assertEqual(compact_status('SLURM PENDING')['status'], 'unknown')
        self.assertIn('PENDING', compact_status('SLURM PENDING')['scheduler'])


class RuntimeProfileTests(unittest.TestCase):
    def test_profile_is_explicit_and_legacy_is_unchanged(self):
        from robocub.tools.cluster_research import runtime_setup, RUNTIME_PROFILES
        self.assertEqual(runtime_setup('legacy'), '')
        self.assertEqual(RUNTIME_PROFILES['reproducible'], '--xla_gpu_autotune_level=0')
        with self.assertRaises(KeyError): runtime_setup('unrecognized')

    def test_invalid_profile_rejected_before_network(self):
        import subprocess, sys
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / 'robocub/tools/cluster_research.py'
        result = subprocess.run([sys.executable,str(script),'status','--run','unused',
                                 '--runtime-profile','unrecognized'],capture_output=True,text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)
