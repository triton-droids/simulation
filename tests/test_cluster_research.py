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
    def test_invalid_resources_rejected_before_network(self):
        import subprocess, sys
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / 'robocub/tools/cluster_research.py'
        for option, value in [('--cpus','0'),('--cpus','9'),('--memory-gb','7'),('--memory-gb','33')]:
            with self.subTest(option=option, value=value):
                result = subprocess.run([sys.executable,str(script),'status','--run','unused',option,value],capture_output=True,text=True)
                self.assertEqual(result.returncode,2)
                self.assertIn('Resources must be',result.stderr)
