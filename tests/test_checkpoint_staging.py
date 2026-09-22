from pathlib import Path
import tempfile
import unittest

from source.utils.checkpoint_staging import save_staged_checkpoint


class CheckpointStagingTests(unittest.TestCase):
    def test_copy_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "saved"
            def write(path):
                path.mkdir()
                (path / "policy").write_bytes(b"policy")
                (path / "metadata").mkdir()
                (path / "metadata" / "state").write_bytes(b"state")
            save_staged_checkpoint(target, write)
            self.assertEqual((target / "policy").read_bytes(), b"policy")
            self.assertEqual((target / "metadata" / "state").read_bytes(), b"state")
            with self.assertRaises(FileExistsError):
                save_staged_checkpoint(target, write)

    def test_writer_failure_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "saved"
            def fail(path):
                path.mkdir()
                raise RuntimeError("save failed")
            with self.assertRaises(RuntimeError):
                save_staged_checkpoint(target, fail)
            self.assertFalse(target.exists())
