"""Stage checkpoint directory transactions off Windows-mounted filesystems."""
from pathlib import Path
import shutil
import tempfile


def save_staged_checkpoint(destination, write_checkpoint):
    """Finalize locally, then copy to a fresh destination; retain partial copies."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    # Under WSL /tmp is Linux-native, avoiding DrvFS directory rename failures.
    with tempfile.TemporaryDirectory(prefix="g1-checkpoint-", dir="/tmp" if Path("/proc/sys/kernel").exists() else None) as temporary:
        staged = Path(temporary) / "checkpoint"
        write_checkpoint(staged)
        shutil.copytree(staged, destination)
