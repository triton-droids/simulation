"""Print the selected robot, simulator, checkpoint, and exact weights path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robocub.user_cli.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["status", *sys.argv[1:]]))
