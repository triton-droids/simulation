"""One obvious entry point for watching, training, and inspecting RoboCup work."""

from pathlib import Path
import sys


# Put the repository root before this `App` directory so the `robocub/`
# package wins over this intentionally similarly named launcher file.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from robocub.user_cli.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
