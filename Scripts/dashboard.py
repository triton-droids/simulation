"""Open the unified RoboCup dashboard."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robocub.user_cli.dashboard import main

if __name__ == '__main__':
    raise SystemExit(main())
