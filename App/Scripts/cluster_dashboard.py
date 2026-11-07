"""Open the visual Atlantis training dashboard from PyCharm's green button."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robocub.user_cli.cluster_dashboard import main


if __name__ == "__main__":
    raise SystemExit(main())
