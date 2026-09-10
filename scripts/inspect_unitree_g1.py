"""Print deterministic metadata for the repository-pinned Unitree G1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.robots.unitree_g1 import UnitreeG1Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Explicit local scene_mjx.xml override.")
    parser.add_argument("--menagerie-root", help="Pinned Menagerie checkout root.")
    parser.add_argument("--cache-root", help="Alternate hidden cache root.")
    parser.add_argument("--no-fetch-model", action="store_false", dest="fetch_model")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.set_defaults(fetch_model=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = UnitreeG1Model(
        model_path=args.model,
        menagerie_root=args.menagerie_root,
        cache_root=args.cache_root,
        fetch=args.fetch_model,
    )
    payload = {
        "model_source": robot.source_record,
        "metadata": robot.metadata.to_dict(),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
