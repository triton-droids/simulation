# scripts/tools/batch_convert_urdf.py
"""
# Run example:
python source/tritonhumanoid/tritonhumanoid/utils/convert_urdf_usd.py \
  --in-dir source/tritonhumanoid/tritonhumanoid/assets/ \
  --out-dir source/tritonhumanoid/tritonhumanoid/assets/ \
  --merge-joints --headless
  
python scripts/zero_agent.py --task=Isaac-Hexapod-Velocity-Direct-v0 --num_envs=16

python scripts/tools/batch_convert_urdf.py \
  --in-dir source/hexapod/hexapod/assets/test \
  --out-dir source/hexapod/hexapod/assets/test \
  --pattern "*.urdf" \
  --nat-freq 25.0 --zeta 1.0 \
  --merge-joints --headless --no-window
"""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Batch URDF→USD with Natural gains / force drive / static base")
parser.add_argument("--in-dir", required=True, help="Folder to search for URDFs (recurses)")
parser.add_argument("--out-dir", required=True, help="Base folder to write per-model subfolders")
parser.add_argument("--pattern", default="*.urdf", help="Glob pattern (recurses)")
parser.add_argument("--merge-joints", action="store_true", default=False)
parser.add_argument("--nat-freq", type=float, default=5.0, help="Natural frequency (Hz)")
parser.add_argument("--zeta", type=float, default=1.0, help="Damping ratio")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app  # one Kit session

# Import after Kit is up
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

in_root = Path(args.in_dir).resolve()
out_root = Path(args.out_dir).resolve()
out_root.mkdir(parents=True, exist_ok=True)

urdfs = sorted(in_root.rglob(args.pattern))
if not urdfs:
    print(f"No URDFs under {in_root} matching {args.pattern}")
else:
    print(f"Found {len(urdfs)} URDF(s). Writing per-model subfolders under {out_root} ...")

for i, urdf in enumerate(urdfs, 1):
    model_name = urdf.stem
    model_dir = out_root / model_name            # e.g., .../rounded_hand/
    model_dir.mkdir(parents=True, exist_ok=True)
    usd_filename = model_name + ".usd"           # e.g., rounded_hand.usd

    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(model_dir),                  # write into per-model folder
        usd_file_name=usd_filename,
        fix_base=False,                           # Static Base
        merge_fixed_joints=args.merge_joints,
        force_usd_conversion=True,
        link_density=1240,                  # roughly PLA plastic
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",                  # Drive type: Force
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
                natural_frequency=args.nat_freq, # Joint config: Natural
                damping_ratio=args.zeta,
            ),
        ),
    )

    print(f"[{i}/{len(urdfs)}] {urdf} -> {model_dir / usd_filename}")
    UrdfConverter(cfg)

app.close()