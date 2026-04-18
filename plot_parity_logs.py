"""
Plot IsaacLab vs MuJoCo parity logs.

Example:
  python plot_parity_logs.py \
    --isaac_npz parity/isaac_policy_rollout.npz \
    --mujoco_npz parity/mujoco_replay_from_isaac.npz \
    --compare_json parity/mujoco_replay_from_isaac_compare.json \
    --out_dir parity/plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_CHANNELS = [
    "lin_vel_cmd",
    "ang_vel_cmd",
    "ang_vel_cmd_scaled",
    "up_cmd",
    "commands",
    "act_pos_scaled",
    "act_vel_scaled",
    "prev_actions",
    "clock",
    "actions_applied",
]


def _align_by_lag(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return x[:-lag], y[lag:]
    if lag < 0:
        return x[-lag:], y[:lag]
    return x, y


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _load_compare(compare_json: str | None) -> dict[str, Any]:
    if compare_json is None:
        return {}
    path = Path(compare_json)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _channel_lag(compare: dict[str, Any], channel: str, dim: int) -> int:
    rows = compare.get(channel)
    if not isinstance(rows, list):
        return 0
    for row in rows:
        if int(row.get("dim", -1)) == dim:
            return int(row.get("best_lag_steps", 0))
    return 0


def _plot_channel(
    channel: str,
    isaac: np.ndarray,
    mujoco: np.ndarray,
    time_s: np.ndarray,
    compare: dict[str, Any],
    use_best_lag: bool,
    out_path: Path,
) -> dict[str, Any]:
    isaac = _to_2d(isaac)
    mujoco = _to_2d(mujoco)
    T = min(isaac.shape[0], mujoco.shape[0], time_s.shape[0])
    isaac = isaac[:T]
    mujoco = mujoco[:T]
    time_s = time_s[:T]
    D = min(isaac.shape[1], mujoco.shape[1])
    isaac = isaac[:, :D]
    mujoco = mujoco[:, :D]

    fig, axes = plt.subplots(D, 2, figsize=(12, max(3, 2.2 * D)), squeeze=False)
    fig.suptitle(channel)

    dim_stats = []
    for d in range(D):
        lag = _channel_lag(compare, channel, d) if use_best_lag else 0
        xi, yi = _align_by_lag(isaac[:, d], mujoco[:, d], lag)
        ti, _ = _align_by_lag(time_s, time_s, lag)
        err = yi - xi

        corr = _safe_corr(xi, yi)
        rmse = float(np.sqrt(np.mean(err * err))) if err.size > 0 else 0.0

        ax0 = axes[d, 0]
        ax0.plot(ti, xi, label="isaac", linewidth=1.2)
        ax0.plot(ti, yi, label="mujoco", linewidth=1.0)
        ax0.set_ylabel(f"dim {d}")
        ax0.grid(True, alpha=0.25)
        if d == 0:
            ax0.legend(loc="upper right")
        ax0.set_title(f"overlay | lag={lag} | corr={corr:.3f}")

        ax1 = axes[d, 1]
        ax1.plot(ti, err, color="tab:red", linewidth=1.0)
        ax1.grid(True, alpha=0.25)
        ax1.set_title(f"error (mujoco-isaac) | rmse={rmse:.4g}")

        if d == D - 1:
            ax0.set_xlabel("time [s]")
            ax1.set_xlabel("time [s]")

        dim_stats.append({"dim": d, "lag": lag, "corr": corr, "rmse": rmse})

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"dims": D, "stats": dim_stats}


def main():
    parser = argparse.ArgumentParser(description="Plot parity channels from Isaac and MuJoCo NPZ logs.")
    parser.add_argument("--isaac_npz", type=str, required=True, help="IsaacLab log .npz")
    parser.add_argument("--mujoco_npz", type=str, required=True, help="MuJoCo log .npz")
    parser.add_argument("--compare_json", type=str, default=None, help="Optional compare metrics json")
    parser.add_argument("--out_dir", type=str, default="parity/plots", help="Output directory for plots")
    parser.add_argument(
        "--channels",
        type=str,
        default=",".join(DEFAULT_CHANNELS),
        help="Comma-separated channel list",
    )
    parser.add_argument(
        "--use_best_lag",
        action="store_true",
        default=False,
        help="Use best_lag_steps from compare_json when available",
    )
    args = parser.parse_args()

    isaac = np.load(args.isaac_npz)
    mujoco = np.load(args.mujoco_npz)
    compare = _load_compare(args.compare_json)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    common = [c for c in channels if c in isaac.files and c in mujoco.files]
    missing = [c for c in channels if c not in common]

    if "time_s" in isaac.files:
        time_s = np.asarray(isaac["time_s"], dtype=np.float64)
    elif "time_s" in mujoco.files:
        time_s = np.asarray(mujoco["time_s"], dtype=np.float64)
    else:
        # fallback to step index
        t_len = min(isaac[common[0]].shape[0], mujoco[common[0]].shape[0]) if common else 0
        time_s = np.arange(t_len, dtype=np.float64)

    summary = {
        "isaac_npz": os.path.abspath(args.isaac_npz),
        "mujoco_npz": os.path.abspath(args.mujoco_npz),
        "compare_json": None if args.compare_json is None else os.path.abspath(args.compare_json),
        "use_best_lag": bool(args.use_best_lag),
        "channels_plotted": common,
        "channels_missing": missing,
        "per_channel": {},
    }

    for ch in common:
        out_path = out_dir / f"{ch}.png"
        stats = _plot_channel(
            channel=ch,
            isaac=np.asarray(isaac[ch], dtype=np.float64),
            mujoco=np.asarray(mujoco[ch], dtype=np.float64),
            time_s=time_s,
            compare=compare,
            use_best_lag=bool(args.use_best_lag),
            out_path=out_path,
        )
        summary["per_channel"][ch] = stats

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[INFO] Wrote plots to: {out_dir}")
    print(f"[INFO] Summary: {summary_path}")
    if missing:
        print(f"[WARN] Missing channels skipped: {missing}")


if __name__ == "__main__":
    main()
