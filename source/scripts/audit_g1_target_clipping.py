"""Offline actuator-limit audit of four predetermined saved A13 trajectories."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source.scripts.probe_g1_swing_kinematics import load_model


def clipping_stats(actions, pose, scale, limits):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != len(pose) or not np.isfinite(actions).all():
        raise ValueError('Invalid action trace')
    target = pose + np.clip(actions, -1, 1) * scale
    excess = target - np.clip(target, limits[:, 0], limits[:, 1])
    return {'raw_saturation_fraction': np.mean(np.abs(actions) >= 1, axis=0).tolist(),
            'target_clipped_fraction': np.mean(np.abs(excess) > 1e-7, axis=0).tolist(),
            'target_excess_rms_rad': np.sqrt(np.mean(excess**2, axis=0)).tolist(),
            'target_excess_max_rad': np.max(np.abs(excess), axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    model, _ = load_model()
    from mujoco_playground._src.locomotion.g1.g1_constants import RESTRICTED_JOINT_RANGE
    pose = model.keyframe('knees_bent').qpos[7:]
    names = [model.joint(int(pair[0])).name for pair in model.actuator_trnid]
    assert all(int(model.jnt_qposadr[int(pair[0])]) == 7+i for i,pair in enumerate(model.actuator_trnid))
    records = {}
    for policy in ('b08', 'o01'):
        cfg = json.loads(Path(f'results/post_f1_{policy}/train/environment_effective_config.json').read_text())
        assert cfg.get('action_filter_alpha', 1.) == 1.
        limits = np.asarray(RESTRICTED_JOINT_RANGE) if cfg['restricted_joint_range'] else model.actuator_ctrlrange.copy()
        for command in ('stand', 'forward'):
            path = Path(f'results/post_f1_a13/{policy}_{command}/{policy.upper()}_trace.npz')
            with np.load(path) as trace:
                a = trace['action']
                assert len(a) == 500
                records[f'{policy}_{command}'] = {'source': str(path), 'steps': len(a), 'restricted_joint_range': cfg['restricted_joint_range'], 'limits': limits.tolist(),
                    'windows': {label: clipping_stats(a[window], pose, cfg['action_scale'], limits)
                                for label, window in [('whole',slice(None)),('first100',slice(0,100)),('last250',slice(-250,None))]}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'full_validation': False, 'training_steps': 0, 'physics_rollout': False,
              'actuator_joint_names': names, 'records': records}
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
