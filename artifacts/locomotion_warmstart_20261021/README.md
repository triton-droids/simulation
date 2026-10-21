# Triton velocity-policy imitation warm start

`model.pt` is a supervised initializer for the 10-DoF Triton lower-body
velocity policy. It was trained from
`logs/legs_tracking/20260914_170827/reference.npz` with commit `d4b47d1` using:

```bash
cd mjlab
PYTHONPATH=../cad uv run --no-sync python ../cad/train_imitation_warmstart.py \
  --motion ../logs/legs_tracking/20260914_170827/reference.npz \
  --output ../logs/legs_velocity/imitation_init_unclipped.pt \
  --epochs 3000 --batch-size 128 --num-envs 1 --device cpu
```

The deterministic actor takes the velocity task's 41-value observation and
outputs ten dimensionless joint-position actions. The environment multiplies
each output by the 0.2 rad action scale before applying joint safety limits and
the target slew limit.

Training result:

- reference frames: 299 at 50 Hz;
- maximum absolute action target: 3.44;
- action MSE: `3.840986e-05`;
- reconstructed joint-target RMSE: `0.0012395 rad`;
- SHA-256: `d4eb8450f1e5072b495f6939b0190718194b75114c81fbf4a7da6adb15aa6cff`.

Resume velocity PPO with:

```bash
cad/run_velocity_experiment.sh \
  --num-envs 4096 --target-speed 3.0 --hours 8 \
  --run-dir logs/legs_velocity/velocity_unclipped_v1 \
  --resume artifacts/locomotion_warmstart_20261021/model.pt
```

This checkpoint reproduces an open-loop reference and is not a deployable
hardware policy. Evaluate it only in simulation. A stable tracking-teacher
rollout and closed-loop distillation step should precede PPO fine-tuning and
hardware deployment.
