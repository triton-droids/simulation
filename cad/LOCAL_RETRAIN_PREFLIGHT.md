# Local retraining preflight — 2026-10-06

Status: preparation only; training has NOT started.

Repository: /home/thewheelneedsme/Projects/droids/simulation
Local branch: codex/hardware-retrain-local, based on 6fcd451.
Original run logs/legs_tracking/20260914_170827 was not modified.

## Hardware evidence
Downloaded ten NPZ logs from triton-droids/embedded commit 2728fe19eb17908ac5bd4d88c8e9a131111a9afd to hardware_data/embedded_2728fe1. See audit.json there for per-run metadata, command tracking residuals, logged motor torque and velocity statistics.
Four live logs stopped after five missed motor replies; one stopped on a knee position limit. Earlier and later logs use different PD gains. Do not pool them for fitting without conditioning on gains and freshness. Motor torque feedback is not calibrated joint torque, particularly across the ankle linkage. Statistics are descriptive, not system identification.
Latest metadata confirms 50 Hz, target slew 1 rad/s, motor command caps 80/40/11 Nm, kp 100/100/100/80/22 per leg and kd 2. Fit the exact command guard and linkage before treating these caps as a simple joint-torque clamp.

## Execution blocker
Running kernel 7.0.0-34-generic. NVIDIA PCI device is visible. nvidia-smi fails, torch.cuda.is_available() is False, device_count is zero; modinfo nvidia reports module not found, DKMS status empty.
APT has linux-modules-nvidia-595-open-7.0.0-34-generic available but uninstalled. A simulated installation proposes 16 upgrades and 7 new packages, including NVIDIA 595.91.07 userspace and a newer kernel. No driver packages were installed and no reboot was performed. System package repair is outside the writable workspace and requires an administrator.

## Environment and baseline
Existing environment: Python 3.13, torch 2.9.0+cu128, MuJoCo 3.8.1, rsl-rl-lib 5.2.0. The fetched launcher instead pins Python 3.12 and its lockfile differs; resolve this deliberately before launch rather than silently replacing the working environment.
CPU baseline ONNX verifier passed sampled frames and last-frame clamping, maximum reference error 1.19e-07. This is not all-frame verification or a physics evaluation.

## Remaining work before production training
- Audit and reproduce deployed software torque guard and ankle linkage; the branch still models datasheet peaks only.
- Resolve fixed-reference feasibility under 1 rad/s target slew without changing the deploy contract.
- Finish configurable model parameters and sustained torque accounting.
- Validate exact IMU rotation, per-physics-step torque limits and extreme-condition coverage.
- Ensure baseline evaluation actually runs baseline ONNX (existing launcher runs the PT checkpoint).
- Smoke-test environment and 56-D/50 Hz contract after CUDA repair, then start a fresh run and retain all requested evaluations and artifacts.

## Resolved and tested (2026-10-06)

User installed the matching NVIDIA module and loaded it. CUDA tensor execution now passes on RTX 5060 Ti. Local corrections add configurable conservative software torque caps, exact bounded IMU rotation and per-physics-step torque-speed updates. Corrected 256-env / 3-iteration training and all-frame ONNX verification passed (max reference error 2.36e-07). A short wall-clock budget test stopped cleanly, evaluated all three 64-env / 10-second conditions and saved best.pt / best.onnx. Nominal baseline ONNX evaluation completed with 608 failure resets; this is a baseline, not a walking acceptance pass. All sampled torque caps passed.

The user requested an approximately eight-hour run and explicitly requested to start it themselves. The full run has NOT been started. Launcher: cad/run_hardware_8h.sh. Outputs go to a fresh logs/legs_tracking_hardware/<timestamp>_8h directory. Checkpoint every 500 iterations; selection every 5000; eight-hour budget checked at checkpoints; final validation adds a few minutes.
