# Field01 HPC runbook

No remote job has been submitted. The SLURM file is a template; account, partition,
module names and site approval are deliberately unspecified. First reproduce N=1
headless locally. Pin the same dependency versions and source hashes; copy licensed
G1 assets with their source LICENSE, not unknown team assets. Use a fresh output
path. Never install into the existing locomotion environment.

Windows: `.cache/field01_venv/Scripts/python.exe robocub/tools/benchmark_field01.py --robot g1 --counts 1 16 256 --output Artifacts/Results/field01/fresh_gpu_batch`

Linux: `bash scripts/benchmark_field01.sh Artifacts/Results/field01/fresh_gpu_batch`

The benchmark records commands, exit codes, logs, throughput and once-per-second
whole-device NVIDIA memory/utilization samples. Memory includes the desktop and
other processes; sampling can miss short peaks. Torch allocator measurements omit
Warp allocations. Setup/warmup is excluded from timed throughput. Each child has
a ten-minute timeout and the batch stops on failure. No fake completion on timeout.

Only consider 1024/4096 after N=256 passes with adequate memory headroom. This laptop
has 8GB shared with its graphical desktop. Do not open hundreds of rendered cameras.
Use per-job compiler caches in scheduler scratch. For browser viewers bind localhost
and use approved SSH port forwarding; never publicly expose the visualization server.

A larger compute allocation does not fix importer, physics-calibration or API defects.
All recorded throughput is scripted simulation, not PPO training throughput. A G1
fall under nominal joint holding is a controller limitation; no learned locomotion
policy has been integrated into this soccer task yet.
