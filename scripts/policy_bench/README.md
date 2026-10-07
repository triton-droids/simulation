# Standalone tracking ONNX bench

This program reads the ESP32-S3 IMU directly over serial and writes ten target
joint angles in radians and degrees to CSV. It sends no motor commands and
uses explicit zero/reference joint-state placeholders. Use the sibling embedded
repository's `docs/tracking_policy_ros2.md` for the ROS 2 integration instead.
Only one program should own the IMU serial port at a time.

Install dependencies in a dedicated environment:

```bash
cd ~/Github/simulation
git lfs install
git lfs pull --include="logs/legs_tracking/20260914_170827/20260914_170827.onnx" origin
/usr/bin/python3 -m venv .venv-policy
.venv-policy/bin/python -m pip install -r scripts/policy_bench/requirements.txt
```

Run the default 50 Hz, 30-second serial benchmark (460800 baud):

```bash
bash scripts/run_policy_bench.sh
```

Keep the IMU stationary for the initial two-second gyro calibration. Generated
`angles.csv` and `summary.json` are saved under `policy_bench_runs/` and ignored
by Git. `POLICY_OUTPUT_DIR` selects a fresh output folder; `POLICY_PYTHON` selects
another interpreter. Additional arguments override launcher defaults:

```bash
bash scripts/run_policy_bench.sh --source mock --duration 2 --print-hz 0
.venv-policy/bin/python -m unittest discover -s scripts/policy_bench -p 'test_*.py'
```

Mock input verifies inference/timing, not MuJoCo physics. CSV/console writing
is included in this standalone bench's work timing. Rates above 50 Hz and
`--unpaced` are stress/throughput tests, not validated deployment frequencies.
The reference has 299 frames at 50 Hz and holds its last frame after 5.96 s.
IMU samples older than 100 ms stop inference with a failed result. Hardware
mounting is assumed to be identity until calibrated; no real encoder feedback
or actuator dynamics are measured.
