# Ball motion-tracking evaluation

Compare single-ball position/tracking outputs from td_perception, Triton, or any
system exporting the CSV contract below. No real dataset is included. The
evaluator uses Python 3.10+ and the standard library; no installation or ROS is
needed for scoring. This evaluates horizontal ball motion, not detector mAP,
multi-object identity tracking, airborne 3D trajectories, or robot behavior.

## Try it now (synthetic only)

```sh
python3 motion_eval.py demo
python3 -m unittest -v
```

Open `demo_output/report/report.html`. It compares accurate, noisy/false-track,
and intermittent synthetic estimates of the same trajectory. These are generated
fixtures, not actual models or evidence of real-world accuracy.

## Evaluate your outputs later

Copy `examples/evaluation.json` beside your CSV files, fill in file paths and the
shared ground coordinate frame, then run:

```sh
python3 motion_eval.py evaluate /path/to/evaluation.json --out reports/comparison
```

Outputs: `report.html` (table and error-over-time charts), `report.md`,
`results.json`, `summary.csv`, and `run_XX_samples.csv` with alignment and errors.
Input paths are relative to the manifest. Names are arbitrary; add more runs to
compare a model, filter, depth method, or feature on the SAME reference sequence.

### Ground truth CSV

```csv
timestamp_s,present,x_m,y_m,vx_mps,vy_mps
0.000,1,3.00,1.00,0.30,0.00
0.033,1,3.01,1.00,0.30,0.00
```

Timestamps: seconds on the same clock as predictions. Coordinates: meters in the
manifest's ground frame, X forward/Y left. Velocity is optional; leave BOTH
components blank if unknown. Do not invent velocity truth by assuming the
tracker's own estimates are correct. For a stationary ball, repeat its measured
position on a uniform timeline and use zero velocity. No ball in the evaluation
region: `present=0`, positions may be blank. Occluded but existing ball: keep
`present=1` if its location is known. Omit unlabeled times instead of labeling
them absent. Accuracy needs present samples; false-track scoring needs absent
samples. Ground truth must be independent measurements or simulation truth.

### Predictions CSV

```csv
timestamp_s,status,x_m,y_m,vx_mps,vy_mps,published_s
0.000,tracking,3.04,1.01,0.28,0.00,0.025
0.033,coasting,3.05,1.01,0.28,0.00,0.058
0.066,lost,,,,,
```

Statuses: `tracking`, `coasting`, `tentative`, `lost`. Only the first two are
usable. Missing output rows also count as unavailable. Velocities and
`published_s` are optional. For latency, published_s must be actual publication
time on the SAME clock; state timestamp must represent the corresponding state
time. Bag receipt time is NOT publication time. For coasting, state-age latency
does not mean latency since the last actual camera observation.

All timestamps must increase strictly. Nonfinite values and malformed rows are
rejected. No interpolation, guessed time offset, or automatic trajectory fitting
is applied. Each GT timestamp selects the nearest output within time_tolerance_s
(ties select earlier output). The same output can match adjacent GT samples;
choose a small tolerance based on known timing and report it consistently.
Fast balls can move significantly during a timestamp mismatch. Sample GT
uniformly: scores are sample-based, not duration-weighted.

## What the score means

**Accuracy within threshold** = GT-present samples with a usable position within
the chosen threshold / ALL GT-present samples. Default threshold: 0.20 m. Missing
or invalid tracks reduce this score, preventing selective output from looking
artificially accurate. The threshold is task-specific, not a universal standard.

Also reported:

- Availability: fraction of GT-present samples with a usable matched output.
- Horizontal position RMSE, median, p95, on available present samples only.
- Horizontal velocity RMSE when independent GT and predicted velocity exist.
- False-track rate: usable outputs at GT-absent samples / absent samples.
- p50/p95 latency when supplied, alignment offsets, longest missing sample run,
  and denominators. Missing metrics are N/A, never silently zero.

False-track sample rate is not false-alarm events/minute. No velocity is inferred
from noisy finite differences. No automatic winner is chosen: compare accuracy,
false tracks, latency, and failure scenarios together. This release does not
measure reacquisition events or kick reaction lag; those need event annotations.

## Compatibility with the existing ROS2 repos

The core evaluator reads neutral CSV so it remains usable as both repos evolve.
An optional offline exporter reads existing ROS2 messages:

```sh
# In the ROS2 environment with td_msgs built and sourced:
python3 export_rosbag.py /path/to/bag --kind td --out td.csv

# In a ROS2 environment; no custom message package needed:
python3 export_rosbag.py /path/to/bag --kind triton --out triton.csv \
  --transform camera_to_ground.json
```

Use `--storage-id mcap` if autodetection fails and the installed ROS2 has the MCAP
plugin. Override topics with `--topic`. The exporter uses header timestamps and
does NOT mislabel bag receipt delay as processing latency.

| Source | Adapter |
|---|---|
| td_perception `/ball/state`, `td_msgs/msg/BallState` | Position, velocity, numeric track statuses |
| Triton `/vision/ball_3d_position`, `geometry_msgs/msg/PointStamped` | Position only; each published point is usable; missing messages become gaps |

The exporter writes a `.meta.json` sidecar; scoring checks its frame. Do not
delete it to bypass a mismatch. CSV without sidecars relies on the user's frame
declaration. Both predictions and truth must describe the SAME physical point:
td estimates ball center, while Triton's depth sample may describe a visible
surface. Resolve that difference before interpreting the comparison.

Triton's optical XYZ MUST be transformed to ground coordinates before scoring
XY. For a fixed camera/rig, pass JSON with `source_frame`, `target_frame`, and a
calibrated 4x4 rigid `matrix` (source→target). Example STRUCTURE only:

```json
{
  "source_frame": "camera_color_optical_frame",
  "target_frame": "base_footprint",
  "matrix": [[0,0,1,0],[-1,0,0,0],[0,-1,0,1.2],[0,0,0,1]]
}
```

That illustrative matrix assumes a forward-facing, untilted camera 1.2 m above
the origin; replace it with measured calibration. A moving head or robot needs
timestamped TF per sample; this exporter supports FIXED transforms only. Do not
apply a fixed transform to moving-camera data. td outputs base_footprint; use
the stationary rig initially. On a moving robot, GT must be expressed in the
same time-varying frame, and velocity GT must mean physical ground velocity
expressed in that frame, not the derivative of robot-relative position.

For future data collection: record source images, CameraInfo, TF, and outputs;
include aligned depth if testing it. The evaluator scores the output states; it
does not launch incomplete detector nodes or run YOLO on recordings. Obtain each
candidate's CSV from a separate replay of the SAME recording. Fit parameters on
validation recordings and freeze them before held-out testing. Compare complete
sessions, not random neighboring frames. Real deployment speed needs an on-device
measurement; fast offline replay does not establish Jetson latency.
