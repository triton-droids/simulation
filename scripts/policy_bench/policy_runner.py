#!/usr/bin/env python3
"""Run the bundled walking ONNX on a bench, with no motor transport."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np
import onnx
import onnxruntime as ort
import serial

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'logs/legs_tracking/20260914_170827/20260914_170827.onnx'


def parse_imu(line, fmt):
    """Return (acceleration m/s², angular velocity rad/s, optional device us)."""
    try:
        if fmt == 'json-si':
            data = json.loads(line)
            acc = np.asarray(data['accel_mps2'], dtype=np.float64)
            gyro = np.asarray(data['gyro_rad_s'], dtype=np.float64)
            device_us = int(data['t_us']) if 't_us' in data else None
        elif fmt == 'csv-g-dps':
            values = [float(v) for v in line.strip().split(',')]
            if len(values) != 7:
                return None
            device_us = int(values[0] * 1000)
            acc = np.asarray(values[1:4]) * 9.80665
            gyro = np.deg2rad(values[4:7])
        else:
            values = dict(re.findall(r'\b([ag][xyz])\s*=\s*(-?\d+)', line))
            acc = np.asarray([float(values[k]) for k in ('ax', 'ay', 'az')]) / 16384 * 9.80665
            gyro = np.deg2rad(np.asarray([float(values[k]) for k in ('gx', 'gy', 'gz')]) / 131)
            device_us = None
        if acc.shape != (3,) or gyro.shape != (3,):
            return None
        if not np.isfinite(acc).all() or not np.isfinite(gyro).all():
            return None
        if not 0.1 < np.linalg.norm(acc) < 100:
            return None
        return acc, gyro, device_us
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


class GravityFilter:
    """Gyro propagation plus low-pass specific-force correction; bench estimator."""
    def __init__(self):
        self.gravity = None
        self.last_ns = None

    def update(self, acc, omega, now_ns):
        measured = -acc / np.linalg.norm(acc)
        if self.gravity is None:
            self.gravity = measured.copy()
        else:
            dt = (now_ns - self.last_ns) * 1e-9
            if 0 < dt < 0.2:
                self.gravity -= np.cross(omega, self.gravity) * dt
                if 0.8 * 9.80665 < np.linalg.norm(acc) < 1.2 * 9.80665:
                    alpha = 1 - math.exp(-dt / 0.5)
                    self.gravity = (1-alpha) * self.gravity + alpha * measured
                self.gravity /= np.linalg.norm(self.gravity)
            else:
                self.gravity = measured.copy()
        self.last_ns = now_ns
        return self.gravity.copy()


class SerialImu:
    def __init__(self, port, baud, fmt, rotation, calibration_seconds):
        self.port, self.baud, self.fmt = port, baud, fmt
        self.rotation = rotation
        self.calibration_seconds = calibration_seconds
        self.bias = np.zeros(3)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None
        self.valid = self.invalid = 0
        self.lines = []
        self.error = None
        self.arrival_intervals = []
        self.device_intervals = []
        self.calibration = {}
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()
        deadline = time.monotonic() + 8 + self.calibration_seconds
        while time.monotonic() < deadline:
            with self.lock:
                if self.latest is not None:
                    return
                if self.error:
                    raise RuntimeError(self.error)
            time.sleep(0.02)
        self.close()
        raise RuntimeError(f'No calibrated IMU samples from {self.port}; recent lines: {self.lines[-4:]}')

    def _read(self):
        calibration = []
        first_ns = last_ns = last_device_us = None
        calibrated = self.calibration_seconds == 0
        filt = GravityFilter()
        try:
            with serial.Serial(self.port, self.baud, timeout=0.05, write_timeout=0.05) as ser:
                ser.reset_input_buffer()
                while not self.stop_event.is_set():
                    raw = ser.readline(1024)
                    if not raw:
                        continue
                    now_ns = time.perf_counter_ns()
                    line = raw.decode('utf-8', errors='replace').strip()
                    parsed = parse_imu(line, self.fmt)
                    if parsed is None:
                        self.invalid += 1
                        self.lines = (self.lines + [line])[-8:]
                        continue
                    acc, gyro, device_us = parsed
                    acc = self.rotation @ acc
                    gyro = self.rotation @ gyro
                    self.valid += 1
                    if last_ns is not None:
                        self.arrival_intervals.append((now_ns-last_ns)*1e-6)
                    if device_us is not None and last_device_us is not None:
                        delta = (device_us-last_device_us) % (2**32)
                        if delta < 1_000_000:
                            self.device_intervals.append(delta/1000)
                    last_ns, last_device_us = now_ns, device_us
                    if not calibrated:
                        first_ns = now_ns if first_ns is None else first_ns
                        calibration.append((acc, gyro))
                        if (now_ns-first_ns)*1e-9 < self.calibration_seconds:
                            continue
                        accelerations = np.asarray([v[0] for v in calibration])
                        velocities = np.asarray([v[1] for v in calibration])
                        if len(calibration) < 20 or np.max(velocities.std(axis=0)) > 0.03 or np.linalg.norm(velocities.mean(axis=0)) > 0.15 or not np.all((np.linalg.norm(accelerations, axis=1) > 8.8) & (np.linalg.norm(accelerations, axis=1) < 10.8)):
                            raise RuntimeError('IMU must stay stationary during startup calibration; check units and wiring')
                        self.bias = velocities.mean(axis=0)
                        self.calibration = {'samples': len(calibration), 'gyro_bias_rad_s': self.bias.tolist(), 'gyro_std_rad_s': velocities.std(axis=0).tolist()}
                        calibrated = True
                    omega = gyro-self.bias
                    gravity = filt.update(acc, omega, now_ns)
                    with self.lock:
                        self.latest = (now_ns, self.valid, omega, gravity)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.latest

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)


class Policy:
    def __init__(self, model, threads=1):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        inputs = {v.name: (v.shape, v.type) for v in self.session.get_inputs()}
        if inputs != {'obs': ([1, 56], 'tensor(float)'), 'time_step': ([1, 1], 'tensor(float)')}:
            raise ValueError(f'Unexpected policy interface: {inputs}')
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        expected = 'command,base_ang_vel,joint_pos,joint_vel,actions,gravity'
        if self.metadata.get('observation_names') != expected:
            raise ValueError('Observation order does not match this runner')
        self.names = self.metadata['joint_names'].split(',')
        self.offset = np.fromstring(self.metadata['default_joint_pos'], sep=',', dtype=np.float32)
        self.scale = np.fromstring(self.metadata['action_scale'], sep=',', dtype=np.float32)
        tensors = {t.name: onnx.numpy_helper.to_array(t) for t in onnx.load(str(model)).graph.initializer}
        self.reference_q = tensors['joint_pos.1']
        self.reference_dq = tensors['joint_vel.1']
        if self.reference_q.shape != self.reference_dq.shape or self.reference_q.shape[1] != 10 or len(self.names) != 10:
            raise ValueError('Unexpected joint reference dimensions')
        self.frames = len(self.reference_q)

    def observation(self, frame, omega, gravity, q, dq, previous):
        obs = np.concatenate((self.reference_q[frame], self.reference_dq[frame], omega, q-self.offset, dq, previous, gravity)).astype(np.float32)[None, :]
        if obs.shape != (1, 56) or not np.isfinite(obs).all():
            raise ValueError('Invalid observation')
        return obs

    def run(self, obs, frame):
        result = self.session.run(['actions'], {'obs': obs, 'time_step': np.asarray([[frame]], dtype=np.float32)})[0][0]
        if result.shape != (10,) or not np.isfinite(result).all():
            raise ValueError('Non-finite or incorrectly shaped policy output')
        return result, self.offset + self.scale * result


def stats(values):
    if not values:
        return None
    a = np.asarray(values)
    return {'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99)), 'max': float(a.max())}


def sleep_until(deadline_ns):
    remaining = (deadline_ns-time.perf_counter_ns())*1e-9
    if remaining > 0:
        time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--source', choices=('mock', 'serial'), default='serial')
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--baud', type=int, default=460800)
    parser.add_argument('--imu-format', choices=('json-si', 'csv-g-dps', 'mpu6050-raw'), default='json-si')
    parser.add_argument('--rotation', type=float, nargs=9, default=[1,0,0,0,1,0,0,0,1], help='row-major proper rotation: sensor coordinates to robot coordinates')
    parser.add_argument('--calibration-seconds', type=float, default=2)
    parser.add_argument('--max-imu-age-ms', type=float, default=100)
    parser.add_argument('--joint-state', choices=('zero', 'reference'), default='zero', help='Explicit placeholders; no encoder or motor feedback')
    parser.add_argument('--hz', type=float, default=50)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--unpaced', action='store_true', help='Inference/loop throughput, not physical control frequency')
    parser.add_argument('--print-hz', type=float, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not all(np.isfinite(v) for v in (args.hz,args.duration,args.max_imu_age_ms,args.calibration_seconds,args.print_hz)) or args.hz <= 0 or args.duration <= 0 or args.max_imu_age_ms <= 0 or args.calibration_seconds < 0 or args.print_hz < 0 or args.threads < 1 or args.warmup < 0:
        parser.error('Invalid timing or thread argument')
    if args.out.exists():
        parser.error('Output directory already exists; choose a fresh directory')
    args.out.mkdir(parents=True)
    rotation = np.asarray(args.rotation).reshape(3,3)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation),1):
        parser.error('--rotation must be an orthonormal rotation with determinant +1')
    policy = Policy(args.model, args.threads)
    previous = np.zeros(10, dtype=np.float32)
    mock_omega, mock_gravity = np.zeros(3), np.array([0,0,-1])
    # Warmup includes varied valid reference commands; reset previous actions afterwards.
    for i in range(args.warmup):
        frame = i % policy.frames
        obs = policy.observation(frame, mock_omega, mock_gravity, policy.offset, np.zeros(10), previous)
        previous, _ = policy.run(obs, frame)
    previous[:] = 0
    summary = {'status': 'starting', 'source': args.source, 'motor_connected': False, 'joint_feedback': args.joint_state + '_placeholder', 'requested_hz': args.hz, 'unpaced': args.unpaced, 'reference_hz': 50, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(), 'model': str(args.model.resolve()), 'providers': policy.session.get_providers(), 'threads': args.threads, 'joint_names': policy.names, 'metadata': policy.metadata, 'imu_rotation': rotation.tolist(), 'note': 'Bench timing only; no measured joint feedback or motor command transport. Identity IMU mounting is unverified. Rates above 50 Hz are stress tests; reference advances at 50 Hz.'}
    imu = None
    samples = []
    count = skipped = reused = 0
    peak = np.zeros(10)
    last_seq = None
    started_ns = stopped_ns = None
    error = None
    try:
        if args.source == 'serial':
            imu = SerialImu(args.port,args.baud,args.imu_format,rotation,args.calibration_seconds)
            print('Keep IMU stationary during startup calibration.', flush=True)
            imu.start()
        period_ns = round(1e9/args.hz)
        started_ns = time.perf_counter_ns()
        deadline_ns = started_ns
        print_due = started_ns
        with (args.out/'angles.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_s','frame','imu_seq','imu_age_ms','start_lateness_ms','inference_ms','gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s','gravity_x','gravity_y','gravity_z']+[n+'_target_rad' for n in policy.names]+[n+'_target_deg' for n in policy.names])
            while time.perf_counter_ns()-started_ns < args.duration*1e9:
                if not args.unpaced:
                    sleep_until(deadline_ns)
                begin_ns = time.perf_counter_ns()
                elapsed = (begin_ns-started_ns)*1e-9
                if elapsed >= args.duration:
                    break
                if imu:
                    stamp_ns, seq, omega, gravity = imu.snapshot()
                    age_ms = (begin_ns-stamp_ns)*1e-6
                    if age_ms > args.max_imu_age_ms:
                        raise RuntimeError(f'IMU stale: {age_ms:.1f} ms; inference stopped')
                    reused += int(seq == last_seq)
                    last_seq = seq
                else:
                    omega, gravity, seq, age_ms = mock_omega, mock_gravity, -1, 0
                frame = min(int(elapsed*50), policy.frames-1)
                # Hold at the final reference frame instead of introducing an untrained wrap jump.
                q = policy.offset if args.joint_state == 'zero' else policy.reference_q[frame]
                dq = np.zeros(10) if args.joint_state == 'zero' else policy.reference_dq[frame]
                obs = policy.observation(frame,omega,gravity,q,dq,previous)
                inference_start = time.perf_counter_ns()
                actions, targets = policy.run(obs,frame)
                inference_ms = (time.perf_counter_ns()-inference_start)*1e-6
                previous = actions
                peak = np.maximum(peak,np.abs(targets))
                lateness_ms = max(0,(begin_ns-deadline_ns)*1e-6) if not args.unpaced else 0
                degrees = np.rad2deg(targets)
                writer.writerow([elapsed,frame,seq,age_ms,lateness_ms,inference_ms,*omega,*gravity,*targets,*degrees])
                if args.print_hz and begin_ns >= print_due:
                    print(json.dumps({'elapsed_s': round(elapsed,3), 'imu_age_ms': round(age_ms,2), 'target_deg': dict(zip(policy.names,np.round(degrees,3).tolist()))}), flush=True)
                    print_due = begin_ns + round(1e9/args.print_hz)
                end_ns = time.perf_counter_ns()
                samples.append((begin_ns, inference_ms, (end_ns-begin_ns)*1e-6, lateness_ms, age_ms, int(not args.unpaced and end_ns > deadline_ns+period_ns)))
                count += 1
                if not args.unpaced:
                    deadline_ns += period_ns
                    if end_ns > deadline_ns:
                        missed = (end_ns-deadline_ns)//period_ns+1
                        skipped += missed
                        deadline_ns += missed*period_ns
            stopped_ns = time.perf_counter_ns()
        summary['status'] = 'completed'
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
    except Exception as exc:
        error = str(exc)
        summary['status'] = 'failed'
        summary['error'] = error
    finally:
        if imu:
            imu.close()
            summary['imu'] = {'port':args.port,'format':args.imu_format,'valid_samples':imu.valid,'invalid_lines':imu.invalid,'calibration':imu.calibration,'arrival_interval_ms':stats(imu.arrival_intervals),'device_interval_ms':stats(imu.device_intervals),'recent_unparsed_lines':imu.lines[-4:]}
        if started_ns:
            stopped_ns = stopped_ns or time.perf_counter_ns()
            elapsed_s = (stopped_ns-started_ns)*1e-9
            begins = [s[0] for s in samples]
            summary.update({'iterations':count,'elapsed_s':elapsed_s,'achieved_loop_hz': count/elapsed_s,'inference_ms':stats([s[1] for s in samples]),'work_ms_including_csv_and_print':stats([s[2] for s in samples]),'start_interval_ms':stats((np.diff(begins)*1e-6).tolist()),'start_lateness_ms':stats([s[3] for s in samples]),'imu_age_ms':stats([s[4] for s in samples]),'deadline_misses':sum(s[5] for s in samples),'skipped_slots':int(skipped),'imu_sample_reuses':reused,'target_peak_abs_deg':np.rad2deg(peak).tolist()})
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
