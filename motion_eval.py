#!/usr/bin/env python3
"""Single-ball ground-plane tracking evaluation. Standard library only."""
import argparse
import bisect
import csv
import html
import json
import math
from pathlib import Path
import random
import statistics

VALID = {"tracking", "coasting"}
STATUSES = VALID | {"tentative", "lost"}


def number(row, key, required=True):
    value = row.get(key, "")
    if value in (None, ""):
        if required:
            raise ValueError(f"missing {key}")
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key}")
    return value


def read_csv(path, ground_truth=False):
    result = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp_s", "present"} if ground_truth else {"timestamp_s", "status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {sorted(required)}")
        for line, row in enumerate(reader, 2):
            try:
                t = number(row, "timestamp_s")
                if result and t <= result[-1]["t"]:
                    raise ValueError("timestamps must be strictly increasing (no duplicates)")
                if ground_truth:
                    if row["present"].strip() not in {"0", "1"}:
                        raise ValueError("present must be 0 or 1")
                    valid = row["present"].strip() == "1"
                    status = None
                else:
                    status = row["status"].strip().lower()
                    if status not in STATUSES:
                        raise ValueError(f"unknown status {status!r}")
                    valid = status in VALID
                item = {"t": t, "valid": valid, "status": status}
                for key in ("x_m", "y_m"):
                    item[key] = number(row, key, required=valid)
                for key in ("vx_mps", "vy_mps"):
                    item[key] = number(row, key, required=False)
                if (item["vx_mps"] is None) != (item["vy_mps"] is None):
                    raise ValueError("provide both velocity components or neither")
                item["published_s"] = number(row, "published_s", required=False)
                if item["published_s"] is not None and item["published_s"] < t:
                    raise ValueError("published_s precedes timestamp_s")
                result.append(item)
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{line}: {e}") from e
    if ground_truth and not result:
        raise ValueError("ground truth is empty")
    return result


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    i = (len(ordered) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (i - lo)


def rmse(values):
    return math.sqrt(statistics.mean(v * v for v in values)) if values else None


def ratio(a, b):
    return a / b if b else None


def score(gt, predictions, tolerance=0.05, threshold=0.2):
    """Sample-based scoring; nearest estimate by state timestamp, never fill gaps."""
    times = [p["t"] for p in predictions]
    details, errors, velocities, latencies, offsets = [], [], [], [], []
    present = available = accurate = absent = false_tracks = 0
    longest_missing = missing_run = 0
    for g in gt:
        j = bisect.bisect_left(times, g["t"])
        candidates = [k for k in (j - 1, j) if 0 <= k < len(times)]
        k = min(candidates, key=lambda k: (abs(times[k] - g["t"]), times[k])) if candidates else None
        p = predictions[k] if k is not None and abs(times[k] - g["t"]) <= tolerance + 1e-9 else None
        usable = p is not None and p["valid"]
        e = ve = None
        if g["valid"]:
            present += 1
            if usable:
                available += 1
                e = math.hypot(p["x_m"] - g["x_m"], p["y_m"] - g["y_m"])
                errors.append(e)
                accurate += e <= threshold
                missing_run = 0
                if g["vx_mps"] is not None and p["vx_mps"] is not None:
                    ve = math.hypot(p["vx_mps"] - g["vx_mps"], p["vy_mps"] - g["vy_mps"])
                    velocities.append(ve)
            else:
                missing_run += 1
                longest_missing = max(longest_missing, missing_run)
        else:
            absent += 1
            false_tracks += usable
            missing_run = 0
        if usable:
            offsets.append(abs(p["t"] - g["t"]))
            if p["published_s"] is not None:
                latencies.append(1000 * (p["published_s"] - p["t"]))
        details.append({
            "timestamp_s": g["t"], "present": int(g["valid"]),
            "matched_timestamp_s": p["t"] if p else None,
            "status": p["status"] if p else "unmatched", "usable": int(usable),
            "gt_x_m": g["x_m"], "gt_y_m": g["y_m"],
            "pred_x_m": p["x_m"] if usable else None,
            "pred_y_m": p["y_m"] if usable else None,
            "position_error_m": e, "velocity_error_mps": ve,
            "accurate": int(e is not None and e <= threshold) if g["valid"] else None,
        })
    metrics = {
        "accuracy_within_threshold": ratio(accurate, present),
        "position_threshold_m": threshold,
        "availability": ratio(available, present),
        "position_rmse_m": rmse(errors),
        "position_median_m": percentile(errors, 0.5),
        "position_p95_m": percentile(errors, 0.95),
        "velocity_rmse_mps": rmse(velocities),
        "false_track_rate_on_absent_samples": ratio(false_tracks, absent),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "alignment_p95_ms": None if not offsets else percentile(offsets, 0.95) * 1000,
        "gt_samples": len(gt), "present_samples": present, "absent_samples": absent,
        "available_present_samples": available, "accurate_present_samples": accurate,
        "false_track_samples": false_tracks, "velocity_samples": len(velocities),
        "latency_samples": len(latencies), "longest_missing_run_samples": longest_missing,
        "prediction_rows": len(predictions),
    }
    return metrics, details


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, percent=False):
    return "N/A" if value is None else (f"{value * 100:.1f}%" if percent else f"{value:.4f}")


def plot(details):
    points = [(d["timestamp_s"], d["position_error_m"]) for d in details if d["position_error_m"] is not None]
    if not points:
        return "<p>No matched position errors to plot.</p>"
    start, end = details[0]["timestamp_s"], details[-1]["timestamp_s"]
    top = max(0.01, max(e for _, e in points))
    # Separate segments at missing/absent samples; never draw across gaps.
    segments, current = [], []
    for d in details:
        e = d["position_error_m"]
        if e is None:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(f'{45 + 700 * (d["timestamp_s"] - start) / max(end - start, 1e-9):.2f},{190 - 160 * e / top:.2f}')
    if current:
        segments.append(current)
    lines = "".join(f'<polyline points="{" ".join(s)}" fill="none" stroke="#2563eb" stroke-width="2"/>' for s in segments)
    return f'<svg viewBox="0 0 780 230" role="img" aria-label="Position error over time"><path d="M45 20V190H750" fill="none" stroke="#888"/>{lines}<text x="45" y="15">Error, meters (top: {top:.3f})</text><text x="45" y="220">Time since start: 0–{end-start:.2f} s; gaps = no scored estimate</text></svg>'


def evaluate(manifest_path, output):
    path = Path(manifest_path).resolve()
    cfg = json.loads(path.read_text())
    if not isinstance(cfg.get("frame"), str) or not cfg["frame"].strip():
        raise ValueError("manifest must declare a shared coordinate frame")
    if not cfg.get("runs"):
        raise ValueError("manifest must include at least one run")
    tolerance = float(cfg.get("time_tolerance_s", 0.05))
    threshold = float(cfg.get("position_threshold_m", 0.2))
    if not math.isfinite(tolerance) or tolerance < 0 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid time tolerance or position threshold")
    gt = read_csv(path.parent / cfg["ground_truth"], ground_truth=True)
    results, all_details, names = [], [], set()
    for run in cfg["runs"]:
        name = run["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("run names must be nonempty and unique")
        names.add(name)
        pred_path = path.parent / run["predictions"]
        sidecar = pred_path.with_suffix(".meta.json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            if meta.get("frame") != cfg["frame"] or not meta.get("ground_plane", True):
                raise ValueError(f"{pred_path}: frame mismatch or optical coordinates; transform into the declared ground frame")
        preds = read_csv(pred_path)
        metrics, details = score(gt, preds, tolerance, threshold)
        results.append({"name": name, **metrics})
        all_details.append(details)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    warnings = [
        "No automatic time-offset or coordinate alignment is performed; inputs must already share frame, units, and clock.",
        "Accuracy includes missing, LOST, and TENTATIVE tracks as failures on present samples. RMSE covers available estimates only.",
        "Absent means the ball does not exist in the evaluation region, NOT merely occluded. Label occluded balls present when position is known.",
        "Scores are per ground-truth sample, not duration-weighted. Use a uniform GT sampling rate; nearest predictions may be reused.",
        "A simulated or synthetic report validates tooling, not real-world tracking performance.",
    ]
    report = {"schema_version": 1, "dataset": cfg.get("dataset", path.stem),
              "synthetic": bool(cfg.get("synthetic", False)), "frame": cfg["frame"],
              "time_tolerance_s": tolerance, "notes": warnings, "runs": results}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(output / "summary.csv", results)
    columns = [("Accuracy", "accuracy_within_threshold", True), ("Availability", "availability", True),
               ("Position RMSE (m)", "position_rmse_m", False), ("Position p95 (m)", "position_p95_m", False),
               ("Velocity RMSE (m/s)", "velocity_rmse_mps", False),
               ("False tracks / absent samples", "false_track_rate_on_absent_samples", True),
               ("Latency p95 (ms)", "latency_p95_ms", False)]
    title = f'{report["dataset"]} — tracking evaluation' + (" [SYNTHETIC]" if report["synthetic"] else "")
    md = [f"# {title}", "", f'Frame: `{cfg["frame"]}`. Accuracy = fraction of present GT samples with a usable estimate within {threshold:g} m.', "",
          "| Run | " + " | ".join(c[0] for c in columns) + " |",
          "|---|" + "---|" * len(columns)]
    body = []
    for i, (r, details) in enumerate(zip(results, all_details)):
        write_csv(output / f"run_{i:02d}_samples.csv", details)
        cells = [fmt(r[k], pct) for _, k, pct in columns]
        md.append("| " + r["name"].replace("|", "\\|").replace("\n", " ") + " | " + " | ".join(cells) + " |")
        body.append("<tr><td>" + html.escape(r["name"]) + "</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    md += ["", "N/A means the required reference/output data is unavailable.", ""] + [f"- {w}" for w in warnings]
    (output / "report.md").write_text("\n".join(md) + "\n")
    heads = "<th>Run</th>" + "".join(f"<th>{html.escape(c[0])}</th>" for c in columns)
    charts = "".join(f"<h2>{html.escape(r['name'])}</h2>{plot(d)}" for r, d in zip(results, all_details))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #ddd}}svg{{width:100%;max-width:850px}}li{{margin:8px 0}}.scroll{{overflow-x:auto}}</style>
<h1>{html.escape(title)}</h1><p>Accuracy: usable position within {threshold:g} m / all present ground-truth samples. Frame: {html.escape(cfg['frame'])}.</p>
<div class="scroll"><table><thead><tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
<p>N/A = required reference/output data unavailable. Lower errors and latency are better; higher accuracy and availability are better.</p>{charts}<h2>Interpretation</h2><ul>{''.join('<li>'+html.escape(w)+'</li>' for w in warnings)}</ul></html>'''
    (output / "report.html").write_text(document)
    print("\n".join(md[:6 + len(results)]))
    print(f"\nReport: {output.resolve() / 'report.html'}")
    return report


def demo(output):
    """Deterministic synthetic fixtures, not fabricated real evaluation data."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    gt, clean, noisy, missing = [], [], [], []
    for i in range(301):
        t = i / 30
        present = not (8 <= t <= 9)
        x, y = 1 + 0.3 * t, 0.4 * math.sin(t)
        vx, vy = 0.3, 0.4 * math.cos(t)
        gt.append(dict(timestamp_s=t, present=int(present), x_m=x if present else "", y_m=y if present else "", vx_mps=vx if present else "", vy_mps=vy if present else ""))
        for rows, sigma, drop in ((clean, 0.03, False), (noisy, 0.2, False), (missing, 0.03, i % 5 != 0)):
            status = "lost" if drop or (not present and rows is not noisy) else "tracking"
            rows.append(dict(timestamp_s=t, status=status, x_m=x + rng.gauss(0, sigma), y_m=y + rng.gauss(0, sigma), vx_mps=vx + rng.gauss(0, sigma), vy_mps=vy + rng.gauss(0, sigma), published_s=t + 0.025))
    write_csv(root / "ground_truth.csv", gt)
    runs = []
    for name, rows in (("accurate", clean), ("noisy_false_tracks", noisy), ("accurate_but_missing", missing)):
        write_csv(root / f"{name}.csv", rows)
        runs.append({"name": name, "predictions": f"{name}.csv"})
    manifest = {"dataset": "Synthetic smoke test", "synthetic": True, "frame": "odom", "ground_truth": "ground_truth.csv", "time_tolerance_s": 0.02, "position_threshold_m": 0.2, "runs": runs}
    path = root / "evaluation.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    evaluate(path, root / "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/latest"))
    p = sub.add_parser("demo")
    p.add_argument("--out", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    try:
        if args.command == "demo":
            demo(args.out)
        else:
            evaluate(args.manifest, args.out)
    except (ValueError, KeyError, OSError, TypeError) as e:
        parser.exit(2, f"Evaluation failed: {e}\n")


if __name__ == "__main__":
    main()
