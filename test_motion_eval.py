import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from motion_eval import demo, evaluate, read_csv, score
from export_rosbag import convert_message, validate_transform


def state(t, x=0.0, valid=True):
    return {"t": t, "valid": valid, "status": "tracking" if valid else "lost", "x_m": x,
            "y_m": 0.0, "vx_mps": 0.0, "vy_mps": 0.0, "published_s": None}


class MetricsTests(unittest.TestCase):
    def test_known_error_and_velocity(self):
        g, p = state(0), state(0, 3)
        p["y_m"], p["vx_mps"], p["published_s"] = 4, 2, 0.04
        m, _ = score([g], [p])
        self.assertEqual(m["position_rmse_m"], 5)
        self.assertEqual(m["velocity_rmse_mps"], 2)
        self.assertEqual(m["latency_p95_ms"], 40)

    def test_dropouts_count_against_accuracy(self):
        gt = [state(i) for i in range(10)]
        m, _ = score(gt, [state(0)])
        self.assertEqual(m["accuracy_within_threshold"], 0.1)
        self.assertEqual(m["position_rmse_m"], 0)
        self.assertEqual(m["longest_missing_run_samples"], 9)

    def test_absence_distinct_from_occlusion(self):
        m, _ = score([state(0, valid=False), state(1)], [state(0), state(1)])
        self.assertEqual(m["false_track_rate_on_absent_samples"], 1)
        self.assertEqual(m["accuracy_within_threshold"], 1)

    def test_lost_and_tentative_are_not_usable(self):
        p = state(0)
        p["status"], p["valid"] = "tentative", False
        m, _ = score([state(0)], [p])
        self.assertEqual(m["accuracy_within_threshold"], 0)
        self.assertIsNone(m["position_rmse_m"])

    def test_alignment_does_not_bridge_gaps(self):
        m, _ = score([state(0.2)], [state(0)], tolerance=0.05)
        self.assertEqual(m["availability"], 0)

    def test_empty_predictions(self):
        m, _ = score([state(0)], [])
        self.assertEqual(m["accuracy_within_threshold"], 0)

    def test_duplicate_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.csv"
            for text in ("0,tracking,0,0\n0,tracking,0,0\n", "0,tracking,nan,0\n"):
                p.write_text("timestamp_s,status,x_m,y_m\n" + text)
                with self.assertRaises(ValueError):
                    read_csv(p)

    def test_demo_and_html_report(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            demo(d)
            p = Path(d)
            r = json.loads((p / "report/results.json").read_text())
            a, b, c = r["runs"]
            self.assertTrue(r["synthetic"])
            self.assertGreater(a["accuracy_within_threshold"], b["accuracy_within_threshold"])
            self.assertGreater(a["accuracy_within_threshold"], c["accuracy_within_threshold"])
            self.assertIn("<svg", (p / "report/report.html").read_text())
            (p / "accurate.meta.json").write_text('{"frame":"camera","ground_plane":false}')
            with self.assertRaises(ValueError):
                evaluate(p / "evaluation.json", p / "other")

    def test_adapters_and_rigid_transform(self):
        transform = {"source_frame": "camera", "target_frame": "odom", "matrix":
                     [[0,0,1,0],[-1,0,0,0],[0,-1,0,1],[0,0,0,1]]}
        m = validate_transform(transform)
        msg = NS(header=NS(stamp=NS(sec=1,nanosec=500000000)), point=NS(x=2,y=3,z=4))
        row = convert_message(msg, "triton", m)
        self.assertEqual((row["timestamp_s"],row["x_m"],row["y_m"]), (1.5,4,-2))
        self.assertEqual(row["vx_mps"], "")
        msg.position, msg.velocity, msg.status = msg.point, NS(x=1,y=0,z=0), 2
        self.assertEqual(convert_message(msg,"td")["status"], "coasting")


if __name__ == "__main__":
    unittest.main()
