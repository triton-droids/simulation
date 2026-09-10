import os
from pathlib import Path
import sys

import cv2
import numpy as np

if sys.platform == "win32":
    os.environ["MUJOCO_GL"] = "glfw"

from source.scripts import evaluate_g1


def test_git_record_is_explicit_when_export_has_no_git(monkeypatch) -> None:
    def no_repository(*_args, **_kwargs):
        raise evaluate_g1.subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(evaluate_g1.subprocess, "run", no_repository)

    assert evaluate_g1._git_record() == {
        "available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
    }


def test_video_falls_back_to_opencv_when_ffmpeg_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    frames = [
        np.full((48, 64, 3), fill_value, dtype=np.uint8)
        for fill_value in (0, 80, 160)
    ]

    def missing_ffmpeg(*_args, **_kwargs) -> None:
        raise RuntimeError("Program 'ffmpeg' is not found")

    monkeypatch.setattr(evaluate_g1.media, "write_video", missing_ffmpeg)
    output = tmp_path / "fallback.mp4"

    class FakeEnvironment:
        dt = 0.02

        @staticmethod
        def render(*_args, **_kwargs):
            return frames

    backend = evaluate_g1._write_video(
        FakeEnvironment(), object(), {"pipeline_state": []}, 0, output, 2
    )

    assert backend == "opencv_mp4v"
    assert output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
    finally:
        capture.release()
