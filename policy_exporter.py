"""Video conversion utilities for policy visualization"""

import cv2
from PIL import Image


def save_as_gif(frames, output_path, fps):
    """
    Convert frames to GIF format

    Args:
        frames: List of numpy arrays (RGB format)
        output_path: Path to save the GIF file
        fps: Frames per second
    """
    print(f"Saving {output_path}...")
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),  # milliseconds per frame
        loop=0
    )
    print(f"Saved {output_path}")


def save_as_mp4(frames, output_path, fps):
    """
    Convert frames to MP4 format

    Args:
        frames: List of numpy arrays (RGB format)
        output_path: Path to save the MP4 file
        fps: Frames per second
    """
    print(f"Converting to {output_path}...")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"✓ Saved {output_path}")
