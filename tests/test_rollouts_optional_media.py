"""Non-video training must not load optional video package data."""
import subprocess
import sys


def test_rollout_import_without_video_dependencies():
    code = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'mediapy', 'moviepy'}:
        raise AssertionError('Unexpected video dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from robocub.tools.rollouts import get_rollout, save_rollout, render_video
assert callable(get_rollout) and callable(save_rollout) and callable(render_video)
"""
    subprocess.run([sys.executable, '-c', code], check=True, timeout=90)


def test_render_loads_video_dependencies(monkeypatch):
    import types
    from robocub.tools.rollouts import render_video
    calls = []
    media = types.ModuleType('mediapy')
    media.write_video = lambda path, frames, fps: calls.append(('write', path, fps))
    movie = types.ModuleType('moviepy')
    movie.VideoFileClip = lambda path: path
    movie.clips_array = lambda clips: types.SimpleNamespace(
        write_videofile=lambda path, logger: calls.append(('combine', path)))
    monkeypatch.setitem(sys.modules, 'mediapy', media)
    monkeypatch.setitem(sys.modules, 'moviepy', movie)
    env = types.SimpleNamespace(dt=.02, render=lambda *args, **kwargs: [])
    render_video(env, [], 'unused')
    assert calls == [('write', 'unused-back.mp4', 25.0),
                     ('write', 'unused-side.mp4', 25.0),
                     ('combine', 'unused-eval.mp4')]
