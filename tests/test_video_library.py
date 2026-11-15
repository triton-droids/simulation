from Scripts.index_videos import build_index


def test_index_preserves_originals_and_escapes_links(tmp_path):
    video = tmp_path/'Artifacts/Results/example/videos/a & b.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'fixture, not a playable recording')
    snapshot = tmp_path/'Artifacts/Results/example/source/old.mp4'
    snapshot.parent.mkdir()
    snapshot.write_bytes(b'snapshot fixture')
    before = video.read_bytes()
    rows = build_index(tmp_path)
    assert rows == [{'path':'Artifacts/Results/example/videos/a & b.mp4','bytes':len(before)}]
    page = (tmp_path/'Artifacts/Videos/index.html').read_text(encoding='utf-8')
    assert 'a%20%26%20b.mp4' in page
    assert 'a &amp; b.mp4' in page
    assert video.read_bytes() == before
    assert snapshot.exists()
    assert build_index(tmp_path) == rows
