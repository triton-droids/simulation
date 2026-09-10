from pathlib import Path

from scripts import export_repo_zip


def test_export_skips_excluded_tree_before_file_stat(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "source").mkdir()
    included = tmp_path / "source" / "module.py"
    included.write_text("pass\n", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    excluded = tmp_path / ".cache" / "inaccessible-link"
    excluded.touch()
    original_is_file = Path.is_file

    def guarded_is_file(path: Path) -> bool:
        if ".cache" in path.parts:
            raise OSError("excluded path must not be statted")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", guarded_is_file)

    files = export_repo_zip.iter_export_files(tmp_path, tmp_path / "exports/x.zip")

    assert files == [included]


def test_export_excludes_hydra_outputs(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "config.yaml").write_text("generated: true\n")
    source = tmp_path / "README.md"
    source.write_text("project\n")

    files = export_repo_zip.iter_export_files(tmp_path, tmp_path / "exports/x.zip")

    assert files == [source]
