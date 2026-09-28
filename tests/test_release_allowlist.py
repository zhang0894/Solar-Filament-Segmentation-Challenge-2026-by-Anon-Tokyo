from scripts.build_release import release_files


def test_publication_excludes_local_credentials_and_execution_data(tmp_path):
    for name in [
        "README.md",
        "requirements.txt",
        "scripts/reproduce.py",
        "artifacts/release.json",
        "report/AnonTokyo_Report.pdf",
        "kaggleAPI.txt",
        ".env",
        "data/manifest.json",
        "weights/model.pt",
        "reproduced_submission/model_0/cache.npz",
        "scripts/__pycache__/x.pyc",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    actual = {p.relative_to(tmp_path).as_posix() for p in release_files(tmp_path)}
    assert actual == {
        "README.md",
        "requirements.txt",
        "scripts/reproduce.py",
        "artifacts/release.json",
        "report/AnonTokyo_Report.pdf",
    }
