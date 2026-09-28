"""Archive an explicit source-file allowlist beside a submission, excluding secrets."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from filament.data import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "source_snapshot.zip"
    if archive.exists():
        raise ValueError("Refusing to replace an existing submission source snapshot")
    files = []
    for folder in ["filament", "scripts", "tests"]:
        files.extend((ROOT / folder).rglob("*.py"))
    files.extend((ROOT / "configs").glob("*.json"))
    files.extend(
        ROOT / name
        for name in [
            "README.md",
            "requirements.lock.txt",
            ".gitignore",
            "ruff.toml",
            ".ruff.toml",
        ]
        if (ROOT / name).is_file()
    )
    manifest = {}
    temporary = out / "source_snapshot.partial.zip"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(set(files)):
            relative = file.relative_to(ROOT).as_posix()
            if file.suffix == ".json" and any(
                word in file.name.lower() for word in ["token", "kaggle", "credential"]
            ):
                raise ValueError(
                    "A credential-like filename appeared in the source allowlist"
                )
            content = file.read_bytes()
            z.writestr(relative, content)
            manifest[relative] = hashlib.sha256(content).hexdigest()
    temporary.replace(archive)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "source_manifest.json").write_text(
        json.dumps({"files": manifest, "archive_sha256": archive_hash}, indent=2)
    )
    metadata_path = out / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["source_snapshot_sha256"] = archive_hash
        metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"source_files": len(manifest), "archive_sha256": archive_hash}))


if __name__ == "__main__":
    main()
