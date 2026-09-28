"""Download the competition archive without exposing local credentials."""

import json
import os
import time
import zipfile
from pathlib import Path

import requests


def main():
    root = Path(__file__).resolve().parents[1]
    credential = root / "kaggleAPI.txt"
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token and credential.exists():
        token = credential.read_text().strip()
    if not token:
        raise RuntimeError(
            "Set KAGGLE_API_TOKEN or supply the competition data locally after accepting its rules."
        )
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + token
    data = root / "data"
    data.mkdir(exist_ok=True)
    archive = data / "filament-segmentation-2026.zip"
    if not archive.exists():
        url = "https://www.kaggle.com/api/v1/competitions/data/download-all/filament-segmentation-2026"
        start = time.monotonic()
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Kaggle download returned HTTP {response.status_code}"
                )
            size = int(response.headers.get("Content-Length", 0))
            print(json.dumps({"event": "download_start", "bytes": size}), flush=True)
            temporary = archive.with_suffix(".partial")
            written = 0
            last = start
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    handle.write(chunk)
                    written += len(chunk)
                    if time.monotonic() - last > 10:
                        print(
                            json.dumps(
                                {
                                    "downloaded_MB": round(written / 1e6),
                                    "seconds": round(time.monotonic() - start),
                                }
                            ),
                            flush=True,
                        )
                        last = time.monotonic()
            if not zipfile.is_zipfile(temporary):
                raise RuntimeError("Download was not a ZIP archive")
            temporary.replace(archive)
    with zipfile.ZipFile(archive) as handle:
        for item in handle.infolist():
            dest = (data / item.filename).resolve()
            if not dest.is_relative_to(data.resolve()):
                raise ValueError("Unsafe archive member")
        handle.extractall(data)
        print(
            json.dumps(
                {
                    "event": "extracted",
                    "files": len(handle.infolist()),
                    "archive_MB": round(archive.stat().st_size / 1e6),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
