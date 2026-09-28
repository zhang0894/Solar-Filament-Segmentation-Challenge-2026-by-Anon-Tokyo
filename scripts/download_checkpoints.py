"""Download the published team checkpoint bundle and verify every file."""

import argparse
import hashlib
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from filament.data import ROOT


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_archive(url, target, headers, workers=4):
    """Fetch independent byte ranges, checking coverage before checksum extraction."""
    with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        length = int(r.headers.get("Content-Length", 0))
        if workers == 1 or not length or r.headers.get("Accept-Ranges") != "bytes":
            with target.open("wb") as handle:
                for chunk in r.iter_content(4 * 1024 * 1024):
                    handle.write(chunk)
            return
        # The redirected object URL is signed. Do not forward Kaggle credentials.
        object_url = r.url
    workers = min(workers, length)
    with target.open("wb") as handle:
        handle.truncate(length)

    def fetch(index):
        start = length * index // workers
        end = length * (index + 1) // workers - 1
        for attempt in range(3):
            try:
                with requests.get(
                    object_url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(30, 180),
                ) as response:
                    if response.status_code != 206 or response.headers.get(
                        "Content-Range"
                    ) != f"bytes {start}-{end}/{length}":
                        raise ValueError("Server returned an unexpected byte range")
                    written = 0
                    with target.open("r+b") as output:
                        output.seek(start)
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            written += len(chunk)
                            if written > end - start + 1:
                                raise ValueError("Download exceeded its byte range")
                            output.write(chunk)
                    if written != end - start + 1:
                        raise ValueError("Download ended before its byte range")
                print(json.dumps({"download_segment_complete": index + 1}), flush=True)
                return
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, range(workers)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/checkpoints.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    manifest = json.loads(Path(args.manifest).read_text())
    missing = [x for x in manifest["files"] if not (ROOT / x["path"]).exists()]
    if missing:
        url = manifest.get("download_url")
        if not url:
            raise ValueError("Checkpoint manifest has no published download URL")
        destination = ROOT / "weights/anon_tokyo_bundle.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        headers = {}
        if os.environ.get("KAGGLE_API_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["KAGGLE_API_TOKEN"]
        download_archive(
            url, destination.with_suffix(".partial"), headers, args.workers
        )
        destination.with_suffix(".partial").replace(destination)
        with zipfile.ZipFile(destination) as archive:
            members = {info.filename: info for info in archive.infolist()}
            for record in missing:
                source = record["bundle_filename"]
                if source not in members:
                    raise ValueError(
                        f"Missing checkpoint in published archive: {source}"
                    )
                target = (ROOT / record["path"]).resolve()
                if not target.is_relative_to(ROOT):
                    raise ValueError("Invalid checkpoint destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".partial")
                with archive.open(source) as inp, temporary.open("wb") as output:
                    import shutil

                    shutil.copyfileobj(inp, output, 4 * 1024 * 1024)
                if sha256(temporary) != record["sha256"]:
                    temporary.unlink()
                    raise ValueError(f"Checksum mismatch: {source}")
                temporary.replace(target)
        destination.unlink()
    for record in manifest["files"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Checkpoint differs from release: {record['path']}")
    print(json.dumps({"verified_checkpoints": len(manifest["files"])}))


if __name__ == "__main__":
    main()
