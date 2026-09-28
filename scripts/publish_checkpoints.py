"""Publish an explicit checkpoint allowlist with resumable parallel blob uploads."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="deliverables/AnonTokyo/checkpoint_bundle")
    parser.add_argument(
        "--version-notes", help="Create a new version instead of a dataset"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--receipt", default="deliverables/AnonTokyo/checkpoint_dataset_receipt.json"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    root = Path(__file__).resolve().parents[1]
    if not os.environ.get("KAGGLE_API_TOKEN") and (root / "kaggleAPI.txt").is_file():
        os.environ["KAGGLE_API_TOKEN"] = (root / "kaggleAPI.txt").read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi

    folder = (root / args.folder).resolve()
    # Preserve relative SDK upload identities used by earlier invocations.
    upload_folder = os.path.relpath(folder, Path.cwd())
    manifest = json.loads((folder / "CHECKPOINTS.json").read_text())
    files = [x["bundle_filename"] for x in manifest["files"]]
    files += ["CHECKPOINTS.json", "README.md", "NOTICE.md"]
    if len(set(files)) != len(files) or any(Path(x).name != x for x in files):
        raise ValueError("The checkpoint bundle must contain unique, flat filenames")

    class ParallelKaggleApi(KaggleApi):
        def upload_files(
            self,
            request,
            resources,
            folder,
            blob_type,
            upload_context,
            quiet=False,
            dir_mode="skip",
            ignore_patterns=None,
        ):
            # The SDK creates the dataset/version only after this method returns.
            # Each independent file retains the SDK's resumable upload metadata.
            def upload(filename):
                print(json.dumps({"event": "file_start", "file": filename}), flush=True)
                result = self._upload_file(
                    filename,
                    str(Path(folder) / filename),
                    blob_type,
                    upload_context,
                    True,
                    resources,
                )
                if result is None:
                    raise RuntimeError("Checkpoint upload failed: " + filename)
                print(
                    json.dumps({"event": "file_complete", "file": filename}), flush=True
                )
                return result

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(upload, files))
            request.files.extend(self._new_file(result) for result in results)

    api = ParallelKaggleApi()
    api.authenticate()
    if args.version_notes:
        response = api.dataset_create_version(
            upload_folder,
            args.version_notes,
            quiet=True,
            convert_to_csv=False,
            dir_mode="skip",
        )
    else:
        response = api.dataset_create_new(
            upload_folder, public=True, quiet=True, convert_to_csv=False, dir_mode="skip"
        )
    record = {
        key: str(getattr(response, key, ""))
        for key in ["url", "status", "error", "message", "ref"]
    }
    record.update(
        time_unix=time.time(),
        dataset_ref=manifest["dataset_ref"],
        files=len(manifest["files"]),
    )
    (root / args.receipt).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if getattr(response, "error", None):
        raise RuntimeError("Dataset publication reported an error; inspect the receipt")


if __name__ == "__main__":
    main()
