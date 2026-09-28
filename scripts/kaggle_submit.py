"""Authenticated Kaggle submission with local provenance and no token logging."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file")
    p.add_argument("--message", default="")
    p.add_argument("--status", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not os.environ.get("KAGGLE_API_TOKEN"):
        token_path = root / "kaggleAPI.txt"
        if token_path.exists():
            os.environ["KAGGLE_API_TOKEN"] = token_path.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    competition = "filament-segmentation-2026"
    if args.status:
        submissions = api.competition_submissions(competition, page_size=10)
        for entry in submissions or []:
            # Whitelist fields, never dump API authentication or request objects.
            print(
                json.dumps(
                    {
                        key: str(getattr(entry, key, ""))
                        for key in [
                            "ref",
                            "date",
                            "description",
                            "status",
                            "public_score",
                            "private_score",
                            "error_description",
                        ]
                    }
                )
            )
        return
    if not args.file or not args.message:
        p.error("--file and --message are required to submit")
    path = Path(args.file).resolve()
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["csv_sha256"]:
        raise ValueError("CSV has changed since serialization validation")
    for directory in [root / "submissions", root / "runs"]:
        for previous_path in directory.glob("**/submission_receipt.json"):
            previous = json.loads(previous_path.read_text())
            ref = str(previous.get("response", {}).get("ref", ""))
            if previous.get("sha256") == digest and ref.isdigit():
                print(
                    json.dumps(
                        {
                            "skipped_duplicate": True,
                            "existing_ref": ref,
                            "sha256": digest,
                        }
                    )
                )
                return
    response = api.competition_submit(str(path), args.message, competition, quiet=True)
    record = {
        "file": str(path),
        "sha256": digest,
        "message": args.message,
        "response": {
            key: str(getattr(response, key, ""))
            for key in ["message", "ref", "submission_id", "error"]
        },
    }
    (path.parent / "submission_receipt.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
