"""Reproduce the released prediction file from official images and checkpoints."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from filament.data import ROOT


def run(module, *args):
    subprocess.run(
        [sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True
    )


def validate_test_images(directory, manifest):
    expected = {record["filename"] for record in manifest["files"]}
    if {p.name for p in directory.glob("*.jpeg")} != expected:
        raise ValueError("Test image filenames differ from the released input set")
    for record in manifest["files"]:
        with (directory / record["filename"]).open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if sha != record["sha256"]:
            raise ValueError("Test image content differs: " + record["filename"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Official MAGFiLO_1.0_Kaggle_2026 directory")
    parser.add_argument("--release", default="artifacts/release.json")
    parser.add_argument("--output", default="reproduced_submission")
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    release = json.loads((ROOT / args.release).read_text())
    data = ROOT / "data/MAGFiLO_1.0_Kaggle_2026"
    if args.data_dir:
        source = Path(args.data_dir).resolve()
        if not (source / "test/test_images").is_dir():
            raise ValueError("data-dir must contain test/test_images and train/")
        data.parent.mkdir(exist_ok=True)
        if data.exists() and data.resolve() != source:
            raise ValueError("Existing official data directory points elsewhere")
        if not data.exists():
            data.symlink_to(source, target_is_directory=True)
    if not data.is_dir():
        raise ValueError("Download official competition data or pass --data-dir")
    manifest = ROOT / "data/manifest.json"
    if not manifest.exists():
        run("scripts.prepare_data", "--size", 1024, "--workers", 4)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != release["manifest_sha256"]:
        raise ValueError("Prepared manifest differs from the released training split")
    test_manifest = ROOT / "artifacts/test_images.json"
    if (
        hashlib.sha256(test_manifest.read_bytes()).hexdigest()
        != release["test_manifest_sha256"]
    ):
        raise ValueError("Test input manifest differs from the released artifact")
    validate_test_images(
        data / "test/test_images", json.loads(test_manifest.read_text())
    )
    if args.download_checkpoints:
        run("scripts.download_checkpoints")
    out = (ROOT / args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.allow_existing_output:
        raise ValueError("Choose an empty output directory to avoid stale predictions")
    inference_started = time.perf_counter()
    run(
        "scripts.predict_pipeline",
        "--config",
        ROOT / release["pipeline_config"],
        "--split",
        "test",
        "--output",
        out,
    )
    inference_seconds = time.perf_counter() - inference_started
    csv_path = out / "final/submission.csv"
    run(
        "scripts.audit_submission",
        "--submission",
        csv_path,
        "--test-images",
        data / "test/test_images",
        "--output",
        out / "audit.json",
    )
    actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    result = {
        "submission": str(csv_path),
        "sha256": actual,
        "reference_sha256": release["csv_sha256"],
        "byte_identical": actual == release["csv_sha256"],
        "inference_seconds": inference_seconds,
        "total_seconds_including_preparation_and_download": time.perf_counter()
        - started,
        "note": "CUDA, hardware or library differences can change threshold-edge pixels; the audit still must pass.",
    }
    import torch

    result["runtime"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "deterministic_pipeline": json.loads(
            (ROOT / release["pipeline_config"]).read_text()
        ).get("deterministic", False),
    }
    (out / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
