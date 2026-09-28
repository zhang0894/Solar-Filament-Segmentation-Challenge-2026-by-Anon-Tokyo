"""Assemble an explicit, credential-free Anon Tokyo delivery allowlist."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile

from filament.data import ROOT, read_manifest


def release_files(repo):
    """Enumerate publication inputs, excluding local data and execution caches."""
    for name in [
        "LICENSE",
        ".gitignore",
        "README.md",
        "NOTICE.md",
        "REPRODUCIBILITY.md",
        "MODEL_CARD.md",
        "EXPERIMENTS.md",
        "ORGANIZER_CHECKLIST.md",
        "CITATION.cff",
        "PUBLISHING.md",
        "requirements.txt",
        "pytest.ini",
        "pipeline.ipynb",
    ]:
        file = repo / name
        if file.is_file():
            yield file
    for folder in ["filament", "scripts", "tests", "configs", "artifacts", "report"]:
        for file in sorted((repo / folder).rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                yield file


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--public-score", type=float)
    parser.add_argument("--submission-ref", required=True)
    parser.add_argument("--kaggle-owner", default="zpf999")
    parser.add_argument("--checkpoint-version", type=int, default=1)
    parser.add_argument("--extra-pipeline", action="append", default=[])
    args = parser.parse_args()
    if args.checkpoint_version < 1:
        parser.error("checkpoint-version must be positive")
    delivery = ROOT / "deliverables/AnonTokyo"
    repo = delivery / "repository"
    artifacts = repo / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle = delivery / "checkpoint_bundle"
    bundle.mkdir(exist_ok=True)
    paths = set()
    for filename in [args.pipeline, *args.extra_pipeline]:
        config = json.loads((ROOT / filename).read_text())
        for item in config["models"] + config.get("refiners", []):
            if "directory" in item:
                raise ValueError(
                    "A release pipeline must use checkpoints, not local cached predictions"
                )
            paths.add(item["checkpoint"])
    checkpoints = []
    for relative in sorted(paths):
        source = ROOT / relative
        if not source.is_file():
            raise ValueError(f"Missing release checkpoint: {relative}")
        filename = source.parent.name + "__" + source.name
        target = bundle / filename
        sha = digest(source)
        if target.exists() and digest(target) != sha:
            target.unlink()
        if not target.exists():
            os.link(source, target)
        checkpoints.append(
            {
                "path": relative,
                "bundle_filename": filename,
                "sha256": sha,
                "bytes": source.stat().st_size,
            }
        )
    allowed_bundle = {c["bundle_filename"] for c in checkpoints} | {
        "dataset-metadata.json",
        "CHECKPOINTS.json",
        "README.md",
        "NOTICE.md",
    }
    for file in bundle.iterdir():
        if file.is_file() and file.name not in allowed_bundle:
            file.unlink()  # Remove only obsolete staged hardlinks, never original runs.
    slug = "anon-tokyo-solar-filament-2026-checkpoints"
    dataset_ref = args.kaggle_owner + "/" + slug
    manifest = {
        "team": "Anon Tokyo",
        "dataset_ref": dataset_ref,
        "dataset_version": args.checkpoint_version,
        "download_url": "https://api.kaggle.com/v1/datasets/download/"
        + dataset_ref
        + f"?datasetVersionNumber={args.checkpoint_version}",
        "files": checkpoints,
    }
    (artifacts / "checkpoints.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (bundle / "CHECKPOINTS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (bundle / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Anon Tokyo Solar Filament 2026 Checkpoints",
                "id": dataset_ref,
                "licenses": [{"name": "CC-BY-NC-4.0"}],
                "subtitle": "Fine-tuned inference checkpoints with SHA256 provenance",
                "description": "Fine-tuned inference checkpoints for Anon Tokyo's Solar Filament Segmentation Challenge 2026 solution. Training uses the official competition training masks. Source, configurations, report and reproduction notebook: https://github.com/zhang0894/Solar-Filament-Segmentation-Challenge-2026-by-Anon-Tokyo . The bundle contains no competition images or optimizer states. Upstream pretrained model terms remain applicable; see NOTICE.md.",
            },
            indent=2,
        )
        + "\n"
    )
    (bundle / "README.md").write_text(
        "# Anon Tokyo checkpoints\n\nUse the public repository's `scripts.download_checkpoints` to download and verify this bundle. `CHECKPOINTS.json` maps each file to its expected repository path and SHA256. These are fine-tuned inference states; consult the repository for data acquisition, configuration, validation and attribution.\n"
    )
    copy(repo / "NOTICE.md", bundle / "NOTICE.md")

    for folder in ["filament", "scripts", "tests"]:
        for file in (ROOT / folder).glob("*.py"):
            copy(file, repo / folder / file.name)
    for file in (ROOT / "configs").glob("*.json"):
        copy(file, repo / "configs" / file.name)
    copy(ROOT / "requirements.txt", repo / "requirements.txt")
    copy(ROOT / "pytest.ini", repo / "pytest.ini")
    test_dir = ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    test_manifest = {
        "files": [
            {
                "filename": file.name,
                "bytes": file.stat().st_size,
                "sha256": digest(file),
            }
            for file in sorted(test_dir.glob("*.jpeg"))
        ]
    }
    (artifacts / "test_images.json").write_text(
        json.dumps(test_manifest, indent=2) + "\n"
    )
    for file in (delivery / "report").rglob("*"):
        if not file.is_file():
            continue
        relative = file.relative_to(delivery / "report")
        if file.name.startswith("preview-") or file.name == "main_text.txt":
            continue
        if file.suffix not in {".tex", ".bib", ".pdf", ".png", ".json", ".npz"}:
            continue
        copy(
            file,
            repo
            / "report"
            / ("AnonTokyo_Report.pdf" if file.name == "main.pdf" else relative),
        )
    oof = ROOT / "runs/oof_fixed_recipe_v1"
    copy(oof / "metrics.json", artifacts / "oof_metrics.json")
    copy(
        ROOT / "runs/experiment_decisions.json", artifacts / "experiment_decisions.json"
    )
    copy(
        ROOT / "runs/rcnn_roi_decode_equivalence.json",
        artifacts / "rcnn_roi_decode_equivalence.json",
    )
    rows = read_manifest()["images"]
    with (artifacts / "folds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "fold", "group", "annotator_entries"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "image_id": r["image_id"],
                "fold": r["fold"],
                "group": r["group"],
                "annotator_entries": len(r["annotators"]),
            }
            for r in rows
        )
    runs = []
    for config_path in sorted((ROOT / "runs").glob("*/config.json")):
        cfg = json.loads(config_path.read_text())
        if cfg.get("kind") not in {"semantic", "instance", "rcnn", "refinement"}:
            continue
        directory = config_path.parent
        history = []
        if (directory / "history.jsonl").exists():
            history = [
                json.loads(line)
                for line in (directory / "history.jsonl").read_text().splitlines()
                if line.strip()
            ]
        epochs = [x.get("epoch", 0) for x in history]
        validations = [x.get("best_pq") for x in history if x.get("best_pq", -1) >= 0]
        timed_epochs = {x["epoch"]: x for x in history if "train_seconds" in x}
        record = {
            "name": directory.name,
            "kind": cfg.get("kind"),
            "fold": cfg.get("fold"),
            "size": cfg.get("size"),
            "planned_epochs": cfg.get("epochs"),
            "completed_epoch": max(epochs, default=0),
            "best_validation_pq": max(validations) if validations else None,
            "recorded_training_seconds": (
                sum(x["train_seconds"] for x in timed_epochs.values())
                if timed_epochs
                else None
            ),
            "config": cfg,
        }
        runs.append(record)
        copy(config_path, artifacts / "experiments" / directory.name / "config.json")
        if history:
            copy(
                directory / "history.jsonl",
                artifacts / "experiments" / directory.name / "history.jsonl",
            )
    (artifacts / "experiment_registry.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "note": "Raw checkpoint selection scores and complete epoch records; calibration/ensemble comparisons are reported separately. All-data runs have no validation score.",
            },
            indent=2,
        )
        + "\n"
    )
    (artifacts / "compute_summary.json").write_text(
        json.dumps(
            {
                "note": "Saved epoch training timers only; excludes validation, inference, preprocessing and unrecorded failed work. Hardware changed during the project. Null denotes no separate training timer.",
                "runs": [
                    {
                        k: run[k]
                        for k in [
                            "name",
                            "kind",
                            "size",
                            "completed_epoch",
                            "recorded_training_seconds",
                        ]
                    }
                    for run in runs
                ],
            },
            indent=2,
        )
        + "\n"
    )
    submission = ROOT / args.submission
    copy(submission / "submission.csv", artifacts / "submission.csv")
    copy(submission / "submission.csv", delivery / "submission.csv")
    copy(delivery / "report/main.pdf", delivery / "AnonTokyo_Report.pdf")
    for name in ["metadata.json", "validation.json", "submission_receipt.json"]:
        target = artifacts / ("submission_" + name if name == "metadata.json" else name)
        if (submission / name).exists():
            value = json.loads((submission / name).read_text())
            # Normalize local workspace prefixes in public audit metadata.
            clean = json.dumps(value, indent=2).replace(str(ROOT) + "/", "")
            target.write_text(clean + "\n")
        elif target.exists():
            target.unlink()
    release = {
        "team": "Anon Tokyo",
        "contact": "pufanzhang1@gmail.com",
        "repository": "https://github.com/zhang0894/Solar-Filament-Segmentation-Challenge-2026-by-Anon-Tokyo",
        "pipeline_config": args.pipeline,
        "manifest_sha256": digest(ROOT / "data/manifest.json"),
        "test_manifest_sha256": digest(artifacts / "test_images.json"),
        "csv_sha256": digest(submission / "submission.csv"),
        "submission_ref": args.submission_ref,
        "public_score": args.public_score,
        "private_score": None,
        "oof_pq": 0.44266329482574596,
        "oof_scope": "Frozen Tiny/refiner single-fold recipe, pooled over five held-out folds; not an OOF estimate for an all-data or test ensemble.",
        "checkpoint_dataset": "https://www.kaggle.com/datasets/" + dataset_ref,
        "checkpoint_dataset_version": args.checkpoint_version,
    }
    (artifacts / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    # Scan actual Kaggle credential bytes without displaying them; passwords are
    # never read, stored, or copied by this packaging utility.
    secrets = [
        file.read_bytes().strip()
        for name in ["kaggleAPI.txt", "github_token.txt"]
        if (file := ROOT / name).is_file()
    ]
    audit = {}
    for file in sorted(release_files(repo)):
        data = file.read_bytes()
        if any(secret and secret in data for secret in secrets):
            raise ValueError("Credential detected in release allowlist")
        audit[file.relative_to(repo).as_posix()] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    (delivery / "file_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    with zipfile.ZipFile(
        delivery / "AnonTokyo_Delivery.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in audit:
            archive.write(repo / relative, "AnonTokyo/" + relative)
        archive.write(delivery / "file_manifest.json", "AnonTokyo/file_manifest.json")
    print(
        json.dumps(
            {
                "files": len(audit),
                "checkpoints": len(checkpoints),
                "checkpoint_bytes": sum(x["bytes"] for x in checkpoints),
                "repo": str(repo),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
