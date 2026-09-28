# Reproduction guide

## Frozen release

`artifacts/release.json` is the authoritative mapping between the final submission, pipeline configuration, public score and file hashes. `artifacts/checkpoints.json` lists every inference checkpoint. The public checkpoint dataset contains the trained weights; the repository contains the source, notebook and configurations. Original competition data must be downloaded under the competition terms.

The final inference path does not initialize encoders from remote pretrained weights. It builds the architecture and restores the supplied fine-tuned state. Training from scratch additionally downloads the corresponding upstream ImageNet initialization.

## Installation and data

Use Python 3.12 and install `requirements.txt`. CUDA 12.8 PyTorch wheels are selected by its extra package index. The tested GPU is RTX 5090 with 32 GB VRAM. The final host has 16 vCPUs and approximately 90 GB usable RAM. CPU-only report exploration is possible, but full-resolution inference and training were tested on CUDA.

Download the official archive from the Kaggle competition. Alternatively, export your own `KAGGLE_API_TOKEN` and run `python -m scripts.download_data` after accepting the rules. Never commit credentials. The extracted directory must have:

```text
MAGFiLO_1.0_Kaggle_2026/
  train/train_images/
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/test_images/
```

Run the reproduction command in the README. It creates a symlink under `data/` when `--data-dir` is supplied. Preparation constructs the canonical 1024-pixel cache and frozen folds. A manifest checksum mismatch is an error, not a reason to silently regenerate new folds.

Inference uses original test images. The 1024 cache is retained for data auditing and training; larger training caches are optional. Allow several GB beyond the official dataset for weights and prediction caches. The download uses four validated byte ranges when the server supports them, then verifies every checkpoint SHA256. `python -m scripts.download_checkpoints --workers 1` provides a sequential alternative. The download temporarily needs both the compressed weight archive and extracted weights. Full experiment replay consumes considerably more storage than inference; check free space before creating 1536/2048 caches or optimizer checkpoints.

## Training and validation

The notebook includes runnable training commands with new run names. Its default training example fits fold 0; set `TRAIN_FOLDS=list(range(5))` and `TRAIN_V2=True` to fit all twelve deployed models and generate `configs/pipeline_retrained.json`. Existing experiment configurations and histories are preserved under `artifacts/experiments/`; these describe the actual executed settings, including fine-tuning stages and selected epochs. The training seed is 20260928. All annotations of an image share a fold. Temporal groups and exact duplicate hashes remain together, as checked by the test suite.

For semantic training, prepare the canonical manifest once with `scripts.prepare_data --size 1024`, then build resolution-specific packed caches with `scripts.build_fast_cache --size SIZE --pack-targets`. See the notebook for commands. `build_fast_cache` preserves the canonical manifest while creating the larger training caches.

Refiners train from proposals generated only for the outer training fold. Their proposals are in-sample within that training fold; inner-OOF proposal generation was not performed. This limitation is discussed in the report. The fixed twelve-epoch refiner state is named `epoch_12.pt` for fold models.

Use `scripts.predict_pipeline --split valid` with a corresponding held-out configuration. The pipeline rejects incompatible fold/provenance combinations. Per-annotator counts are pooled under the official evaluator. The five-fold OOF statistics in the report do not evaluate a model trained on all images.

## Output verification

`scripts.audit_submission` decodes every RLE, checks the 2048×2048 shape, unique row IDs, expected image names, nonempty instance masks and pairwise non-overlap. Images with no predicted instances are explicitly counted in the audit; no artificial empty-mask row is added.

`scripts.reproduce` saves an independent audit and a byte-level CSV comparison. The released mixed pipeline enables deterministic CUDA algorithms and disables cuDNN autotuning in inference. Historical validation retains its original inference settings. Exact CSV equality is the strongest replay check on the original runtime. Different GPU kernels or package versions can perturb threshold-edge pixels, so record both the checksum and environment when comparing a port.

## Report and notebook

The report uses the organizer's unmodified `preamble.tex`. Build it in `report/` with an installation containing `acmart` and its fonts:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

This creates `main.pdf`; the released copy is `AnonTokyo_Report.pdf`. The figures and numerical arrays are included. Regenerating the statistical figures from raw OOF predictions requires first rerunning validation for all five folds using the notebook's `RUN_VALIDATION` path, then:

```bash
python -m scripts.build_report_assets --oof validation_reproduction/oof --output validation_reproduction/report
```

Viewing the delivered evidence does not require this GPU replay.

To execute the inexpensive notebook cells:

```bash
python -m ipykernel install --user --name anon-tokyo --display-name 'Anon Tokyo (Python 3.12)'
python -m jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb --ExecutePreprocessor.kernel_name=anon-tokyo
```

Set the training or inference opt-in flag only when ready for GPU work. Do not execute every historical queue plan as a reproduction shortcut: those plans include deliberately rejected experiments and refer to the original work session.
