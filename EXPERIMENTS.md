# Experimental record

The report contains the main scientific argument. This record connects that argument to the saved configurations, histories and decisions. All PQ values are in [0, 1]. Development-fold values are subject to selection on that fold; they are not independent test estimates.

## Progression

1. **Audit before fitting.** Decode masks, retain separate annotators, check native dimensions, group temporally adjacent observations and exact image duplicates, and freeze five folds.
2. **Establish a semantic baseline.** A 1024-pixel ConvNeXt-Tiny U-Net reaches fold-0 checkpoint-selection PQ 0.3829. Foreground, boundary supervision and instance extraction are evaluated under the organizer's per-annotator metric.
3. **Increase useful spatial detail.** Fine-tune at 1536 pixels, average three flip views, and group nearby fragments without dilating the submitted masks. The calibrated coarse fold-0 recipe reaches 0.4318. The checkpoint-selection score uses a different extraction protocol and should not be substituted for this number.
4. **Refine individual candidates.** Train a ResNet18 U-Net on native crops containing the image and proposal mask. Blend local mask evidence with the proposal, reject low-quality candidates, and use a 64-pixel minimum area. The full refined recipe reaches fold-0 PQ 0.4414.
5. **Confirm across held-out groups.** Freeze the post-processing recipe and complete folds 1–4. Refinement improves every fold; pooled five-fold PQ is 0.4427. The paired gain on confirmation folds is +0.0149, with a temporal-group bootstrap interval [0.0118, 0.0182]. This is evidence for the recipe, not for an ensemble evaluated on those same images.
6. **Test alternatives and retain negative results.** Higher resolution, more TTA, morphology, additional proposal scoring, a second backbone, instance detectors, loss weighting and native tile training are investigated. The selected deployment is recorded separately in `artifacts/release.json`.

## Controlled comparisons

| Development comparison | Fold-0 PQ | Interpretation |
|---|---:|---|
| Coarse 1536, three-view TTA | 0.4318 | Coarse reference |
| Coarse native 2048 | 0.4262 | Higher resolution alone does not help |
| Coarse eight-view TTA | 0.4316 | Additional inference cost is not justified |
| Coarse 1536 + 2048 | 0.4311 | No complementary gain |
| Coarse hysteresis grouping | 0.4327 | Small development-only gain |
| Frozen native refinement | 0.4414 | Refined reference |
| Geometry and quality selection | 0.4421 | Gain below 0.001 |
| Small-island cleanup | 0.4416 | Negligible change |
| Closing, radius 6 | 0.4398 | Shape alteration hurts |
| Refiner quality loss weight 0.5 | 0.4408 | Retain weight 0.1 |
| Component-balanced fine-tuning + refinement | 0.4416 | Negligible development gain |
| Annotation-weighted sampling + refinement | 0.4393 | Retain uniform image sampling |
| Native 1024-pixel tile fine-tuning + refinement | 0.4397 | Global context remains valuable |
| Tiny + V2-Base + refinement | 0.4477 | Complementary signal; fold-1 interval crosses zero |
| Calibrated Mask R-CNN | 0.4035 | Below the semantic recipe |
| Mask R-CNN + transferred refiner | 0.4254 | Improves detector masks but remains below reference |
| Add isolated refined detector instances | 0.4431 | Gain interval [-0.0009, 0.0047]; not promoted |

The V2 mixture improves fold 1 from 0.4319 to 0.4360, but its paired gain interval [-0.0032, 0.0104] crosses zero. The initial sparse-loss Mask2Former run collapses; a dense-loss retry reaches checkpoint-selection PQ 0.2756. These results describe the allocated runs and do not establish an intrinsic limitation of the architectures.

`artifacts/experiment_decisions.json` supplies more detailed comparisons and selection decisions. `artifacts/experiments/*/history.jsonl` preserves training and validation events. The registry labels full-data deployment runs separately; they do not have a held-out validation score.

## Deployment comparison

| Submitted candidate | Public PQ | Role |
|---|---:|---|
| Single Tiny fold + native refiner | 0.38 | First submission |
| Five Tiny folds + five native refiners | 0.38 | Confirmed-recipe ensemble |
| Five Tiny + two V2-Base + five refiners | 0.38 | Released mixed architecture pipeline |
| All-data Tiny + all-data native refiner | 0.38 | Fixed 30/24/12-epoch deployment alternative |

The public display does not distinguish these candidates. The mixed pipeline is retained on the basis of the modest two-fold complementary signal, with its uncertainty explicitly reported. No public-score gain is claimed. See the release identity for the exact CSV and Kaggle reference.

## Engineering progression

| Change | Purpose and measured evidence |
|---|---|
| Packed, memory-mapped image/target caches | Remove repeated compressed-file decoding and per-instance morphology from epochs |
| Persistent workers, pinned memory and a CUDA transfer stream | Overlap CPU loading, transfer and GPU compute |
| Channels-last, BF16, fused AdamW and grouped EMA updates | Reduce convolution, optimizer and EMA overhead |
| CPU validation overlap with checkpoint provenance | Keep GPU training productive while preserving the evaluated model state |
| Worker/thread tuning after migration to 16 vCPUs | Avoid CPU oversubscription on the final host |
| uint8 detector targets | Fourfold target storage/transfer reduction relative to float32 |
| Disable dynamic-ROI cuDNN algorithm search | Avoid repeated autotuning stalls; steady 1024 detector epochs take about 26 seconds |
| Decode compact mask ROIs with aligned coordinates | Exact RLE agreement on six image/resolution checks; sampled 1536 CPU decoder time falls from about 0.57–0.58 s to 0.005–0.007 s |

The decoding measurement is a component microbenchmark. It is not an end-to-end speedup claim for the semantic pipeline. Training histories record throughput and peak memory per epoch. Driver faults interrupted an earlier host; retained checkpoints and configuration validation enabled continuation after migration.

## Remaining uncertainty

The public leaderboard remains lower than grouped local validation. The report therefore separates public scores, OOF evidence, development tuning and full-data deployment. Bootstrap intervals are conditional on fitted models, not nested model-selection intervals. Small-instance recall and annotator disagreement remain material limitations. No claim is made about an unavailable private score.
