# Model card

**Team:** Anon Tokyo · **Contact:** pufanzhang1@gmail.com

**Task:** instance segmentation of solar filaments in native 2048×2048 GONG H-alpha observations for the Solar Filament Segmentation Challenge 2026.

**Architecture:** a ConvNeXt U-Net produces global foreground and boundary logits. Foreground fragments are grouped without thickening their masks. A four-channel ResNet18 U-Net processes image-plus-instance-mask crops at native resolution, predicts a refined mask and a rejection score, and blends local evidence with a smoothed proposal. Confidence-ordered allocation guarantees disjoint output masks. The exact deployed checkpoints and ensemble weights are specified in `artifacts/release.json` and its referenced pipeline configuration.

**Released deployment:** five ConvNeXt-Tiny fold models receive 0.1 probability weight each; two ConvNeXt-V2-Base models (held-out folds 0 and 1) receive 0.25 each. Five native refiners receive equal weight and process identical proposals. The architecture mixture has modest two-fold local evidence; the confirmation-fold gain interval crosses zero.

**Training data:** 707 official training images, 1,154 annotator-image entries and 8,199 annotated instances. Official masks supply task supervision; generic ImageNet encoders supply initialization. Auxiliary supplied classes, spines and boxes are not training targets. All annotations of a given observation are kept in one validation fold.

**Validation:** five temporal/duplicate-grouped folds, 331 groups. Development is performed on fold 0 and the frozen refinement recipe is checked on folds 1–4. Pooled refined OOF PQ is 0.4427; coarse PQ is 0.4287. Paired temporal-group bootstrap confidence intervals are conditional on the fitted models and do not account for all model-selection uncertainty. Public leaderboard scores and the final submission identifier are recorded separately in the release identity. The private score is not known.

**Principal limitations:** small and faint filament recall; inconsistent annotator boundaries or instance grouping; imperfect physical independence of temporal groups; in-sample training proposals for the refiner; domain shift between local validation and the test distribution. A quality-head score is a learned rejection signal, not a calibrated probability. The model was developed for this competition, and its suitability for other instruments or scientific catalog generation has not been established.

**Compute:** one RTX 5090 with 32 GB VRAM, PyTorch 2.8.0, CUDA 12.8 and Python 3.12. CPU capacity changed during experimentation; final runs use 16 vCPUs. Mixed precision, memory-mapped targets, asynchronous transfers and bounded CPU thread pools reduce training stalls.

**Provenance:** checkpoint SHA256 hashes, run configurations, epoch histories, released CSV hash and the independent RLE audit are supplied. Upstream attribution is in `NOTICE.md`. The public weights contain fine-tuned inference states, without optimizer states or source images.
