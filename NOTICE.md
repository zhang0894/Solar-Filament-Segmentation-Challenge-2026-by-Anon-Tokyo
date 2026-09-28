# Attribution and artifact scope

The MIT license covers Anon Tokyo's original source code. Third-party libraries, pretrained weights, competition data and the official report template retain their own terms.

- ConvNeXt and ConvNeXt V2: Meta / original authors; implementations through timm and segmentation-models-pytorch. See https://github.com/facebookresearch/ConvNeXt and https://github.com/facebookresearch/ConvNeXt-V2.
- U-Net and ResNet architectures: cited in the technical report. Implementations through segmentation-models-pytorch and torchvision.
- Model initialization: `timm/convnext_tiny.fb_in22k_ft_in1k`, optional `timm/convnextv2_base.fcmae_ft_in22k_in1k`, and ImageNet ResNet18. Checkpoint identifiers and task fine-tuning configurations accompany each released weight.
- Diagnostic Mask R-CNN and Mask2Former experiments use public COCO initialization; these are separate from the selected semantic/refinement pipeline unless explicitly listed in the release manifest.
- The evaluator follows the organizers' public Self_Evaluation_Notebook, with an additional strict one-to-one diagnostic: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook.
- Report formatting follows the official competition Overleaf template: https://www.overleaf.com/read/vjztpvrnbrdh. Required introduction and acknowledgment text is retained. ACM-style formatting does not imply ACM publication or endorsement.
- GONG/MAGFiLO imagery shown in report figures is attributed in the report's mandatory acknowledgment. The repository does not redistribute the full competition dataset.

The checkpoint bundle is distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). In particular, the upstream [ConvNeXt-V2-Base model card](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k) specifies CC-BY-NC-4.0; the fine-tuned V2 checkpoints retain those noncommercial and attribution conditions. The MIT source-code license does not grant a separate license to those weights.

The checkpoint bundle contains the team's fine-tuned inference states, not optimizer states, API credentials or competition images. Source-code licensing does not replace the competition data agreement or upstream model terms.
