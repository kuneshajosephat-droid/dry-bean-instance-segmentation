# Instance Segmentation of Intact and Broken Dry Beans

## Author

J Kunesha

## Task

Identify each dry bean or broken fragment in a photograph, classify it as intact or broken, and predict a separate pixel mask outlining each object.

## Dataset

The original collection contained 200 personally collected JPEG photographs:

| Image group | Collected photographs |
| --- | ---: |
| Intact only | 40 |
| Broken only | 40 |
| Mixed, separated | 80 |
| Mixed, touching | 40 |
| Total | 200 |

Four duplicate copies were removed, leaving **196 unique photographs**.

Objects were manually outlined with polygons in CVAT and exported in COCO JSON format. The corrected dataset contains **725 object instances: 440 intact and 285 broken**.

Classes:

- **intact:** whole beans, including beans with natural seed scars.
- **broken:** visibly broken beans and detached fragments.

Each object has a separate annotation. Masks follow the visible object boundary and exclude background and shadows.

### Dataset partition

The saved partition follows an approximately 80% training, 10% validation, and 10% test split.

| Split | Photographs | Intact instances | Broken instances |
| --- | ---: | ---: | ---: |
| Training | 156 | 351 | 230 |
| Validation | 20 | 44 | 28 |
| Test | 20 | 45 | 27 |
| Total | 196 | 440 | 285 |

The splits do not overlap. The test set is reserved for final evaluation.

## Model

The model is Mask R-CNN with a ResNet-50-FPN backbone and COCO-pretrained weights.

The unchanged pretrained model was first evaluated on the training photographs. Its classification, box regression, and mask output layers were then replaced for two bean classes and background.

The adapted model was evaluated on the training and validation sets **without training**, establishing a baseline before fine-tuning.

Original implementation: https://github.com/pytorch/vision

Mask R-CNN source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py

## Preprocessing

Images are converted to RGB tensors with values between 0 and 1. Mask R-CNN internally applies RGB normalization using mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

The model resizes images toward an 800-pixel shorter side while limiting the longer side to 1333 pixels.

Horizontal flipping with probability 0.5 is defined for training and applied jointly to images, masks, and bounding boxes. Baseline evaluation uses images without random augmentation.

## Baseline Evaluation

Images are loaded in groups of eight and processed individually to limit CPU memory use. The first image from each group is visualized with its ground truth and predictions.

Evaluation includes:

- COCO mask mAP over IoU thresholds from 0.50 to 0.95.
- Mask AP50, AP75, and AP for each class.
- Bounding-box mAP.
- Average total loss and individual loss components.

Loss components include classification, box regression, mask prediction, RPN objectness, and RPN box regression. Losses are calculated without updating model parameters.

## Code

`01_dataset_and_untrained_baseline.py` includes:

- Dataset and saved-split validation.
- A custom PyTorch dataset class.
- Annotation and augmentation visualizations.
- Inference using the unchanged pretrained model.
- Output-layer replacement for the bean classes.
- Training and validation baseline evaluation without training.
- Export of predictions, figures, metrics, and losses.

The script expects the corrected dataset folder to contain `images/` and `annotations/`, including `instances_all.json`, `instances_train.json`, `instances_val.json`, and `instances_test.json`.

Run: ```bash
python 01_dataset_and_untrained_baseline.py --dataset "path/to/dry_bean_dataset_corrected_196"




## Method Paper
He, K., Gkioxari, G., Dollár, P., and Girshick, R. (2017).
Mask R-CNN. Proceedings of ICCV, 2961–2969.

https://doi.org/10.1109/ICCV.2017.322

## Related Paper
Toda, Y., Okura, F., Ito, J., Okada, S., Kinoshita, T.,
Tsuji, H., and Saisho, D. (2020).
Training instance segmentation neural network with synthetic
datasets for crop seed phenotyping.
Communications Biology, 3, 173.

https://doi.org/10.1038/s42003-020-0905-5

Planned Experiments
Part 1: Transfer Learning
1. Fine-tune the COCO-pretrained Mask R-CNN for intact and broken beans.
2. Evaluate a research-based technique to improve the fine-tuned baseline.
3. Deploy the model with a web interface for uploading photographs and viewing predictions.

Part 2: Training from Scratch
1. Use an AI agent, such as ChatGPT, to help design and implement a small instance segmentation network.
2. Train the network without pretrained weights.
3. Compare it with the fine-tuned Mask R-CNN on the same held-out test set using mask average precision, parameter count, and inference time.

Current Status
Annotation corrections, duplicate removal, dataset validation, and partitioning are complete. Pretrained inference and adapted baseline evaluation without training are complete. Fine-tuning, training from scratch, and deployment remain planned.
