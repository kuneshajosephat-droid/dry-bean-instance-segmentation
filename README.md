# dry-bean-instance-segmentation
Instance segmentation of intact and broken dry beans using Mask R-CNN in PyTorch
# Instance Segmentation of Intact and Broken Dry Beans

## Author : J Kunesha

## Task
Identify each dry bean or broken fragment in a photograph,
assign it an intact or broken label, and predict a separate
pixel mask outlining each object.

## Dataset
The dataset contains 200 personally collected photographs:

| Image group | Photographs |
| --- | ---: |
| Intact only | 40 |
| Broken only | 40 |
| Mixed, separated | 80 |
| Mixed, touching | 40 |
| Total | 200 |

Input images are JPEG photographs. Objects were manually
outlined using polygons in CVAT and exported in COCO JSON
format. The current annotations contain 737 object instances.

Classes:
- intact: whole beans, including beans with natural seed scars.
- broken: visibly broken beans and detached fragments.

Each physical object has its own annotation. Masks follow
the visible object boundary and should exclude background
and shadows. Annotation quality review is ongoing.

## Model
The planned baseline is Mask R-CNN with a ResNet-50-FPN
backbone, pretrained on COCO and fine-tuned for the two
bean classes.

Original implementation:
https://github.com/pytorch/vision

Mask R-CNN source:
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py

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

## Project experiments 

### Part 1: Pretrained Model
1. Build the manually annotated dry bean dataset and run
   the selected implementation on it.
2. Fine-tune COCO-pretrained Mask R-CNN with ResNet-50-FPN
   to segment intact beans and broken pieces.
3. Implement and evaluate a technique from recent research
   to improve performance over the fine-tuned baseline.
4. Deploy the model on a server with a web interface for
   uploading photographs and viewing predictions.

### Part 2: Network Trained from Scratch
1. Use an AI agent, such as ChatGPT, to help design and
   implement a small instance segmentation network.
2. Train this network from scratch without pretrained weights.
3. Compare it with the fine-tuned Mask R-CNN using the same
   held-out test set, examining mask average precision,
   parameter count, and inference time.

## Current Status
Data collection and initial manual annotation are complete.
Boundary refinement and dataset preparation are in progress.
Training and evaluation results will be added after experiments.
