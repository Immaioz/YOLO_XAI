# EXPLAINABLE AI FOR YOLO OBJECT DETECTORS

This repo contains scripts to get and evaluate explanation from YOLOv8 models using EigenCAM framework.
For this application models trained to detect boats have been used, but the process is scalable and extendable to other case-studies.

## Metrics
Metrics used to evaluate the quality of the explanations and the model's performances are AOPC and HAAS. These metrics evaluate the performances of the model as classifier, but also IoU is tested in with the purpose of evaluating also the localization performances of the model taken into account.

## Process

This script performs 3 kind of perturbation on the images in the [dataset](utils/dataset/subset) folder: mean and random perturbation are performed to compute and evaluate AOPC, whereas to compute HAAS least relevant pixels in the saliency map are removed.

### Perturbation

First the heatmap is obtained using EigenCAM, then the grayscale version is used to substitute the least relevant pixels with the mean of the value of pixels in the image or with a random distribution

![](utils/images/output.png)


![](utils/images/mean1.png)


![](utils/images/random1.png)

This process is repeated for different percentiles of least relevant pixels selected, that goes from 0 to 100%.



# Usage:


```bash
usage: argmark [-h] [-d D] [-t T] [-c C] [-f] [-s]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
|`-D`||`dati/trained_model/`|Relative path to the directory containing .pkl files.|
|`-T`||`0`|Select confidence (0), iou (1) or initial confidence (2).|
|`-C`||`7`|Select the specific class (default All).|
|`-F`|||Print confidence and iou on csv|
|`-S`|||Set confidence and iou or confidence only (Single Class)|

