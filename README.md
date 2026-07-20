# SpillGuard

SpillGuard is a deep-learning research project for detecting marine oil spills in Sentinel-1 Synthetic Aperture Radar (SAR) image patches. It performs binary classification of images as **Oil** or **No-Oil**, with particular emphasis on recall so that potential spills are less likely to be missed.

The project compares four ImageNet-pretrained architectures under a consistent experimental setup:

- ConvNeXt-Tiny
- DenseNet121
- EfficientNet-B0
- Vision Transformer (ViT-B/16)

## Repository contents

| Notebook | Purpose |
| --- | --- |
| `SpillGuard_Preprocessing.ipynb` | Inspects and preprocesses the SAR dataset using Lee filtering, logarithmic transformation, percentile clipping, CLAHE, resizing, and stratified splitting. |
| `SpillGuard_ConvNeXt.ipynb` | Trains and evaluates ConvNeXt-Tiny. |
| `SpillGuard_DenseNet.ipynb` | Trains and evaluates DenseNet121. |
| `SpillGuard_EfficientNet.ipynb` | Trains and evaluates EfficientNet-B0. |
| `SpillGuard_ViT.ipynb` | Trains and evaluates ViT-B/16. |

## Dataset

The notebooks use the CSIRO Sentinel-1 SAR Oil/No-Oil dataset, consisting of 5,630 grayscale image patches (3,725 No-Oil and 1,905 Oil). The dataset is not included in this repository.

Place the original images in the following structure:

```text
data/
├── S1SAR_UnBalanced_400by400_Class_0/
└── S1SAR_UnBalanced_400by400_Class_1/
```

The preprocessing notebook produces this local structure:

```text
processed_dataset/
├── train/
├── val/
└── test/
```

Each split contains `Class_0` and `Class_1` directories. Dataset files, processed images, model weights, and generated results are excluded from Git.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
jupyter notebook
```

Run `SpillGuard_Preprocessing.ipynb` first. Then run any of the four model notebooks. Generated weights, metrics, plots, and prediction files are written to `model_outputs/`.

## Method summary

The preprocessing pipeline applies Lee filtering, logarithmic backscatter transformation, percentile clipping, CLAHE contrast enhancement, resizing to 224 x 224 pixels, and ImageNet normalization. The training workflow uses stratified 70/15/15 splits, data augmentation, weighted sampling, and positive-class-weighted loss to address class imbalance.

The accompanying study reports ConvNeXt-Tiny as the strongest evaluated backbone, achieving 95.62% accuracy, 97.20% Oil-class recall, and 99.42% AUC-ROC on the held-out test set.


