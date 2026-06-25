# Unsupervised Learning in Computer Vision — Road Crack Detection

**Course:** DSA4050 — Unsupervised Learning in Computer Vision

## Overview

This project explores unsupervised representation learning methods for distinguishing cracked from uncracked road surfaces, with no ground-truth labels used during training. It covers five core tasks — preprocessing, clustering, dimensionality reduction, contrastive learning, and comparative evaluation — plus a bonus GAN-based feature learning task.

**Central question:** Which unsupervised representation learning method most effectively captures the visual distinction between cracked and uncracked road surfaces without label supervision?

## Dataset

- `road_cracks`: 125 images of cracked road surfaces only
- `train`: 1,389 images of cracked and uncracked surfaces (162 cracked / 1,227 uncracked)
- Images preprocessed to grayscale, resized to 128×128, normalised to [0, 1]
- Augmentations (flips, rotation, crop, colour jitter, Gaussian blur) used only for SimCLR positive-pair generation

## Methods

1. **Clustering (K-Means / DBSCAN)** on HOG (1,764-dim) and pretrained ResNet-18 (512-dim) features
2. **Dimensionality reduction** (PCA, t-SNE) for visualising the CNN feature space
3. **Contrastive learning (SimCLR)** — ResNet-18 encoder + MLP projection head, trained with NT-Xent loss for 20 epochs
4. **Comparative evaluation** of clustering quality (silhouette score) and downstream classification (logistic regression linear probe)
5. **Bonus: DCGAN** trained from scratch for 30 epochs, using discriminator features as an unsupervised representation

## Key Results

| Method | Features | Silhouette Score | Classifier Accuracy |
|---|---|---|---|
| K-Means k=2 | HOG | 0.043 | — |
| K-Means k=2 | Pretrained CNN | 0.449 | 1.000 |
| K-Means k=2 | SimCLR | 0.123 | 0.996 |
| K-Means k=2 | GAN Discriminator | 0.481 | 0.996 |

- **HOG features are insufficient** for this task — near-zero silhouette scores indicate no meaningful cluster structure.
- **Deep features (pretrained, contrastive, or adversarial) all dramatically outperform HOG**, with classification accuracy above 98.9% across the board.
- **DBSCAN failed** under all tested configurations, consistent with the continuous, unimodal nature of road surface feature distributions.
- The **GAN discriminator achieved the best clustering** (silhouette 0.481), edging out the ImageNet-pretrained CNN (0.449), while **SimCLR and the GAN tied on classification accuracy** (99.64%).
- The pretrained CNN reached perfect classification but depends on large-scale ImageNet supervision, limiting its applicability where no such pretrained model exists.

## Conclusions

Unsupervised representation learning can achieve near-perfect categorisation of road crack images without label supervision. Feature quality (deep vs. shallow) matters far more than the specific unsupervised objective used. The main limitation across all methods is dataset size and class imbalance (88% uncracked vs. 12% cracked); future work should test whether the GAN's clustering advantage over SimCLR holds at larger scale.

## Repository Structure

```
.
├── data/                  # road_cracks/ and train/ image collections
├── notebooks/             # task-by-task analysis notebooks
├── src/                   # preprocessing, clustering, SimCLR, DCGAN code
├── figures/                # generated plots (silhouette scores, t-SNE, loss curves)
└── README.md
```

## Requirements

- Python 3.x
- PyTorch, torchvision
- scikit-learn
- scikit-image (HOG)
- matplotlib
