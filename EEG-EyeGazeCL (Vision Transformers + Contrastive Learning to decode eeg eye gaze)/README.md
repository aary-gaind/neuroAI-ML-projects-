# EEG-GazeCL
Contrastive Spatial Decoding of Eye Gaze from EEG
 
## Overview
EEG-GazeCL is a deep learning framework for decoding spatial gaze targets from EEG signals.  
The model learns spatially consistent latent representations using a Vision-Transformer-style EEG encoder combined with distance-aware contrastive learning.

Evaluated on the **EEGEyeNet** dataset, EEG-GazeCL achieves improved spatial accuracy, reduced catastrophic errors, and smoother gaze predictions compared to standard classification baselines.

This repository contains the **original architecture and original methodology**.

---

## Key Ideas
- EEG gaze decoding is inherently spatial, not purely categorical
- Standard cross-entropy penalizes all errors equally
- Spatial contrastive learning differentiates nearby vs. far errors
- Nearby gaze targets → nearby embeddings
- Far gaze targets → well-separated embeddings
- Spatial regularization is applied early in training and annealed over time

---

## Model Overview

**Input**
- EEG trials of shape `(128 channels × 500 time samples)`

**Backbone**
- EEG Vision Transformer (EEGViT)
  - Temporal convolution → time patches
  - Spatial convolution → channel grouping
  - Transformer encoder with CLS token

**Outputs**
- Discrete gaze class (dot target)
- Latent embedding used for spatial regularization

---

## Spatial Contrastive Learning

A distance-aware spatial contrastive loss is applied:
||z_i - z_j|| ∝ ||g_i - g_j||

Where:
- `z_i` is the EEG embedding
- `g_i` is the 2D gaze position

**Design choices**
- Contrastive loss applied only during early epochs (annealing)
- Gaussian weighting based on gaze distance
- Joint optimization with standard cross-entropy loss

---

## Evaluation Metrics

In addition to classification accuracy, spatial performance is evaluated using:

- Median pixel error
- Mean pixel error
- P(exact): exact dot prediction
- P(within 1 step): nearest-neighbor prediction
- P(within 2 steps): within two spatial steps

These metrics provide a faithful assessment of spatial gaze decoding quality.

---

## Dataset

**EEGEyeNet – Position Task (Dot Targets)**

Each subject file contains:
- `EEG`: `(N, T, C)` EEG trials
- `labels`: `(N, 3)` → `[trial_id, x, y]`

Subjects are stored as separate `.npz` files and merged during training.

---

## Results (Single Subject)

Performance after spatial contrastive learning with annealing:

- Exact accuracy: ~45%  
  (Fine-grained spatial decoding at this resolution is rare in EEG literature)
- Within 1 step: ~54%
- Within 2 steps: ~71%
- Median pixel error: ~55 px (state-of-the-art)

Errors are predominantly local, with rare catastrophic failures.
