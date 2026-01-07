EEG-MotorImagery-BCI
Project Overview

This repository implements a high-fidelity, classical EEG signal processing and motor imagery classification pipeline, designed for robust, interpretable, and subject-specific brain-computer interface (BCI) applications. The primary objective is to extract discriminative neural features corresponding to imagined left- and right-hand movements from multi-channel EEG recordings and to classify these signals using a classical machine learning approach.

The pipeline serves as a lightweight, fully interpretable baseline for motor imagery decoding, providing a structured preprocessing, feature extraction, and classification workflow that can be easily scaled or extended for multi-subject studies or integrated into hybrid EEG-to-ML/DL frameworks.

Dataset Handling and Preprocessing

EDF Acquisition
Raw EEG data are loaded from standard European Data Format (EDF) files. Each recording contains multiple electrode channels sampled at 160 Hz. Metadata including annotations and segment markers are extracted for precise temporal segmentation.

Channel Alignment
Channels are reordered according to a standardized montage. Missing electrodes in either the recording or the target montage are handled via zero-padding to maintain matrix consistency across trials, ensuring compatibility with spatial filtering operations.

Segment Concatenation
Multi-segment EEG trials are concatenated into a continuous matrix per electrode. This allows for consistent temporal windowing and filtering across all trials, mitigating boundary artifacts during bandpass filtering.

Spatial Coordinate Integration
Electrode positions in 3D space are imported to enable surface Laplacian computation. This spatial derivation enhances local signal-to-noise ratio and attenuates volume conduction effects, emphasizing true cortical activity relevant to motor imagery.

Surface Laplacian Computation
The Laplacian perrinX method is applied to the preprocessed EEG data, producing a surface-referenced signal that reduces common-mode potentials and improves discriminability of motor cortex activations.

Signal Processing and Feature Extraction

Bandpass Filtering
Two primary frequency bands are extracted:

Alpha (8–12 Hz): Captures sensorimotor rhythms associated with idling and motor preparation.

Beta (13–30 Hz): Captures motor execution and sensorimotor desynchronization events.

Fourth-order Butterworth filters are applied bidirectionally using zero-phase forward-backward filtering to prevent phase distortion.

Trial Windowing
Data are segmented into subject-specific time windows corresponding to left- and right-hand motor imagery periods. Padding is applied to minimize edge effects during filtering operations.

Common Spatial Patterns (CSP)
Spatial filters are derived to maximize variance differences between imagined movement classes:

High-variance projections correspond to class-specific neural activations.

Low-variance projections suppress non-discriminative components.

These projections are applied to filtered alpha and beta signals, yielding class-specific feature vectors.

Feature Engineering
Log-variance of CSP-projected signals is computed for each trial, forming a low-dimensional feature representation. This captures the relative amplitude modulations induced by motor imagery while preserving interpretability.

Classification

Label Encoding
Binary labels are assigned to trials:

Left-hand imagery: +1

Right-hand imagery: -1

This encoding aligns with standard SVM formulations for linear and non-linear classification.

Support Vector Machine (SVM)
Features are input to a classical SVM classifier:

Linear kernel: Captures linearly separable components of the motor imagery variance features.

Radial basis function (RBF) kernel: Captures non-linear interactions between features if present.

Data are standardized to zero-mean and unit variance to ensure numerical stability.

Cross-Validation
K-fold cross-validation is performed to assess generalization performance and mitigate overfitting. Classification loss metrics are reported for quantitative evaluation.

Technical Significance

This pipeline demonstrates several key neuroengineering and signal processing principles:

Interpretable Feature Extraction: CSP and log-variance features provide direct insight into cortical activation patterns during motor imagery.

Spatial Filtering: Surface Laplacian derivation reduces volume conduction artifacts, improving spatial resolution without requiring high-density EEG.

Robust Preprocessing: Segmentation, zero-padding, and careful channel alignment ensure reproducibility across datasets with variable electrode configurations.

Classical Baseline for BCI: The pipeline functions as a benchmark for comparing more complex machine learning or deep learning models, providing a lightweight and explainable starting point for motor imagery decoding.

Applications

Non-invasive BCI control for assistive devices (e.g., prosthetics, exoskeletons)

Motor imagery research for neurorehabilitation

Benchmarking EEG-based classification frameworks

Real-time motor cortex activation monitoring

Dependencies

MATLAB signal processing toolbox (for filtering, CSP, and matrix operations)

EDF reading and metadata parsing functionality

CSV import utilities for electrode coordinates

Future Extensions

Integration with multi-subject datasets for transfer learning

Incorporation of adaptive CSP and online SVM updating for real-time BCI

Extension to multi-class motor imagery paradigms (e.g., left hand, right hand, feet)

Replacement of SVM with deep neural networks while maintaining CSP-derived feature interpretability

This README positions the project as a rigorously engineered, research-ready pipeline for EEG motor imagery decoding, emphasizing reproducibility, technical depth, and interpretability in classical BCI design.
