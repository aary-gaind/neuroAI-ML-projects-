EEG Functional Connectivity Analysis for Cognitive State Detection

Objective:
This pipeline implements a high-resolution analytical framework for quantifying cognitive states—specifically, focus, work engagement, and distraction—via EEG-derived functional connectivity. It leverages precise spatiotemporal signal processing to extract network-level interactions between the Default Mode Network (DMN) and Central Executive Network (CEN).

Overview

Cognitive states are reflected in dynamic coordination patterns across large-scale cortical networks. Traditional EEG metrics (power spectra, event-related potentials) provide limited insight into network-level communication. This project transcends conventional approaches by:

Capturing multi-scale EEG dynamics across 64 channels with fine-grained temporal segmentation.

Aligning empirical electrode layouts with standardized montages, ensuring cross-subject and cross-study comparability.

Enhancing spatial specificity through Perrin Laplacian surface filtering, effectively attenuating volume conduction while highlighting local cortical sources.

Targeted frequency decomposition, isolating alpha (8–12 Hz) and beta (13–30 Hz) bands, critical for attention modulation, working memory, and task engagement.

Computing robust functional connectivity metrics, specifically Spearman rank correlations across temporally segmented windows, allowing statistically reliable estimates of network coordination.

Methodological Depth
1. Multi-Segment Temporal Integration

EEG is inherently non-stationary. The pipeline concatenates multiple short segments to reconstruct continuous signals while preserving temporal dynamics. This approach maximizes statistical power for downstream connectivity analysis and ensures that transient cognitive events are captured within network-level metrics.

2. Channel Alignment and Standardization

Electrode montages often differ between acquisition systems. Misaligned or missing channels introduce significant confounds in connectivity analyses. By systematically mapping empirical electrodes to a reference montage and replacing absent channels with null representations, this framework preserves network topology and enables direct cross-condition comparisons.

3. Spatial Filtering via Laplacian Transformation

Surface Laplacian (Perrin method) reduces volume conduction artifacts, isolating local cortical activity while maintaining global network interactions. This critical step ensures that functional connectivity estimates reflect genuine inter-regional interactions rather than spurious correlations induced by field spread.

4. Frequency-Specific Network Isolation

Alpha and beta oscillations are differentially implicated in cognitive control and attentional states:

Alpha (8–12 Hz): Inverse marker of task engagement; dominant in DMN during mind-wandering and internally directed thought.

Beta (13–30 Hz): Correlates with active maintenance of working memory and task-focused attention, predominantly in CEN regions.

Bandpass filtering, followed by amplitude-squared extraction, captures the oscillatory power fluctuations driving network synchronization.

5. Functional Connectivity Estimation

Spearman rank correlations are computed in temporally segmented windows to quantify inter-electrode and inter-network synchrony. This rank-based approach is robust to non-Gaussian amplitude distributions and outliers, critical for EEG signals with highly skewed power spectra.

DMN Connectivity (Adj1): Aggregated correlations across canonical DMN electrodes, reflecting internally directed cognition and potential distractibility.

CEN Connectivity (Adj2): Aggregated correlations across task-positive executive control regions, reflecting focused engagement and active cognitive control.

Temporal averaging across sequential windows further stabilizes estimates, capturing enduring network states rather than transient, noise-driven fluctuations.

6. Cognitive State Inference

By contrasting DMN and CEN connectivity metrics:

High DMN, low CEN: Indicative of internal distraction or mind-wandering.

Low DMN, high CEN: Reflects sustained attention and task engagement.

Intermediate patterns: Suggest partial engagement or fluctuating focus.

This dual-network framework enables quantitative, reproducible indices of focus, engagement, and distraction, suitable for adaptive neurofeedback, workload monitoring, or cognitive state prediction in real-world tasks.

Significance

This project integrates state-of-the-art EEG preprocessing, advanced spatial filtering, and frequency-specific network analysis to push beyond conventional univariate metrics. By operationalizing cognitive states as dynamic inter-network interactions, it provides:

High temporal resolution: Captures rapid transitions in attention and engagement.

Network-level specificity: Focuses on physiologically meaningful cortical systems rather than isolated electrodes.

Robust statistical inference: Rank-based correlations and temporal aggregation reduce susceptibility to noise and artifacts.

Scalable framework: Applicable across datasets, electrode configurations, and cognitive paradigms.

The resulting framework represents a cutting-edge methodology for translating raw EEG signals into actionable measures of cognitive state, with potential applications in neuroergonomics, brain-computer interfaces, and adaptive learning systems.
