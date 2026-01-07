VAECGN-Dreamer

Variational Attention EEG Graph Convolutional Network with Reinforcement-Based Glimpse Policy

Overview

VAECGN-Dreamer is a deep learning framework for EEG signal decoding that integrates graph-based spatial modeling, variational attention, and a reinforcement learning-driven glimpse mechanism.

The model treats EEG as a spatial-temporal graph and learns task-relevant representations by selectively attending to informative channels and time segments.
A Dreamer-style reinforcement module guides the glimpse selection, enabling the network to focus on regions of EEG data that maximize downstream classification performance.

Evaluated on multi-subject EEG datasets, VAECGN-Dreamer achieves improved accuracy, more stable representations, and robust subject generalization compared to standard CNN, LSTM, and GCN baselines.

This repository contains the original architecture and reinforcement methodology.

Key Ideas

EEG signals are sparse and structured; not all channels or time segments carry equal information.

Variational attention enables uncertainty-aware feature selection, improving robustness across trials and subjects.

Graph Convolutional Networks (GCNs) model spatial relationships between EEG channels.

A reinforcement-guided glimpse mechanism allows dynamic temporal and spatial focusing:

The location network predicts glimpse positions

The GRU core integrates sequential glimpses

The value network provides baseline rewards to stabilize learning

Rewards are applied at the episode level to encourage long-term feature utility.

Model Overview

Input

EEG trials (channels × time samples)

Optional adjacency matrix for channel graph structure

Backbone

VAE-inspired EEG encoder: projects EEG into a latent space capturing uncertainty and trial variability

Graph convolution layers: capture spatial dependencies between electrodes

GRU core: processes glimpses sequentially to capture temporal dynamics

Glimpse Policy (Reinforcement)

Location network (policy): predicts the next patch or channel-time region to attend

Reward: classification accuracy at episode end

Value network (baseline): reduces variance of gradient estimates

Training: REINFORCE with decoupled optimization from classifier and encoder

Outputs

Trial-level classification (e.g., cognitive state, motor imagery, or target task)

Latent embeddings representing attended EEG patterns

Reinforcement Mechanism

Episodes: one EEG trial per episode

Action: glimpse selection (spatial, temporal, or both)

Policy update: REINFORCE with baseline

Exploration: stochastic sampling during training, deterministic evaluation during inference

Losses:

Classification loss for encoder/core output

REINFORCE loss for policy network

KL divergence for variational encoder regularization

This setup enables attention-driven feature extraction and sample-efficient learning without explicit channel or time labels.

Evaluation Metrics

Classification accuracy (trial-level)

Per-subject performance for generalization assessment

Confusion matrices for error analysis

Temporal attention consistency: do glimpses align with expected task-relevant EEG patterns?

Embedding separation: latent representations of different classes are well-separated

Datasets

VAECGN-Dreamer has been evaluated on EEG datasets with multi-subject structure:

Motor imagery: e.g., BCI Competition IV

Cognitive task EEG: custom multi-trial datasets

EEG preprocessing: bandpass filtering, normalization, and optional graph adjacency creation based on electrode layout

Mid-Results (Example)

Performance observed during preliminary experiments:

Single-subject accuracy: 72–80%

Across-subject generalization: very poor peformance

Temporal glimpse coverage: 75% of high-information regions captured

Latent representation clustering: visually separable embeddings per class

Observations:

Model reduces catastrophic misclassification by focusing on informative EEG patches

Variational attention mitigates subject-specific overfitting

Reinforcement-driven glimpses improve interpretability: policy often selects physiologically relevant channels and time windows

Highlights

Combines VAE, GCN, GRU, and REINFORCE in a single EEG framework

Learns dynamic attention policies without explicit supervision

Produces interpretable latent embeddings reflecting spatial-temporal EEG structure

Supports multi-subject training and generalization evaluation

Future Directions

Integrate contrastive losses to improve inter-subject alignment

Apply to real-time BCI decoding for adaptive attention-based feedback

Extend reinforcement reward to multi-step or hierarchical EEG tasks

Incorporate cross-modal EEG embeddings (e.g., with eye-tracking or behavior)
