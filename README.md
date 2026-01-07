# neuroAI-ML-projects

Research-driven NeuroAI and neural decoding projects focused on learning generalizable representations from EEG and other neural signals.

This repository explores how modern representation learning, contrastive objectives, and biologically informed inductive biases can be used to move beyond subject-specific decoding toward scalable, cross-subject neural models.

---

## 🔬 Research Focus

My work sits at the intersection of:
- EEG neural decoding
- Representation learning for brain signals
- Subject generalization and domain adaptation
- Multimodal alignment (EEG–Vision–Text)
- Transformer-based temporal modeling

A central question across projects is:

> **How can we learn neural representations that transfer across subjects, tasks, and modalities rather than overfitting to individual brains?**

---

## 🧠 Core Themes

- **Subject Generalization**  
  Learning EEG representations invariant to subject identity using contrastive learning, domain adaptation, and alignment losses.

- **Multimodal NeuroAI**  
  Aligning EEG with vision and language embeddings to ground neural activity in shared representational spaces.

- **Temporal Modeling of Brain Signals**  
  Transformers, GRUs, and convolutional temporal encoders for high-resolution EEG dynamics.

- **Biologically Motivated Inductive Biases**  
  Channel-wise modeling, spatial constraints, connectivity-based features, and graph-based reasoning.

---

## 📂 Repository Structure

Each folder corresponds to a self-contained research project, typically including:
- Problem formulation
- Dataset preprocessing
- Model architecture
- Training and evaluation code
- Experimental results and analysis

Example project areas include:
- EEG → Vision alignment via contrastive learning
- EEG-based gaze and motor decoding
- Connectivity-based neural representations
- Masked autoencoding and representation learning on EEG

---

## 🧪 Research Practices

Across projects, I aim to follow research-oriented ML practices:
- Subject-level train/validation/test splits
- Leave-one-subject-out (LOO) evaluation where applicable
- Explicit chance-level baselines
- Ablation and failure-mode analysis
- Reproducible experiments (fixed seeds, logged configs)

Accuracy alone is not treated as sufficient evidence of learning.

---

## 📊 Results & Evaluation

Results are reported using task-appropriate metrics such as:
- Classification accuracy vs. chance
- Cross-subject generalization gaps
- Retrieval accuracy in embedding space
- Representation clustering quality

Where relevant, I focus on **why** models fail on certain subjects rather than only reporting aggregate performance.

---


## 🔧 Environment

Most projects use:
- Python 3.9–3.11  
- PyTorch 2.x  
- NumPy, SciPy, MNE  
- CUDA-enabled training where applicable  

Dependencies are documented per project where needed.

---

## 🚧 Ongoing Work

Current directions under active development:
- Improving cross-subject alignment on large-scale EEG datasets
- Stronger multimodal contrastive objectives
- Reducing subject leakage in representation learning
- Exploring scaling behavior of EEG transformers

---

## 📬 Contact

This repository is intended as a research portfolio for NeuroAI and applied ML roles.  
Questions, feedback, or discussion are welcome via GitHub Issues or direct contact.
