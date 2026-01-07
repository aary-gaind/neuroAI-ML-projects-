Neural Signal-to-Language Mapping via Multimodal Embedding Alignment

This repository implements a brain-to-language decoding pipeline that maps high-dimensional EEG time series into CLIP’s visual-language embedding space and subsequently projects these embeddings into a frozen large language model (LLM) token space for natural language generation. The system demonstrates multimodal representation alignment between neural signals, visual semantics, and language.

1. Architecture Overview
1.1 EEG Feature Encoding

The EEG encoder is a deep spatiotemporal convolutional architecture designed to capture both temporal dynamics and spatial dependencies across EEG channels:

TemporalBlock: Multi-layer 2D dilated convolutions with residual connections. Each layer applies variable dilation rates (1, 1), (1, 2), (1, 4), ... to extract features across multiple temporal scales.

SpatialBlock: 2D convolutions across the channel dimension, capturing spatial correlations across electrodes.

ResidualBlocks + Downsampling: Stack of residual modules with downsampling convolutional layers to refine feature representations and reduce dimensionality.

Classifier Head: Fully connected layers projecting the flattened spatiotemporal features into an intermediate embedding space (D=512), aligned with CLIP embeddings.

Mathematically:
hEEG​=fproj​(fres​(fspatial​(ftemporal​(X))))

where 𝑋 ∈ 𝑅 (𝐶 × 𝐻 × 𝑊) is the EEG tensor.

1.2 CLIP Embedding Alignment We leverage OpenAI CLIP (ViT-B/32) to encode visual semantics into a joint image-text embedding space: 
ℎ_𝐶 𝐿 𝐼 𝑃 = 𝐶 𝐿 𝐼 𝑃_𝑖 𝑚 𝑎 𝑔 𝑒 ( 𝐼 )

The EEG encoder is trained to minimize MSE between projected EEG embeddings and CLIP image embeddings:
L_MSE​= 1/N (N∑ i=1   ∥hEEG(i)​−hCLIP(i)​∥_2 ^2


an object classification loss is used as well
The combined loss ensures both semantic alignment and class discriminability.

1.3 Projection into LLM Embedding Space To integrate EEG embeddings with a frozen LLM: A linear projector 𝑃 : 𝑅 𝐷 𝐸 𝐸 𝐺 → 𝑅 𝐷 𝐿 𝐿 𝑀 P:R D EEG ​ →R D LLM ​ maps EEG embeddings into the token embedding space of the LLM. Special tokens <image> and <object_string> are used in the prompt template. During training, these tokens’ embeddings are replaced with projected EEG embeddings and object label embeddings, enabling the LLM to condition on neural signals.
​
Formally:
ELLMinput​[:,timage​]=P(hEEG​),ELLMinput​[:,tobj​]= ELLM​(object label)

1.4 Prompt Template

We use a consistent prompt template for training and inference:

"<image> <object_string> Describe this in one sentence:"


<image>: replaced with EEG-projected embedding

<object_string>: replaced with embedding of object label

The LLM generates natural language captions conditioned on both EEG and object semantics.

2. Training Pipeline

EEG Encoder + Classifier

Input: raw EEG (C, H, W)

Output: EEG embedding + object logits

Loss: MSE(EEG -> CLIP) + CrossEntropy(object classification)

LLM Fine-tuning with EEG Conditioning

Freeze LLM weights; only train the projector.

Replace <image> and <object_string> tokens with embeddings.

Optimize cross-entropy loss on text generation with masked prompts.

Data Loading and Collation

Custom collate_fn handles:

Batched EEG embeddings

Tokenization of prompts + object labels

Label masking to ignore prompt tokens

Optimization

Adam optimizer with gradient clipping

Learning rate scheduling can be applied for projector stability

3. Verification & Debugging Utilities

Forward Pass Tests: validate shapes of EEG embeddings, projected embeddings, and classifier logits.

Embedding Heatmaps: visualize EEG embeddings for batch inspection.

Token Embedding Inspection: confirm <image> and <object_string> replacements in LLM input.

Batch Inference + Cosine Similarity: compute alignment between EEG-projected embeddings and CLIP embeddings.

Caption Table: tabular view of object labels, embeddings norms, and generated captions.

4. Inference

The generate_caption function supports single-sample EEG → text generation:

caption = generate_caption(eeg_sample, "piano", eeg_classifier, projector, llm, tokenizer)
print(caption)


For batch inference with visualization:

captions, projected_embeddings, similarity = batch_inference_and_visualize(
    eeg_batch, object_labels, hclip_batch, eeg_classifier, projector, llm, tokenizer
)


This enables full pipeline verification and alignment inspection between neural signals and visual-language semantics.

