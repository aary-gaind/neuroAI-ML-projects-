E2T-PTR

Brain Signal-to-Text Generation Using Masked Autoencoding and LLM Prompt Injection

This repository implements a state-of-the-art EEG-to-Text pipeline that integrates masked autoencoding, contrastive alignment between EEG and text embeddings, and prompt injection into a frozen large language model (LLM) for natural language generation directly from neural activity. The system is designed to handle spatiotemporal EEG sequences, reconstruct masked signals, and generate semantically aligned textual outputs.

Architecture Overview
1. EEG Representation and Masking

EEG signals are modeled as spatiotemporal sequences of shape [time, channels].

Masked Autoencoding is applied to EEG tokens:

Random masking of 75% of EEG tokens per sequence.

Sentence-level tokens are always masked to encourage global contextual reconstruction.

Positional embeddings are added to preserve temporal structure.

Masked tokens are replaced with a learnable mask embedding during decoding.

2. Text Representation and Masking

Text is tokenized using Facebook BART tokenizer.

75% of non-special tokens are masked to emulate masked language modeling (MLM) during training.

Special tokens and padding are excluded from masking.

Text tokens are projected into a 768-dimensional embedding space aligned with EEG embeddings.

3. EEG Encoder

Implemented as a multi-layer Transformer encoder:

Input: padded EEG sequences [B, T, D]

Output: 768-dimensional embeddings [B, T, 768]

Encoder ignores masked tokens during attention via src_key_padding_mask.

Projects EEG to the shared embedding space for downstream alignment.

4. Modality-Specific Streams

EEG Stream Encoder and Text Stream Encoder:

Both use feedforward + layer normalization blocks.

Masked EEG embeddings are zeroed to prevent information leakage.

Global Average Pooling produces sequence-level embeddings for contrastive alignment.

5. Contrastive Alignment

Contrastive loss aligns EEG embeddings with corresponding text embeddings:

Normalizes embeddings (L2 normalization).

Computes similarity logits via dot product scaled by temperature.

Cross-entropy loss encourages diagonal alignment (perfect EEG→Text mapping).

6. Joint Transformer

Concatenates EEG and text embeddings along the sequence dimension.

Processes combined sequences through a joint Transformer encoder.

Applies joint attention mask to ensure padded and masked tokens are ignored.

Splits outputs back into EEG and text streams for individual decoding.

7. EEG Decoder

Reconstructs masked EEG tokens using a Transformer encoder layer.

Adds learned positional encodings to maintain temporal structure.

Output projected back to original EEG feature dimensionality.

8. Text Decoder

Projects joint text embeddings back to token logits using a linear layer.

Supports CrossEntropyLoss training against masked tokens.

9. EEG-to-Text Prompt Injection (E2T-PTR)

Frozen BART LLM receives EEG-derived embeddings via a projector.

Starts generation with BOS token.

Outputs natural language sequences aligned with original EEG semantics.

Fully differentiable pipeline except for frozen LLM weights.

Data Handling

Dataset: ZuCo or other EEG-text aligned corpora.

Collate function:

Pads EEG sequences to MAX_EEG_SEQ_LEN, aligned to multiples of 8.

Pads text sequences to nearest multiple of 8.

Returns padded_eeg, attention_mask, eeg_mask, encoder_mask, masked input_ids, labels, and text_attention_mask.

Visualization utilities available to inspect EEG and text masking per batch.

Training Loop

Trains EEG encoder, modality streams, joint transformer, decoders, and EEG→embedding projector.

Losses:

EEG Reconstruction (MSE) on masked tokens.

Text MLM (CrossEntropy) on masked tokens.

Contrastive Loss between EEG and text embeddings.

Supports GPU acceleration with torch.device("cuda").

Sanity Checks & Debugging

Includes automatic batch sanity check:

Validates shapes of embeddings, masks, and outputs.

Confirms EEG masking and text masking are applied correctly.

Generates sample texts using the frozen LLM for inspection.

Includes mask visualization function:

Plots EEG and text masks for each sample.

White = masked, Black = visible tokens.

Confirms proper masking proportions.

Usage Example
from torch.utils.data import DataLoader
dataset = ZucoDataset("./zuco_data")
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)

EEG_encode, eeg_stream, text_stream, joint_stream, eeg_decoder, text_decoder, hidden_dim = train_function(dataloader, num_epochs=1)

batch_sanity_check(dataloader, EEG_encode, eeg_stream, text_stream, joint_stream,
                   eeg_decoder, text_decoder, hidden_dim, bart_model, tokenizer)

batch = next(iter(dataloader))
visualize_masks(batch, num_samples=3)
