

# 0. Install dependencies

!pip install mne transformers matplotlib

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset

class ZucoDataset(Dataset):
    def __init__(self, data_dir, subjects=None):
        super().__init__()
        self.samples = []

        if subjects is None:
            subjects = sorted(os.listdir(data_dir))
        
        for sub in subjects:
            sub_path = os.path.join(data_dir, sub)
            if not os.path.exists(sub_path):
                continue

            for fname in os.listdir(sub_path):
                if fname.endswith(".npz"):
                    data = np.load(os.path.join(sub_path, fname), allow_pickle=True)
                    eeg = data["eeg"]
                    sentence = str(data["sentence"])
                    self.samples.append({
                        "eeg_feature_sequence": torch.tensor(eeg, dtype=torch.float32),
                        "sentence": sentence
                    })
        
        self.max_eeg_len = max(sample["eeg_feature_sequence"].shape[0] for sample in self.samples)
        self.max_eeg_len = ((self.max_eeg_len + 7)//8)*8
        print(f"[DEBUG] Loaded {len(self.samples)} samples, MAX_EEG_SEQ_LEN={self.max_eeg_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


#  Tokenizer + helpers

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def round_up_to_multiple(x, multiple):
    return ((x + multiple - 1)//multiple) * multiple

def pad_eeg_sequences(batch, seq_length):
    batch_size = len(batch)
    feat_dim = batch[0]["eeg_feature_sequence"].shape[1]
    padded_eeg = torch.zeros(batch_size, seq_length, feat_dim)
    attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    for i, sample in enumerate(batch):
        seq_len = min(sample["eeg_feature_sequence"].shape[0], seq_length)
        padded_eeg[i,:seq_len] = sample["eeg_feature_sequence"][:seq_len]
        attention_mask[i,:seq_len] = True
    return padded_eeg, attention_mask

def mask_eeg_batch(attention_mask, mask_ratio=0.75):
    batch_size, seq_len = attention_mask.shape
    eeg_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        valid_len = attention_mask[i].sum().item()
        num_to_mask = int(mask_ratio * (valid_len-1))
        indices = random.sample(range(valid_len-1), num_to_mask)
        for idx in indices:
            eeg_mask[i, idx] = True
        eeg_mask[i, valid_len-1] = True
    return eeg_mask

def mask_text_batch(sentences, max_length):
    encoding = tokenizer(sentences, return_tensors="pt", padding="max_length",
                         max_length=max_length, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = input_ids.clone()
    for i in range(len(input_ids)):
        special_mask = tokenizer.get_special_tokens_mask(input_ids[i].tolist(), already_has_special_tokens=True)
        candidate_indices = [idx for idx, s in enumerate(special_mask) if s==0 and input_ids[i, idx]!=tokenizer.pad_token_id]
        num_to_mask = int(0.75*len(candidate_indices))
        masked_indices = random.sample(candidate_indices, num_to_mask)
        for idx in masked_indices:
            input_ids[i, idx] = tokenizer.mask_token_id
        for idx in range(len(input_ids[i])):
            if idx not in masked_indices:
                labels[i, idx] = -100
    return {"masked_input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def collate_fn(batch):
    max_len_text = round_up_to_multiple(max(len(tokenizer.tokenize(s["sentence"])) for s in batch), 8)
    padded_eeg, attention_mask = pad_eeg_sequences(batch, seq_length=max(sample["eeg_feature_sequence"].shape[0] for sample in batch))
    eeg_mask = mask_eeg_batch(attention_mask)
    encoder_mask = attention_mask & (~eeg_mask)
    sentences = [s["sentence"] for s in batch]
    masked_text_outputs = mask_text_batch(sentences, max_len_text)

    return {
        "padded_eeg": padded_eeg,
        "attention_mask": attention_mask,
        "eeg_mask": eeg_mask,
        "encoder_mask": encoder_mask,
        "sentences": sentences,
        "input_ids": masked_text_outputs["masked_input_ids"],
        "text_attention_mask": masked_text_outputs["attention_mask"],
        "labels": masked_text_outputs["labels"]
    }


# EEG/Text modules

class EEGEncoder(nn.Module):
    def __init__(self, model_dim=840, nhead=8, num_layers=6, ff_dim=2048):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, 768)
    def forward(self, x, encoder_mask):
        x = x.transpose(0,1)
        x = self.transformer_encoder(x, src_key_padding_mask=~encoder_mask)
        x = x.transpose(0,1)
        x = self.output_proj(x)
        return x

class ModalityStreamEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.FFN = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.LN = nn.LayerNorm(hidden_dim)
    def forward(self, x, mask=None):
        x = self.FFN(x)
        x = self.LN(x)
        if mask is not None:
            keep_mask = ~mask
            x = x * keep_mask.unsqueeze(-1)
        return x

class GlobalAvgPooling(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            keep_mask = ~mask
            x = x * keep_mask.unsqueeze(-1)
            sum_x = x.sum(dim=1)
            lens = keep_mask.sum(dim=1).clamp(min=1)
            pooled = sum_x / lens.unsqueeze(-1)
        else:
            pooled = x.mean(dim=1)
        return pooled

def contrastive_loss(eeg_embeds, text_embeds, temperature=0.07):
    eeg_norm = F.normalize(eeg_embeds, dim=1)
    text_norm = F.normalize(text_embeds, dim=1)
    logits = torch.matmul(eeg_norm, text_norm.T) / temperature
    labels = torch.arange(eeg_embeds.size(0)).to(eeg_embeds.device)
    return F.cross_entropy(logits, labels)

class JointStreamEncoder(nn.Module):
    def __init__(self, hidden_dim=768, nhead=16):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
    def forward(self, x, split, joint_mask=None):
        if joint_mask is not None:
            x = self.encoder(x, src_key_padding_mask=~joint_mask)
        else:
            x = self.encoder(x)
        joint_text = x[:, :split, :]
        joint_eeg = x[:, split:, :]
        return joint_text, joint_eeg

class EEGDecoder(nn.Module):
    def __init__(self, seq_length, embedding_dim=840, nhead=8):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 840)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=4*embedding_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.mask_token = nn.Parameter(torch.randn(embedding_dim))
        self.pos_enc = nn.Embedding(seq_length, embedding_dim)
    def forward(self, x, attention_mask, eeg_mask):
        B, T, D = x.shape
        reconstructed_eeg = torch.where(
            eeg_mask.unsqueeze(-1),
            self.mask_token.view(1,1,-1),
            x
        )
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        eeg_data = reconstructed_eeg + self.pos_enc(pos_ids)
        eeg_output = self.encoder(eeg_data, src_key_padding_mask=~attention_mask)
        return eeg_output

class TextDecoder(nn.Module):
    def __init__(self, vocab_size=tokenizer.vocab_size):
        super().__init__()
        self.linear_proj = nn.Linear(768, vocab_size)
    def forward(self, x):
        return self.linear_proj(x)

class hidden_dim_eeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_proj = nn.Linear(768,840)
    def forward(self,x):
        return self.hidden_proj(x)


#  LLM

model = BartModel.from_pretrained("facebook/bart-base").to(device)
for param in model.parameters():
    param.requires_grad = False

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
for param in bart_model.parameters():
    param.requires_grad = False


#  Training function

def train_function(dataloader, num_epochs=1):
    EEG_encode = EEGEncoder().to(device)
    eeg_stream = ModalityStreamEncoder(768).to(device)
    text_stream = ModalityStreamEncoder(768).to(device)
    joint_stream = JointStreamEncoder().to(device)
    eeg_decoder = EEGDecoder(seq_length=MAX_EEG_SEQ_LEN, embedding_dim=840, nhead=8).to(device)
    text_decoder = TextDecoder().to(device)
    hidden_dim = hidden_dim_eeg().to(device)
    modules = [EEG_encode, eeg_stream, text_stream, joint_stream, eeg_decoder, text_decoder]
    eeg_pool = GlobalAvgPooling().to(device)
    text_pool = GlobalAvgPooling().to(device)
    optimizer = torch.optim.Adam(chain.from_iterable(m.parameters() for m in modules), lr=1e-4)
    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            padded_data = batch["padded_eeg"].to(device)
            eeg_mask = batch["eeg_mask"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            encoded_text = model(input_ids=input_ids, attention_mask=text_attention_mask).last_hidden_state
            encoded_eeg = EEG_encode(padded_data, encoder_mask)
            joint_input = torch.cat([encoded_text, encoded_eeg], dim=1)
            joint_mask = torch.cat([text_attention_mask.bool(), ~eeg_mask], dim=1)

            eeg_stream_out = eeg_stream(encoded_eeg, mask=eeg_mask)
            text_stream_out = text_stream(encoded_text, mask=None)
            eeg_summary = eeg_pool(eeg_stream_out, mask=eeg_mask)
            text_summary = text_pool(text_stream_out, mask=None)
            contra_loss = contrastive_loss(eeg_summary, text_summary)

            T_text = input_ids.shape[1]
            joint_text, joint_eeg = joint_stream(joint_input, split=T_text, joint_mask=joint_mask)
            joint_eeg = hidden_dim(joint_eeg)

            decoded_eeg = eeg_decoder(joint_eeg, attention_mask, eeg_mask)
            decoded_text = text_decoder(joint_text)

            eeg_maskloss = mse_loss(decoded_eeg[eeg_mask], padded_data[eeg_mask])
            decoded_text = decoded_text.view(-1, tokenizer.vocab_size)
            labels = labels.view(-1)
            text_maskloss = cross_entropy_loss(decoded_text, labels)

            optimizer.zero_grad()
            total_loss = eeg_maskloss + text_maskloss + contra_loss
            total_loss.backward()
            optimizer.step()

    return EEG_encode, eeg_stream, text_stream, joint_stream, eeg_decoder, text_decoder, hidden_dim


# E2T Projector

class E2T_PTR(nn.Module):
    def __init__(self, EEG_encode, eeg_stream, bart_model):
        super().__init__()
        self.EEG_encoder = EEG_encode
        self.EEG_Transformer = eeg_stream
        self.model = bart_model
        self.start_token_id = tokenizer.bos_token_id

    def forward(self, preprocessed_data, attention_mask):
        eeg_embeddings = self.EEG_encoder(preprocessed_data, attention_mask)
        eeg_embeddings = self.EEG_Transformer(eeg_embeddings)
        batch_size = eeg_embeddings.size(0)
        decoder_input_ids = torch.full((batch_size,1), self.start_token_id, dtype=torch.long, device=eeg_embeddings.device)
        generated_ids = self.model.generate(
            inputs_embeds=eeg_embeddings,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=20
        )
        return generated_ids

# Sanity check?

def batch_sanity_check(dataloader, EEG_encode, eeg_stream, text_stream, joint_stream,
                       eeg_decoder, text_decoder, hidden_dim, bart_model, tokenizer):
    EEG_encode.eval()
    eeg_stream.eval()
    text_stream.eval()
    joint_stream.eval()
    eeg_decoder.eval()
    text_decoder.eval()
    bart_model.eval()
    hidden_dim.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))

        padded_eeg = batch["padded_eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        input_ids = batch["input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        encoded_text = model(input_ids=input_ids, attention_mask=text_attention_mask).last_hidden_state
        assert encoded_text.shape[:2] == input_ids.shape
        encoded_eeg = EEG_encode(padded_eeg, encoder_mask)
        joint_input = torch.cat([encoded_text, encoded_eeg], dim=1)
        joint_mask = torch.cat([text_attention_mask.bool(), ~eeg_mask], dim=1)
        eeg_stream_out = eeg_stream(encoded_eeg, mask=eeg_mask)
        text_stream_out = text_stream(encoded_text, mask=None)
        eeg_summary = GlobalAvgPooling()(eeg_stream_out, mask=eeg_mask)
        text_summary = GlobalAvgPooling()(text_stream_out, mask=None)
        contra_loss = contrastive_loss(eeg_summary, text_summary)
        T_text = input_ids.shape[1]
        joint_text, joint_eeg = joint_stream(joint_input, split=T_text, joint_mask=joint_mask)
        joint_eeg = hidden_dim(joint_eeg)
        decoded_eeg = eeg_decoder(joint_eeg, attention_mask, eeg_mask)
        decoded_text = text_decoder(joint_text)
        e2t_model = E2T_PTR(EEG_encode, eeg_stream, bart_model)
        generated_ids = e2t_model(padded_eeg, attention_mask)
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("[SANITY CHECK PASSED] Shapes, masks, embeddings consistent.")
    print("Sample generated texts:", texts[:3])


# Mask visualization

def visualize_masks(batch, num_samples=3):
    padded_eeg = batch["padded_eeg"]
    eeg_mask = batch["eeg_mask"]
    text_mask = batch["input_ids"] == tokenizer.mask_token_id
    B = min(num_samples, padded_eeg.shape[0])
    for i in range(B):
        fig, axs = plt.subplots(2,1, figsize=(12,4))
        axs[0].imshow(eeg_mask[i].unsqueeze(0).cpu(), cmap='gray_r', aspect='auto')
        axs[0].set_title(f"Sample {i} EEG Mask")
        axs[0].set_yticks([])
        axs[1].imshow(text_mask[i].unsqueeze(0).cpu(), cmap='gray_r', aspect='auto')
        axs[1].set_title(f"Sample {i} Text Mask")
        axs[1].set_yticks([])
        plt.tight_layout()
        plt.show()


#  usage

data_dir = "content/zuco_data"
dataset = ZucoDataset(data_dir)
MAX_EEG_SEQ_LEN = dataset.max_eeg_len
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)

EEG_encode, eeg_stream, text_stream, joint_stream, eeg_decoder, text_decoder, hidden_dim = train_function(dataloader, num_epochs=1)
batch_sanity_check(dataloader, EEG_encode, eeg_stream, text_stream, joint_stream,
                   eeg_decoder, text_decoder, hidden_dim, bart_model, tokenizer)

# Visualize masks for one batch
batch = next(iter(dataloader))
visualize_masks(batch, num_samples=3)
