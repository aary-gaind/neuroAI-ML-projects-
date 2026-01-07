import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Step 0: Load CLIP
# -----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_hclip_batch(image_paths):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        hclip = clip_model.get_image_features(**inputs)  # shape: (B, D)
    return hclip

# -----------------------------
# Step 1: EEG Feature Extractor
# -----------------------------
from layers import *  # make sure TemporalBlock, SpatialBlock, ResidualBlock, ConvLayer2D exist

class FeaturesExtractor(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride):
        super().__init__()
        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temporal_layers, temporal_kernel, temporal_stride, temporal_dilation_list, input_width
        )
        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers, out_channels, num_spatial_layers, spatial_stride, in_height
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers, down_kernel, down_stride, 0, 1
                )
            ) for _ in range(num_residual_blocks)
        ])
        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1
        )

    def forward(self, x):
        out = self.temporal_block(x)
        out = self.spatial_block(out)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.final_conv(out)
        return out

class Model(nn.Module):
    def __init__(self, in_channels=1, temp_channels=10, out_channels=50, num_classes=40, embedding_size=1000,
                 input_width=440, input_height=128, temporal_dilation_list=[(1,1),(1,2),(1,4),(1,8),(1,16)],
                 temporal_kernel=(1,33), temporal_stride=(1,2),
                 num_temp_layers=4, num_spatial_layers=4, spatial_stride=(2,1),
                 num_residual_blocks=4, down_kernel=3, down_stride=2):
        super().__init__()
        self.encoder = FeaturesExtractor(in_channels, temp_channels, out_channels, input_width, input_height,
                                        temporal_kernel, temporal_stride,
                                        temporal_dilation_list, num_temp_layers,
                                        num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride
                                        )
        # Determine encoding size dynamically
        dummy = torch.zeros(1, in_channels, input_height, input_width)
        encoding_size = self.encoder(dummy).contiguous().view(-1).size(0)

        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, embedding_size),
            nn.ReLU(True),
            nn.Linear(embedding_size, num_classes),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out


# EEG Classifier

class EEGClassifier(nn.Module):
    def __init__(self, eeg_encoder, embedding_dim, num_classes):
        super().__init__()
        self.encoder = eeg_encoder
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        heeg = self.encoder(x)  # (B, embedding_dim)
        logits = self.classifier(heeg)  # (B, num_classes)
        return heeg, logits


# Projector EEG -> LLM

class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


#  EEG Dataset

class EEGDataset(Dataset):
    def __init__(self, eeg_data_list, labels_list, hclip_list):
        assert len(eeg_data_list) == len(labels_list) == len(hclip_list)
        self.eeg_data = eeg_data_list
        self.labels = labels_list
        self.hclip = hclip_list

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx], self.hclip[idx]


#  LLM + Tokenizer

model_name = "gpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)

special_tokens_dict = {"additional_special_tokens": ["<image>", "<object_string>"]}
tokenizer.add_special_tokens(special_tokens_dict)
llm.resize_token_embeddings(len(tokenizer))

prompt_template = "<image> <object_string> Describe this in one sentence:"


# Collate Function

def collate_fn(batch):
    eeg_embeddings = torch.stack([item["Heeg"] for item in batch])
    object_labels = [item["object_label"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    captions = [item["caption"] for item in batch]

    # 1. Full chat
    full_chats = [
        tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": caption},
        ], tokenize=False)
        for prompt, caption in zip(prompts, captions)
    ]

    tokenized = tokenizer(full_chats, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    # Mask prompt
    prompt_only_chats = [
        tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ], tokenize=True)["input_ids"]
        for prompt in prompts
    ]
    prompt_lengths = [len(x) for x in prompt_only_chats]
    labels = input_ids.clone()
    for i, length in enumerate(prompt_lengths):
        labels[i, :length] = -100

    # Object label tokens
    object_label_tokens = tokenizer(object_labels, padding=True, truncation=True, return_tensors="pt")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "eeg_embeddings": eeg_embeddings,
        "object_label_ids": object_label_tokens.input_ids,
        "labels": labels,
    }

# Training Function

def train_model(train_dataset, num_epochs=10, batch_size=4, eeg_embedding_dim=512, projector_hidden_dim=1024, num_classes=20):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # EEG -> CLIP classifier
    eeg_encoder = Model()
    eeg_classifier = EEGClassifier(eeg_encoder, embedding_dim=eeg_embedding_dim, num_classes=num_classes).to(device)

    # Projector
    projector = Projector(eeg_embedding_dim, llm.config.hidden_size).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(list(eeg_classifier.parameters()) + list(projector.parameters()) + list(llm.parameters()), lr=1e-4)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            eeg_embeddings = batch["eeg_embeddings"].to(device)
            object_input_ids = batch["object_label_ids"].to(device)

            # EEG forward
            Heeg, logits = eeg_classifier(eeg_embeddings)
            loss_eeg = mse_loss(Heeg, eeg_embeddings) + ce_loss(logits, torch.argmax(logits, dim=1))

            # LLM embeddings
            token_embeddings = llm.get_input_embeddings()(input_ids)

            # Replace <image> tokens with projected EEG
            image_token_id = tokenizer.convert_tokens_to_ids("<image>")
            image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
            projected = projector(Heeg)
            token_embeddings[image_positions] = projected[image_positions[0]]

            # Replace <object_string> tokens with first object token embedding
            object_token_id = tokenizer.convert_tokens_to_ids("<object_string>")
            object_positions = (input_ids == object_token_id).nonzero(as_tuple=True)
            object_embeds = llm.get_input_embeddings()(object_input_ids)
            token_embeddings[object_positions] = object_embeds[:, 0][object_positions[0]]

            # LLM forward
            outputs = llm(inputs_embeds=token_embeddings, attention_mask=attention_mask, labels=labels)
            loss_llm = outputs.loss

            # Total loss
            total_loss = loss_eeg + loss_llm

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss.item():.4f}")

    return eeg_classifier, projector, llm


@torch.no_grad()
def generate_caption(eeg_sample, object_label, eeg_classifier, projector, llm, tokenizer, prompt_template="<image> <object_string> Describe this in one sentence:", max_length=50):
    """
    eeg_sample: torch.Tensor, shape (1, channels, height, width) - single EEG input
    object_label: str, the object label to insert into <object_string>
    eeg_classifier: trained EEGClassifier
    projector: trained EEG->LLM projector
    llm: trained/fine-tuned LLM
    tokenizer: corresponding tokenizer
    prompt_template: string with <image> <object_string>
    max_length: max tokens to generate
    """
    eeg_classifier.eval()
    projector.eval()
    llm.eval()

    #  Get EEG embedding
    Heeg, _ = eeg_classifier(eeg_sample.to(device))  # (1, embedding_dim)

    #  Build input prompt
    prompt_text = prompt_template.replace("<image>", "<image>").replace("<object_string>", "<object_string>")
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    # Replace <image> token
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
    projected = projector(Heeg)  # (1, hidden_size)
    token_embeddings = llm.get_input_embeddings()(input_ids)
    token_embeddings[image_positions] = projected[image_positions[0]]

    # Replace <object_string> token
    object_token_id = tokenizer.convert_tokens_to_ids("<object_string>")
    object_positions = (input_ids == object_token_id).nonzero(as_tuple=True)
    object_input_ids = tokenizer(object_label, return_tensors="pt").input_ids.to(device)
    object_embeds = llm.get_input_embeddings()(object_input_ids)
    token_embeddings[object_positions] = object_embeds[:, 0][object_positions[0]]

    # Generate caption using LLM
    generated_ids = llm.generate(
        inputs_embeds=token_embeddings,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    #  Decode text
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption
    
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(" Starting verification tests ")

    #  Simulate dummy EEG data
    batch_size = 2
    channels, height, width = 1, 128, 440
    dummy_eeg = torch.randn(batch_size, channels, height, width)

    # Simulate dummy labels
    dummy_labels = torch.randint(0, 20, (batch_size,))
    dummy_hclip = torch.randn(batch_size, 512)  # same size as CLIP embeddings

    # Create Dataset + DataLoader
    train_dataset = EEGDataset([dummy_eeg[i] for i in range(batch_size)],
                               [dummy_labels[i] for i in range(batch_size)],
                               [dummy_hclip[i] for i in range(batch_size)])
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

    #  Instantiate models
    eeg_encoder = Model()
    eeg_classifier = EEGClassifier(eeg_encoder, embedding_dim=512, num_classes=20).to(device)
    projector = Projector(512, llm.config.hidden_size).to(device)

    #  Forward pass through EEG classifier
    for X, y, Hclip in loader:
        X = X.to(device)
        y = y.to(device)
        Hclip = Hclip.to(device)

        heeg, logits = eeg_classifier(X)
        print("EEG embedding shape:", heeg.shape)  # should be (batch_size, 512)
        print("Logits shape:", logits.shape)       # should be (batch_size, num_classes)

        #  Project EEG embeddings
        projected = projector(heeg)
        print("Projected embedding shape (LLM dim):", projected.shape)  # (batch_size, hidden_size)

        #  Simple check: loss computation
        mse_loss = nn.MSELoss()(heeg, Hclip)
        print("MSE loss (EEG -> Hclip):", mse_loss.item())
        break


    #  Token embedding inspection

    prompt = "<image> <object_string> Describe this in one sentence:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    token_embeddings = llm.get_input_embeddings()(input_ids)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    object_token_id = tokenizer.convert_tokens_to_ids("<object_string>")

    print("Token IDs for <image> and <object_string>:", image_token_id, object_token_id)
    print("Token embeddings shape:", token_embeddings.shape)  # (1, seq_len, hidden_size)


    #  Generate  caption

    dummy_eeg_sample = torch.randn(1, channels, height, width)
    dummy_object_label = "piano"
    caption = generate_caption(dummy_eeg_sample, dummy_object_label,
                               eeg_classifier, projector, llm, tokenizer)
    print("Generated Caption (dummy):", caption)

    #  Visualization
  
    # Compare embeddings visually
    plt.figure(figsize=(6,4))
    plt.title("Dummy EEG embedding heatmap")
    plt.imshow(heeg.detach().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel("Embedding dimension")
    plt.ylabel("Batch")
    plt.show()

    print("Verification complete")
    
    
    
#  Batch inference + visualization

@torch.no_grad()
def batch_inference_and_visualize(eeg_batch, object_labels, hclip_batch, 
                                  eeg_classifier, projector, llm, tokenizer,
                                  prompt_template="<image> <object_string> Describe this in one sentence:"):
    """
    eeg_batch: torch.Tensor (B, channels, height, width)
    object_labels: list of strings length B
    hclip_batch: torch.Tensor (B, clip_dim) optional, can be None
    """
    eeg_classifier.eval()
    projector.eval()
    llm.eval()

    batch_size = eeg_batch.size(0)
    captions = []
    projected_embeddings = []

    for i in range(batch_size):
        eeg_sample = eeg_batch[i:i+1]  # (1, C, H, W)
        object_label = object_labels[i]
        caption = generate_caption(eeg_sample, object_label, eeg_classifier, projector, llm, tokenizer, prompt_template)
        captions.append(caption)

        # store projected EEG embedding
        Heeg, _ = eeg_classifier(eeg_sample.to(device))
        projected = projector(Heeg)
        projected_embeddings.append(projected.squeeze(0).cpu())

    projected_embeddings = torch.stack(projected_embeddings)  # (B, hidden_size)

    # Compute similarity to CLIP embeddings if provided
    if hclip_batch is not None:
        hclip_batch = hclip_batch.to(device)
        # normalize
        proj_norm = projected_embeddings / projected_embeddings.norm(dim=1, keepdim=True)
        hclip_norm = hclip_batch / hclip_batch.norm(dim=1, keepdim=True)
        similarity = torch.mm(proj_norm, hclip_norm.T)  # (B, B)
    else:
        similarity = None

    # Visualization
    import matplotlib.pyplot as plt
    import pandas as pd

    #
    print("\n=== Generated Captions ===")
    for i, cap in enumerate(captions):
        print(f"[{i}] Object: {object_labels[i]} --> Caption: {cap}")

    #
    if similarity is not None:
        plt.figure(figsize=(6,5))
        plt.title("EEG projected embedding vs CLIP similarity")
        plt.imshow(similarity.cpu().numpy(), cmap="viridis")
        plt.colorbar(label="Cosine similarity")
        plt.xlabel("CLIP samples")
        plt.ylabel("EEG samples")
        plt.show()

    # table of embeddings norms
    embedding_norms = projected_embeddings.norm(dim=1)
    df = pd.DataFrame({
        "Object": object_labels,
        "Caption": captions,
        "Embedding Norm": embedding_norms.numpy()
    })
    print("\n=== Embedding norms and captions ===")
    print(df)

    return captions, projected_embeddings, similarity
    
#random test     
batch_size = 3
channels, height, width = 1, 128, 440
dummy_eeg_batch = torch.randn(batch_size, channels, height, width)
dummy_object_labels = ["piano", "cat", "guitar"]
dummy_hclip_batch = torch.randn(batch_size, 512)  # for similarity

captions, projected_embeddings, similarity = batch_inference_and_visualize(
    dummy_eeg_batch,
    dummy_object_labels,
    dummy_hclip_batch,
    eeg_classifier,
    projector,
    llm,
    tokenizer
)
