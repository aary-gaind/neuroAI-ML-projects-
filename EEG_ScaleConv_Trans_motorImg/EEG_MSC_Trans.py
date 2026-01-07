import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mne
import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


# EEG Dataset preprocessing


def preprocess_gdf(filepath, event_ids=[5,6,7,8], window_size_sec=3, step_size_sec=0.25, max_windows=6):
    raw = mne.io.read_raw_gdf(filepath, eog=['EOG-left','EOG-central','EOG-right'], preload=True)
    raw.drop_channels(['EOG-left','EOG-central','EOG-right'])
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=-0.25, tmax=4.25, preload=True)
    X = epochs.get_data()
    y = epochs.events[:,2]
    label_mapping = {label:i for i,label in enumerate(np.unique(y))}
    y_mapped = np.array([label_mapping[label] for label in y])
    sfreq = epochs.info['sfreq']
    window_size = int(window_size_sec*sfreq)
    step_size = int(step_size_sec*sfreq)
    X_slid, y_slid = [], []
    for trial,label in zip(X, y_mapped):
        n_times = trial.shape[1]
        start,end,win_cnt = 0, window_size, 0
        while end <= n_times and win_cnt < max_windows:
            X_slid.append(trial[:,start:end])
            y_slid.append(label)
            start += step_size
            end += step_size
            win_cnt += 1
    return np.stack(X_slid), np.array(y_slid)

# Load example subjects
X1, y1 = preprocess_gdf('/content/data/A04T.gdf')

# Second subject with MAT labels
raw = mne.io.read_raw_gdf('/content/data/A04E.gdf', eog=['EOG-left','EOG-central','EOG-right'], preload=True)
raw.drop_channels(['EOG-left','EOG-central','EOG-right'])
events,_ = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id={'783':7}, tmin=-0.25, tmax=4.25, preload=True)
X = epochs.get_data()
mat = scipy.io.loadmat('A04E.mat')
y = mat['classlabel'].squeeze()
label_mapping = {label:i for i,label in enumerate(np.unique(y))}
y_mapped = np.array([label_mapping[label] for label in y])

window_size_sec, step_size_sec, max_windows = 3, 0.25, 6
sfreq = epochs.info['sfreq']
window_size = int(window_size_sec*sfreq)
step_size = int(step_size_sec*sfreq)
X_slid, y_slid = [], []
for trial,label in zip(X, y_mapped):
    n_times = trial.shape[1]
    start,end,win_cnt = 0, window_size, 0
    while end <= n_times and win_cnt < max_windows:
        X_slid.append(trial[:,start:end])
        y_slid.append(label)
        start += step_size
        end += step_size
        win_cnt += 1

X2 = np.stack(X_slid)
y2 = np.array(y_slid)
X_combined = np.concatenate([X1, X2], axis=0)
y_combined = np.concatenate([y1, y2], axis=0)
print("Combined X shape:", X_combined.shape, "Combined y shape:", y_combined.shape)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.5, random_state=42, stratify=y_combined)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_dataset.tensors[0].shape[0])
test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])


# CNN + Transformer EEG model

class EEGBranch(nn.Module):
    def __init__(self, C, F1, Kc, P):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, F1, kernel_size=(1,Kc), padding=(0,Kc//2), bias=False)
        self.spatial_conv = nn.Conv2d(F1, F1, kernel_size=(C,1), groups=F1, bias=False)
        self.batch_norm = nn.BatchNorm2d(F1)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1,P), stride=P)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = F.pad(x, pad=(50,50))
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.squeeze(2)
        x = x.permute(0,2,1)
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim*heads==embed_size
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = self.values(values).reshape(N,value_len,self.heads,self.head_dim)
        keys = self.keys(keys).reshape(N,key_len,self.heads,self.head_dim)
        queries = self.queries(query).reshape(N,query_len,self.heads,self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))
        attention = torch.softmax(energy/(self.head_dim**0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N,query_len,self.heads*self.head_dim)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,value,key,query,mask):
        attention = self.attention(value,key,query,mask)
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super().__init__()
        self.device = device
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_size))
    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        cls_tokens = self.cls_token.expand(N,1,-1)
        x = torch.cat([cls_tokens, x], dim=1)
        positions = torch.arange(0, seq_length+1).unsqueeze(0).expand(N, seq_length+1).to(self.device)
        out = self.dropout(x + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out,out,out,mask)
        return out

class MultiBranchEEGEncoder(nn.Module):
    def __init__(self, C, F1, kernel_sizes, P, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super().__init__()
        assert embed_size == F1*len(kernel_sizes)
        self.branches = nn.ModuleList([EEGBranch(C,F1,Kc,P) for Kc in kernel_sizes])
        self.encoder = Encoder(embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
    def forward(self,x):
        branch_outputs = [b(x) for b in self.branches]
        x_cat = torch.cat(branch_outputs, dim=-1)
        encoded = self.encoder(x_cat, mask=None)
        return encoded

class EEGClassifier(nn.Module):
    def __init__(self, encoder, embed_size, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(p=0.25)
    def forward(self,x):
        encoded = self.encoder(x)
        cls_embedding = encoded[:,0,:]
        out = self.fc1(self.dropout(cls_embedding))
        return out

# Instantiate model

encoder = MultiBranchEEGEncoder(
    C=22, F1=16, kernel_sizes=[45,65,85], P=52,
    embed_size=48, num_layers=5, heads=8, device=device,
    forward_expansion=4, dropout=0.25, max_length=22
)
model = EEGClassifier(encoder=encoder, embed_size=48, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training loop

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0,0,0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0,0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            out = model(x)
            loss = criterion(out,y)
            total_loss += loss.item()*x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return total_loss/total, correct/total


# Train model 

num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")


# Test evaluation

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# Research-level analysis

#  Attention maps
def plot_attention_maps(model, x_sample, head=0):
    model.eval()
    x_sample = x_sample.to(device).unsqueeze(0).unsqueeze(1)
    x_cat = torch.cat([b(x_sample) for b in model.encoder.branches], dim=-1)
    cls_token = model.encoder.encoder.cls_token.expand(1,1,-1)
    x_with_cls = torch.cat([cls_token, x_cat], dim=1)
    positions = torch.arange(0,x_cat.shape[1]+1).unsqueeze(0).to(device)
    out = x_with_cls + model.encoder.encoder.position_embedding(positions)
    first_layer = model.encoder.encoder.layers[0]
    q = first_layer.attention.queries(out)
    k = first_layer.attention.keys(out)
    N, seq_len_plus1, embed_size = q.shape
    head_dim = embed_size // first_layer.attention.heads
    q = q.reshape(N,seq_len_plus1,first_layer.attention.heads,head_dim)
    k = k.reshape(N,seq_len_plus1,first_layer.attention.heads,head_dim)
    attn_scores = torch.einsum('nqhd,nkhd->nhqk',[q,k])/(head_dim**0.5)
    attn_scores = torch.softmax(attn_scores, dim=-1)
    plt.figure(figsize=(8,6))
    plt.imshow(attn_scores[0, head].cpu(), cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Head {head} - Sample 0')
    plt.xlabel('Key sequence positions')
    plt.ylabel('Query sequence positions')
    plt.show()

x_sample, _ = next(iter(test_loader))
plot_attention_maps(model, x_sample[0])

#  Branch activations
def plot_branch_activations(model, x_sample):
    model.eval()
    x_sample = x_sample.to(device).unsqueeze(0).unsqueeze(1)
    branch_outputs = [b(x_sample) for b in model.encoder.branches]
    for i, branch_out in enumerate(branch_outputs):
        mean_over_features = branch_out.mean(dim=-1).squeeze().cpu().numpy()
        plt.figure(figsize=(8,3))
        plt.plot(mean_over_features)
        plt.title(f'Branch {i} Temporal Activation')
        plt.xlabel('Time tokens')
        plt.ylabel('Mean activation')
        plt.show()

plot_branch_activations(model, x_sample[0])

#  Channel importance
def plot_channel_importance(model, branch_idx=0):
    weights = model.encoder.branches[branch_idx].spatial_conv.weight
    channel_importance = weights.abs().sum(dim=(0,1,3)).cpu().numpy()
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(channel_importance)), channel_importance)
    plt.xlabel('EEG Channel')
    plt.ylabel('Importance')
    plt.title(f'Branch {branch_idx} Channel Contribution')
    plt.show()

plot_channel_importance(model, branch_idx=0)