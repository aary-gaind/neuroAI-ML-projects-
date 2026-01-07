import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mne
import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# GPU setup

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# Data preprocessing

def preprocess_gdf(filepath, matpath=None, event_ids=[5,6,7,8], 
                   window_size_sec=3, step_size_sec=0.25, max_windows=6):
    raw = mne.io.read_raw_gdf(filepath, eog=['EOG-left','EOG-central','EOG-right'], preload=True)
    raw.drop_channels(['EOG-left','EOG-central','EOG-right'])
    events, _ = mne.events_from_annotations(raw)

    if matpath is None:
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=-0.25, tmax=4.25, preload=True)
        X = epochs.get_data()
        y = epochs.events[:, 2]
    else:
        epochs = mne.Epochs(raw, events, event_id={'783': 7}, tmin=-0.25, tmax=4.25, preload=True)
        X = epochs.get_data()
        mat = scipy.io.loadmat(matpath)
        y = mat['classlabel'].squeeze()

    # Map labels to 0..N-1
    unique_labels = np.unique(y)
    label_mapping = {label:i for i,label in enumerate(unique_labels)}
    y_mapped = np.array([label_mapping[label] for label in y])

    # Sliding windows
    sfreq = epochs.info['sfreq']
    window_size = int(window_size_sec*sfreq)
    step_size = int(step_size_sec*sfreq)

    X_slid, y_slid = [], []
    for trial_data, label in zip(X, y_mapped):
        start, windows_in_trial = 0, 0
        end = window_size
        n_times = trial_data.shape[1]
        while end <= n_times and windows_in_trial < max_windows:
            segment = trial_data[:, start:end]
            X_slid.append(segment)
            y_slid.append(label)
            start += step_size
            end += step_size
            windows_in_trial += 1

    return np.stack(X_slid), np.array(y_slid)


# CNN module

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(22,16,kernel_size=125,padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.avgpool1 = nn.AvgPool1d(kernel_size=4, stride=1)
        self.dropout1 = nn.Dropout(p=0.5)

        self.sep_block = nn.Sequential(
            nn.Conv1d(16,16,kernel_size=16, groups=16, padding='same'),
            nn.Conv1d(16,24,kernel_size=1),
            nn.BatchNorm1d(24),
            nn.ReLU()
        )
        self.avgpool2 = nn.AvgPool1d(kernel_size=8,stride=1)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self,x):
        # x: [batch, channels, time]
        x = self.block1(x)
        flat1 = x.view(x.size(0), -1)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.sep_block(x)
        flat2 = x.view(x.size(0), -1)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        flat_out = x.view(x.size(0), -1)
        return flat_out, flat1, flat2


# LSTM module

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=22, hidden_size=50, num_layers=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        x = x.permute(2,0,1)  # [time, batch, channels]
        out, (hn, cn) = self.lstm(x)
        final_hidden = hn[-1]        # [batch, hidden]
        final_hidden = self.dropout(final_hidden)
        return final_hidden


# Fully connected fusion network

class FNN(nn.Module):
    def __init__(self,num_classes=4):
        super().__init__()
        self.cnn = CNN()
        self.lstm = LSTM()

        # Infer total features with dummy input
        with torch.no_grad():
            dummy = torch.zeros(1,22,750)
            f_conv, f1, f2 = self.cnn(dummy)
            f_lstm = self.lstm(dummy)
        total_features = f_conv.size(1) + f1.size(1) + f2.size(1) + f_lstm.size(1)

        self.fc = nn.Linear(total_features,num_classes)

    def forward(self,x):
        f_conv, f1, f2 = self.cnn(x)
        f_lstm = self.lstm(x)
        fusion = torch.cat([f_conv,f1,f2,f_lstm], dim=1)
        out = self.fc(fusion)
        return out


# Training function

def train_model(train_loader, val_loader, num_epochs=400, lr=1e-3):
    net = FNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(num_epochs):
        net.train()
        batch_losses = []
        batch_accs = []

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = net(Xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_accs.append((torch.argmax(yhat,1)==yb).float().mean().item())

        train_loss.append(np.mean(batch_losses))
        train_acc.append(100*np.mean(batch_accs))

        # Validation
        net.eval()
        val_batch_losses = []
        val_batch_accs = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                yhat_v = net(Xv)
                loss_v = criterion(yhat_v, yv)
                val_batch_losses.append(loss_v.item())
                val_batch_accs.append((torch.argmax(yhat_v,1)==yv).float().mean().item())

        val_loss.append(np.mean(val_batch_losses))
        val_acc.append(100*np.mean(val_batch_accs))

        if (epoch+1) % 50 == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss[-1]:.4f} | Train Acc: {train_acc[-1]:.2f}% | Val Loss: {val_loss[-1]:.4f} | Val Acc: {val_acc[-1]:.2f}%")

    return net, train_loss, train_acc, val_loss, val_acc


# Test evaluation

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            yhat = model(X_test)
            loss = criterion(yhat, y_test)
            acc = (torch.argmax(yhat,1)==y_test).float().mean().item()
    print(f"Test Loss: {loss.item():.4f} | Test Accuracy: {acc*100:.2f}%")
    return loss.item(), acc*100


# Training / Validation curves

def plot_training_curves(train_loss, train_acc, val_loss, val_acc):
    epochs = range(1, len(train_loss)+1)
    
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# EEG segment visualization

def visualize_eeg_segment(X, trial_idx=0, channel_idx=None):
    """
    X: [n_samples, n_channels, n_times]
    trial_idx: which trial to plot
    channel_idx: None = all channels, int = specific channel
    """
    X_trial = X[trial_idx]  # [channels, times]

    if channel_idx is None:
        plt.figure(figsize=(12,6))
        for ch in range(X_trial.shape[0]):
            plt.plot(X_trial[ch] + ch*50, label=f'Ch {ch}')  # offset for visibility
        plt.xlabel('Timepoints')
        plt.ylabel('Amplitude (uV, offset for clarity)')
        plt.title(f'EEG Trial {trial_idx} - All channels')
        plt.show()
    else:
        plt.figure(figsize=(10,4))
        plt.plot(X_trial[channel_idx])
        plt.xlabel('Timepoints')
        plt.ylabel('Amplitude (uV)')
        plt.title(f'EEG Trial {trial_idx} - Channel {channel_idx}')
        plt.show()