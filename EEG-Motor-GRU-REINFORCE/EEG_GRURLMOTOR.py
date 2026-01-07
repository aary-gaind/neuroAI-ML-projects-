
# EEG GRU + REINFORCE Agent Full Workflow w/ BCI Motor Imagery data (Comp IV 2a data)


import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, mne, scipy.io, os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


#  Data preprocessing

from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/BCICIV_2a.zip /content
%%capture
!unzip /content/BCICIV_2a.zip -d data

def preprocess_gdf(filepath,event_ids=[5,6,7,8],window_size_sec=3,step_size_sec=0.25,max_windows=6):
    raw=mne.io.read_raw_gdf(filepath,eog=['EOG-left','EOG-central','EOG-right'],preload=True)
    raw.drop_channels(['EOG-left','EOG-central','EOG-right'])
    events,_=mne.events_from_annotations(raw)
    epochs=mne.Epochs(raw,events,event_id=event_ids,tmin=-0.25,tmax=4.25,preload=True)
    X=epochs.get_data()
    y=epochs.events[:,2]
    label_mapping={label:i for i,label in enumerate(np.unique(y))}
    y_mapped=np.array([label_mapping[label] for label in y])
    sfreq=epochs.info['sfreq']
    window_size=int(window_size_sec*sfreq)
    step_size=int(step_size_sec*sfreq)
    X_slid,y_slid=[],[]
    for trial,label in zip(X,y_mapped):
        n_times=trial.shape[1]
        start,end,win_cnt=0,window_size,0
        while end<=n_times and win_cnt<max_windows:
            X_slid.append(trial[:,start:end])
            y_slid.append(label)
            start+=step_size
            end+=step_size
            win_cnt+=1
    return np.stack(X_slid), np.array(y_slid)

# Load two subjects as example
X1,y1=preprocess_gdf('/content/data/A04T.gdf')

# Second subject with MAT labels
raw=mne.io.read_raw_gdf('/content/data/A04E.gdf',eog=['EOG-left','EOG-central','EOG-right'],preload=True)
raw.drop_channels(['EOG-left','EOG-central','EOG-right'])
events,_=mne.events_from_annotations(raw)
epochs=mne.Epochs(raw,events,event_id={'783':7},tmin=-0.25,tmax=4.25,preload=True)
X=epochs.get_data()
mat=scipy.io.loadmat('A04E.mat')
y=mat['classlabel'].squeeze()
label_mapping={label:i for i,label in enumerate(np.unique(y))}
y_mapped=np.array([label_mapping[label] for label in y])

window_size_sec,step_size_sec,max_windows=3,0.25,6
sfreq=epochs.info['sfreq']
window_size=int(window_size_sec*sfreq)
step_size=int(step_size_sec*sfreq)
X_slid,y_slid=[],[]
for trial,label in zip(X,y_mapped):
    n_times=trial.shape[1]
    start,end,win_cnt=0,window_size,0
    while end<=n_times and win_cnt<max_windows:
        X_slid.append(trial[:,start:end])
        y_slid.append(label)
        start+=step_size
        end+=step_size
        win_cnt+=1

X2=np.stack(X_slid)
y2=np.array(y_slid)
X_combined=np.concatenate([X1,X2],axis=0)
y_combined=np.concatenate([y1,y2],axis=0)
print("Combined X shape:",X_combined.shape,"Combined y shape:",y_combined.shape)


# Train/Val/Test split

X_train,X_temp,y_train,y_temp=train_test_split(X_combined,y_combined,test_size=0.5,random_state=42,stratify=y_combined)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.3,random_state=42,stratify=y_temp)

X_train_tensor=torch.tensor(X_train,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train,dtype=torch.long)
X_val_tensor=torch.tensor(X_val,dtype=torch.float32)
y_val_tensor=torch.tensor(y_val,dtype=torch.long)
X_test_tensor=torch.tensor(X_test,dtype=torch.float32)
y_test_tensor=torch.tensor(y_test,dtype=torch.long)

train_dataset=TensorDataset(X_train_tensor,y_train_tensor)
val_dataset=TensorDataset(X_val_tensor,y_val_tensor)
test_dataset=TensorDataset(X_test_tensor,y_test_tensor)

train_loader=DataLoader(train_dataset,batch_size=36,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=val_dataset.tensors[0].shape[0])
test_loader=DataLoader(test_dataset,batch_size=test_dataset.tensors[0].shape[0])


# Agent definition

class MultiScaleGlimpseNet(nn.Module):
    def __init__(self,patch_sizes=[7,15,31],num_channels=2,embed_dim=128):
        super().__init__()
        self.patch_sizes=patch_sizes
        self.num_channels=num_channels
        input_size=sum([num_channels*p for p in patch_sizes])
        self.glimpse_fc=nn.Sequential(nn.Linear(input_size,embed_dim),nn.ReLU(),nn.Linear(embed_dim,embed_dim))
        self.location_fc=nn.Sequential(nn.Linear(2,embed_dim),nn.ReLU(),nn.Linear(embed_dim,embed_dim))
    def forward(self,eeg_signal,location):
        C,T=eeg_signal.shape
        c_norm,t_norm=location[0]
        c=int(round(c_norm.item()*(C-1)))
        t=int(round(t_norm.item()*(T-1)))
        patches=[]
        for p in self.patch_sizes:
            half=p//2
            padded=F.pad(eeg_signal,(half,half),mode='constant',value=0)
            patch=padded[c:c+self.num_channels,t-half:t+half]
            patches.append(patch.flatten())
        glimpse_vector=torch.cat(patches,dim=0)
        h_MS=self.glimpse_fc(glimpse_vector)
        h_l=self.location_fc(location)
        return F.relu(h_MS+h_l)

class GRUCore(nn.Module):
    def __init__(self,input_dim=128,hidden_dim=256):
        super().__init__()
        self.gru=nn.GRUCell(input_dim,hidden_dim)
    def forward(self,glimpse,h_prev):
        return self.gru(glimpse,h_prev)

class LocationNetwork(nn.Module):
    def __init__(self,hidden_dim=256):
        super().__init__()
        self.fc=nn.Linear(hidden_dim,2)
        self.log_std=nn.Parameter(torch.zeros(2))
    def forward(self,h):
        mean=torch.sigmoid(self.fc(h))
        std=self.log_std.exp()
        return mean,std

class EEGAgent(nn.Module):
    def __init__(self,patch_sizes=[7,15,31],num_channels=2,glimpse_embed_dim=128,hidden_dim=256,num_classes=4):
        super().__init__()
        self.glimpse_net=MultiScaleGlimpseNet(patch_sizes,num_channels,glimpse_embed_dim)
        self.core=GRUCore(glimpse_embed_dim,hidden_dim)
        self.location_net=LocationNetwork(hidden_dim)
        self.classifier=nn.Linear(hidden_dim,num_classes)
        self.value_net=nn.Linear(hidden_dim,1)
    def forward(self,eeg_signal,init_loc,num_glimpses):
        h_t=torch.zeros(1,self.core.gru.hidden_size).to(eeg_signal.device)
        location=init_loc.clone().detach()
        loc_means,loc_log_probs,baselines=[],[],[]
        for _ in range(num_glimpses):
            glimpse=self.glimpse_net(eeg_signal,location).unsqueeze(0)
            h_t=self.core(glimpse,h_t)
            mean,std=self.location_net(h_t.detach())
            dist=torch.distributions.Normal(mean,std)
            location=dist.rsample()
            location=torch.clamp(location,0,1)
            log_prob=dist.log_prob(location).sum(dim=-1)
            value=self.value_net(h_t.detach()).squeeze()
            loc_means.append(mean)
            loc_log_probs.append(log_prob)
            baselines.append(value)
        logits=self.classifier(h_t)
        return logits,loc_means,loc_log_probs,baselines


#  Training

agent=EEGAgent(num_classes=4).to(device)
optimizer_cls=optim.Adam(list(agent.classifier.parameters())+list(agent.core.parameters())+list(agent.glimpse_net.parameters()),lr=1e-3)
optimizer_loc=optim.Adam(agent.location_net.parameters(),lr=1e-4)
optimizer_val=optim.Adam(agent.value_net.parameters(),lr=1e-3)

num_epochs=5
num_glimpses=6

for epoch in range(num_epochs):
    for eeg_signal,label in train_loader:
        eeg_signal=eeg_signal.squeeze(0).to(device)
        label=label.squeeze(0).to(device)
        init_loc=torch.rand(1,2).to(device)
        logits,loc_means,loc_log_probs,baselines=agent(eeg_signal,init_loc,num_glimpses)
        pred=logits.argmax(dim=-1)
        correct=(pred==label).float()
        R=correct.detach().expand(len(loc_log_probs))
        baselines=torch.stack(baselines)
        advantages=R-baselines.detach()
        ce_loss=F.cross_entropy(logits,label.unsqueeze(0))
        reinforce_loss=-(torch.stack(loc_log_probs)*advantages).mean()
        value_loss=F.mse_loss(baselines,R)
        optimizer_cls.zero_grad(); ce_loss.backward(); optimizer_cls.step()
        optimizer_loc.zero_grad(); reinforce_loss.backward(); optimizer_loc.step()
        optimizer_val.zero_grad(); value_loss.backward(); optimizer_val.step()
    print(f"Epoch {epoch+1}: CE={ce_loss.item():.4f}, REINFORCE={reinforce_loss.item():.4f}, Value={value_loss.item():.4f}")


#  Evaluation

def evaluate_agent(agent,loader):
    agent.eval(); correct=0; total=0
    with torch.no_grad():
        for eeg_signal,label in loader:
            eeg_signal=eeg_signal.squeeze(0).to(device)
            label=label.squeeze(0).to(device)
            init_loc=torch.rand(1,2).to(device)
            logits,_,_,_ = agent(eeg_signal,init_loc,num_glimpses)
            pred=logits.argmax(dim=-1)
            correct+=(pred==label).sum().item()
            total+=1
    return correct/total

train_acc=evaluate_agent(agent,train_loader)
val_acc=evaluate_agent(agent,val_loader)
test_acc=evaluate_agent(agent,test_loader)
print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")


# Single trial inference

def infer_single_trial(agent,eeg_signal):
    agent.eval()
    eeg_signal=eeg_signal.to(device)
    init_loc=torch.rand(1,2).to(device)
    with torch.no_grad():
        logits,loc_means,_,_ = agent(eeg_signal,init_loc,num_glimpses)
        pred_class=logits.argmax(dim=-1).item()
    locs=torch.stack(loc_means).cpu().numpy()
    plt.figure(figsize=(6,3))
    plt.scatter(locs[:,1],locs[:,0],c='r')
    plt.xlabel('Time'); plt.ylabel('Channel')
    plt.title(f'Glimpses for Predicted Class {pred_class}'); plt.show()
    return pred_class,locs

sample_eeg=X_test_tensor[0]
pred_class,locs=infer_single_trial(agent,sample_eeg)
print("Predicted class:",pred_class)


#  All test glimpse visualization

def visualize_all_test_glimpses(agent,loader):
    agent.eval(); all_locs=[]; all_labels=[]
    with torch.no_grad():
        for eeg_signal,labels in loader:
            eeg_signal=eeg_signal.squeeze(0).to(device)
            labels=labels.squeeze(0).cpu().numpy()
            init_loc=torch.rand(1,2).to(device)
            _,loc_means,_,_ = agent(eeg_signal,init_loc,num_glimpses)
            locs=torch.stack(loc_means).cpu().numpy()
            for loc in locs: all_locs.append(loc)
            all_labels.extend([labels]*len(locs))
    all_locs=np.array(all_locs)
    all_labels=np.array(all_labels)
    plt.figure(figsize=(10,4))
    for cls in np.unique(all_labels):
        mask=all_labels==cls
        plt.scatter(all_locs[mask,1],all_locs[mask,0],label=f'Class {cls}',alpha=0.6)
    plt.xlabel('Time'); plt.ylabel('Channel'); plt.title('Glimpse Locations Across Test Set'); plt.legend(); plt.show()

visualize_all_test_glimpses(agent,test_loader)


# Accuracy vs number of glimpses

def accuracy_vs_glimpses(agent,loader,max_glimpses=6):
    agent.eval(); acc_list=[]
    with torch.no_grad():
        for num_g in range(1,max_glimpses+1):
            correct=0; total=0
            for eeg_signal,label in loader:
                eeg_signal=eeg_signal.squeeze(0).to(device)
                label=label.squeeze(0).to(device)
                init_loc=torch.rand(1,2).to(device)
                logits,_,_,_ = agent(eeg_signal,init_loc,num_g)
                pred=logits.argmax(dim=-1)
                correct+=(pred==label).sum().item(); total+=1
            acc_list.append(correct/total)
    plt.figure(figsize=(6,4))
    plt.plot(range(1,max_glimpses+1),acc_list,marker='o')
    plt.xlabel('Number of Glimpses'); plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Glimpses'); plt.grid(True); plt.show()
    return acc_list

acc_vs_glimpses=accuracy_vs_glimpses(agent,test_loader,max_glimpses=num_glimpses)
print("Accuracy per glimpse:",acc_vs_glimpses)

# 
# Save model

torch.save(agent.state_dict(),'eeg_agent.pth')
print("Model saved as eeg_agent.pth")