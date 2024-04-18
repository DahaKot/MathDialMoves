# Libraries needed for data preparation
import numpy as np
import pandas as pd

import random

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using CUDA
    random.seed(seed_value)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)  # You can choose any seed value

# Download the dataset and put it in subfolder called data
datapath = "train_only_dialogue_window_1.csv"
df = pd.read_csv(datapath)
df = df[["text", "label"]]

X = df['text']
y=np.unique(df['label'], return_inverse=True)[1]

import transformers

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

import torch

X_list=X.to_list()
X_pt = tokenizer(X_list, padding='max_length', max_length = 512, truncation=True, return_tensors='pt')["input_ids"]

y_list=y.tolist()
y_pt = torch.Tensor(y_list).long()

datapath_test = "test_only_dialogue_window_1.csv"
df_test = pd.read_csv(datapath_test)
df_test = df_test[["text", "label"]]

X_test = df_test['text']
y_test=np.unique(df_test['label'], return_inverse=True)[1]

X_list_test=X_test.to_list()
X_pt_test = tokenizer(X_list_test, padding='max_length', max_length = 512, truncation=True, return_tensors='pt')["input_ids"]

y_list_test=y_test.tolist()
y_pt_test = torch.Tensor(y_list_test).long()

# Convert data to torch dataset

X_pt_train = X_pt
y_pt_train = y_pt
from torch.utils.data import Dataset, DataLoader
class BBCNewsDataset(Dataset):
    """Custom-built BBC News dataset"""

    def __init__(self, X, y):
        """
        Args:
            X, y as Torch tensors
        """
        self.X_train = X
        self.y_train = y
        

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]# Get train and test data in form of Dataset class
train_data_pt = BBCNewsDataset(X=X_pt_train, y=y_pt_train)
test_data_pt = BBCNewsDataset(X=X_pt_test, y=y_pt_test)

# Get train and test data in form of Dataloader class
train_loader_pt = DataLoader(train_data_pt, batch_size=50, shuffle=True)
test_loader_pt = DataLoader(test_data_pt, batch_size=50, shuffle=True)

config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
dbert_pt = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

from torch import nn
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class DistilBertClassification(nn.Module):
    def __init__(self):
        super(DistilBertClassification, self).__init__()
        self.dbert = dbert_pt
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(768,64)
        self.ReLu = nn.ReLU()
        self.linear2 = nn.Linear(64,5)

    def forward(self, x):
        x = self.dbert(input_ids=x)
        x = x["last_hidden_state"][:,0,:]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.ReLu(x)
        logits = self.linear2(x)
        # No need for a softmax, because it is already included in the CrossEntropyLoss
        return logits

model_pt = DistilBertClassification().to(device)

for param in model_pt.dbert.parameters():
    param.requires_grad = False

epochs = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_pt.parameters())

