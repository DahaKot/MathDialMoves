import random

import numpy as np
import pandas as pd
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset

from args_parser import get_args


class CustomDataset(Dataset):
    """Custom-built dataset"""

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
        return self.X_train[idx], self.y_train[idx]


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using CUDA
    random.seed(seed_value)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_tensors(data_path, tokenizer):
    data = pd.read_csv(data_path)
    data = data[["text", "label"]]
    X = data["text"]
    y = np.unique(data["label"], return_inverse=True)[1]

    X_list = X.to_list()
    X_pt = tokenizer(
        X_list, padding='max_length', max_length=512, truncation=True,
        return_tensors='pt'
    )["input_ids"]

    y_list = y.tolist()
    y_pt = torch.Tensor(y_list).long()

    return X_pt, y_pt

def load_dataset(train_path, test_path, tokenizer):
    X_train, y_train = load_tensors(train_path, tokenizer)
    X_test, y_test = load_tensors(test_path, tokenizer)

    train = CustomDataset(X=X_train, y=y_train)
    test = CustomDataset(X=X_test, y=y_test)

    return train, test


# # Download the dataset and put it in subfolder called data
# datapath = "train_only_dialogue_window_1.csv"
# df = pd.read_csv(datapath)
# df = df[["text", "label"]]

# X = df['text']
# y=np.unique(df['label'], return_inverse=True)[1]

# tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# X_list=X.to_list()
# X_pt = tokenizer(X_list, padding='max_length', max_length = 512, truncation=True, return_tensors='pt')["input_ids"]

# y_list=y.tolist()
# y_pt = torch.Tensor(y_list).long()

# datapath_test = "test_only_dialogue_window_1.csv"
# df_test = pd.read_csv(datapath_test)
# df_test = df_test[["text", "label"]]

# X_test = df_test['text']
# y_test=np.unique(df_test['label'], return_inverse=True)[1]

# X_list_test=X_test.to_list()
# X_pt_test = tokenizer(X_list_test, padding='max_length', max_length = 512, truncation=True, return_tensors='pt')["input_ids"]

# y_list_test=y_test.tolist()
# y_pt_test = torch.Tensor(y_list_test).long()

# # Convert data to torch dataset

# X_pt_train = X_pt
# y_pt_train = y_pt


# train_data_pt = BBCNewsDataset(X=X_pt_train, y=y_pt_train)
# test_data_pt = BBCNewsDataset(X=X_pt_test, y=y_pt_test)

# Get train and test data in form of Dataloader class


if __name__ == "__main__":
    set_seed(1) 

    args = get_args()

    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train, test = load_dataset(args.train_path, args.test_path, tokenizer)

    train_loader_pt = DataLoader(train, batch_size=32, shuffle=True)
    test_loader_pt = DataLoader(test, batch_size=32, shuffle=True)

    config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    dbert_pt = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)


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

    from tqdm import tqdm
    # Define the dictionary "history" that will collect key performance indicators during training
    history = {}
    history["epoch"]=[]
    history["train_loss"]=[]
    history["valid_loss"]=[]
    history["train_accuracy"]=[]
    history["valid_accuracy"]=[]

    from datetime import datetime
    # Measure time for training
    start_time = datetime.now()

    # Loop on epochs
    for e in range(epochs):
        
        # Set mode in train mode
        model_pt.train()
        
        train_loss = 0.0
        train_accuracy = []
        
        # Loop on batches
        for X, y in tqdm(train_loader_pt):
            # Get prediction & loss
            prediction = model_pt(X.to(device))
            loss = criterion(prediction, y.to(device))
            
            # Adjust the parameters of the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            prediction_index = prediction.argmax(axis=1)
            accuracy = (prediction_index==y.to(device))
            train_accuracy += accuracy
        
        train_accuracy = (sum(train_accuracy) / len(train_accuracy)).item()
        
        # Calculate the loss on the test data after each epoch
        # Set mode to evaluation (by opposition to training)
        model_pt.eval()
        valid_loss = 0.0
        valid_accuracy = []
        for X, y in tqdm(test_loader_pt):
            
            prediction = model_pt(X.to(device))
            loss = criterion(prediction, y.to(device))

            valid_loss += loss.item()
            
            prediction_index = prediction.argmax(axis=1)
            accuracy = (prediction_index==y.to(device))
            valid_accuracy += accuracy
        valid_accuracy = (sum(valid_accuracy) / len(valid_accuracy)).item()
        
        # Populate history
        history["epoch"].append(e+1)
        history["train_loss"].append(train_loss / len(train_loader_pt))
        history["valid_loss"].append(valid_loss / len(test_loader_pt))
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(valid_accuracy)    
            
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader_pt) :10.3f} \t\t Validation Loss: {valid_loss / len(test_loader_pt) :10.3f}')
        print(f'\t\t Training Accuracy: {train_accuracy :10.3%} \t\t Validation Accuracy: {valid_accuracy :10.3%}')
        
    # Measure time for training
    end_time = datetime.now()
    training_time_pt = (end_time - start_time).total_seconds()

