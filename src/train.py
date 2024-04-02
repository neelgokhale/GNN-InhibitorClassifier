# ../src/train.py

import os
import torch

import pandas as pd
import numpy as np
from sklearn.metrics import \
    confusion_matrix, f1_score, accuracy_score, precision_score, \
    roc_auc_score, recall_score
from torch_geometric.data import DataLoader
from tqdm import tqdm
import mlflow.pytorch

from constants import Constant as c
from dataset import MoleculeDataset
from model import GNN


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# pytorch device setup
device_type = "mps" if (torch.backends.mps.is_available() and c.SYSTEM == "apple") \
    else "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

# dataset setup
train_dataset = MoleculeDataset(root=c.RAW_PATH, filename=c.TRAIN_OSP_FILENAME)
test_dataset = MoleculeDataset(root=c.RAW_PATH, filename=c.TEST_FILENAME)

# model and weights
model = GNN(feature_size=train_dataset[0].x.shape[1]) 
model = model.to(device)

weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

BATCH_SIZE = 256
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

def train(epoch: int) -> float:
    # enumerate over the data
    all_preds = []
    all_labels = []
    
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        # reset gradients
        optimizer.zero_grad()
        # passing node features and the connection info
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        # calculate the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        # update using gradients
        optimizer.step()
        
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
        
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    
    calculate_metrics(all_preds, all_labels, epoch, "train")
    
    return loss

def test(epoch: int) -> float:
    # enumerate over the data
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        batch.to(device)
        # reset gradients
        optimizer.zero_grad()
        # passing node features and the connection info
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        # calculate the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        # update using gradients
        optimizer.step()
        
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
        
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    
    calculate_metrics(all_preds, all_labels, epoch, "test")
    
    return loss

def calculate_metrics(y_pred: np.ndarray, 
                      y_true: np.ndarray, 
                      epoch: int, 
                      type: str):
    try:
        roc = roc_auc_score(y_pred, y_true)
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
    
def run_train_and_test(num_epochs: int) -> None:
    for epoch in range(num_epochs):
        # training (log every epoch)
        model.train()
        loss = train(epoch=epoch).cpu().detach().numpy()
        mlflow.log_metric(key="LOSS-TRAIN", value=float(loss), step=epoch)
        
        # testing (log every 5th epoch)
        model.eval()
        if epoch % 5 == 0:
            loss = test(epoch=epoch).cpu().detach().numpy()
            mlflow.log_metric(key="LOSS-TEST", value=float(loss), step=epoch)
            
        scheduler.step()
    
    mlflow.pytorch.log_model(model, c.MODEL_NAME)
    print("Training and testing complete.")
