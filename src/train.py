# ../src/train.py

import os
import torch

import pandas as pd
import numpy as np
from torch.autograd import Variable
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
test_dataset = MoleculeDataset(root=c.RAW_PATH, filename=c.TEST_FILENAME, test=True)

# model and weights
model = GNN(feature_size=train_dataset[0].x.shape[1]) 
model = model.to(device)

weights = torch.tensor([1], dtype=torch.float32).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)

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
    
    # track steps and loss
    running_loss = 0.0
    step = 0
    
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
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        # update using gradients
        optimizer.step()
        
        # update tracking
        running_loss += loss.item()
        step += 1
        
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy())) # round sigmoid to int
        all_labels.append(batch.y.cpu().detach().numpy())
        
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    
    calculate_metrics(all_preds, all_labels, epoch, "train")
    
    return running_loss / step

def test(epoch: int) -> float:
    # enumerate over the data
    all_preds = []
    all_labels = []
    
    # track steps and loss
    running_loss = 0.0
    step = 0
    
    for batch in test_loader:
        batch.to(device)

        # passing node features and the connection info
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)

        # calculate the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        
        # update tracking
        running_loss += loss.item()
        step += 1
        
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
        
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    
    calculate_metrics(all_preds, all_labels, epoch, "test")
    
    return running_loss / step

def calculate_metrics(y_pred: np.ndarray, 
                      y_true: np.ndarray, 
                      epoch: int, 
                      type: str):
    # precision and recall scores
    prec = precision_score(y_pred, y_true)
    rec = recall_score(y_pred, y_true)
    mlflow.log_metric(key=f"PRECISION-{type.upper()}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"RECALL-{type.upper()}", value=float(rec), step=epoch)

    try:
        roc = roc_auc_score(y_pred, y_true)
        mlflow.log_metric(key=f"ROC-AUC-{type.upper()}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type.upper()}", value=float(0), step=epoch)
    
def run_train_and_test(num_epochs: int) -> None:
    with mlflow.start_run() as run:
        for epoch in range(num_epochs):
            # training (log every epoch)
            model.train() # setup model in training mode
            loss = train(epoch=epoch)
            mlflow.log_metric(key="LOSS-TRAIN", value=float(loss), step=epoch)
            
            # testing (log every 5th epoch)
            model.eval() # setup model in testing mode
            if epoch % 5 == 0:
                loss = test(epoch=epoch)
                mlflow.log_metric(key="LOSS-TEST", value=float(loss), step=epoch)
                
            scheduler.step()
    
    mlflow.pytorch.log_model(model, c.MODEL_NAME)
    print("Training and testing complete.")
