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


def count_parameters(model: torch.nn.Module):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# pytorch device setup
device_type = "mps" if (torch.backends.mps.is_available() and c.SYSTEM == "apple") \
    else "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

# dataset setup
train = MoleculeDataset(root=c.RAW_PATH, filename=c.TRAIN_OSP_PATH)
test = MoleculeDataset(root=c.RAW_PATH, filename=c.TEST_PATH)

# model and weights
model = GNN(feature_size=train[0].x.shape[1])
model = model.to(device)

weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch

BATCH_SIZE = 256
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=True
)

