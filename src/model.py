# ../src/model.py

import os
import torch

import torch.nn.functional as f
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import \
    GATConv, TopKPooling, BatchNorm, global_mean_pool, global_max_pool

torch.manual_seed(1024)


class GNN(torch.nn.Module):
    def __init__(self, feature_size) -> None:
        super(GNN, self).__init__()
        num_classes = 2
        embedding_size = 1024
        
        # GNN layers setup: 
        # GATConv -> head_transform -> pooling
        
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform1 = Linear(embedding_size * 3, embedding_size) # convert back into initial size
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform2 = Linear(embedding_size * 3, embedding_size) # convert back into initial size
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform3 = Linear(embedding_size * 3, embedding_size) # convert back into initial size
        self.pool3 = TopKPooling(embedding_size, ratio=0.2)
        
        # linear layers
        self.linear1 = Linear(embedding_size * 2, 1024) # max and mean pooled into 1 layer
        self.linear2 = Linear(1024, num_classes) # into final output classes
        
    def forward(self, x, edge_attr, edge_index, batch_index):
        # first block
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        # TODO: need to support edge_attrs. Now set to None
        x, edge_index, edge_attr, batch_index, _, _ = self.pool1(
            x, edge_index, None, batch_index
        )
        x1 = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)

        # second block
        x = self.conv2(x, edge_index)
        x = self.head_transform2(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool2(
            x, edge_index, None, batch_index
        )
        x2 = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)

        # third block
        x = self.conv3(x, edge_index)
        x = self.head_transform3(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool3(
            x, edge_index, None, batch_index
        )
        x3 = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)

        # concat pooled vectors
        x = x1 + x2 + x3

        # output block
        x = self.linear1(x).relu()
        x = f.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)

        return x
