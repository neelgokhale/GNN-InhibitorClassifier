# ../src/model.py

import os
import torch

import torch.nn.functional as f
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling, global_mean_pool, global_max_pool

torch.manual_seed(1024)


class GNN(torch.nn.Module):
    def __init__(self, feature_size) -> None:
        super(GNN, self).__init__()
        embedding_size = 128
        n_heads = 4
        self.n_layers = 9
        dropout_rate = 0.9
        top_k_ratio = 0.85
        self.top_k_every_n = 3
        edge_dim = 11
        
        # setup module lists
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        
        # GNN layers setup: 
        # transformer conv -> linear (head transformer) -> batch norm 1d
        
        self.conv1 = TransformerConv(feature_size, 
                                     embedding_size, 
                                     heads=n_heads, 
                                     dropout=dropout_rate, 
                                     edge_dim=edge_dim)
        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)
        
        # add sequential layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(
                embedding_size, embedding_size, heads=n_heads, dropout=dropout_rate, edge_dim=edge_dim
            ))
            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        # linear layers
        self.linear1 = Linear(embedding_size * 2, 256) # max and mean pooled into 1 layer
        self.linear2 = Linear(256, 64) 
        self.linear3 = Linear(64, 1)
        
    def forward(self, x, edge_attr, edge_index, batch_index):
        # first block
        x = self.conv1(x, edge_index, edge_attr)
        x = self.transf1(x)
        x = self.bn1(x)
        
        # holds the intermediate graph representations
        global_reps = []
        
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.transf_layers[i](x)
            x = self.bn_layers[i](x) 
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )          
                # add current representation
                global_reps.append(torch.cat([
                    global_max_pool(x, batch_index), global_mean_pool(x, batch_index)
                ], dim=1))
        
        # sum all reps
        x = sum(global_reps)
        
        # output block
        x = self.linear1(x).tanh() # normalize output between -1, 1
        x = f.dropout(x, p=0.8, training=self.training)
        x = self.linear2(x).tanh()
        x = f.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)
        
        return x
