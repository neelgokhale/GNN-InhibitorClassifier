# ../src/dataset.py

import os
import torch

import pandas as pd
import numpy as np

from typing import Callable, Any
from torch_geometric.data import Dataset, Data, download_url
from rdkit.Chem import MolFromSmiles, rdmolops
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from constants import Constant as c


class MoleculeDataset(Dataset):
    def __init__(self, root: str, 
                 transform: Callable=None, 
                 pre_transform: Callable=None):
        """Initialize dataset

        Args:
            `root` (`str | None`): root folder for data storage
            `transform` (`Callable | None`, optional): transforation protocol. Defaults to None.
            `pre_transform` (`Callable | None`, optional): pre-transformation protocol. Defaults to None.
        """
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> str:
        """Returns the name of the raw file. If raw file does not exist, download function is triggered

        Returns:
            `str`: raw dataset file name
        """
        return "HIV.csv"
    
    @property
    def processed_file_names(self) -> str:
        """If these files are found in raw_dir, processing is skipped
        """
        return 'not_implemented.pt'
        
    def download(self) -> None:
        """Download the HIV dataset from MoleculeNet
        """
        url = c.DATASET_URL
        path = download_url(url, self.raw_dir)
        
    def process(self) -> None:
        """Process raw data
        """
        try:
            self.data = pd.read_csv(self.raw_paths[0])
        except Exception as e:
            print(f"Could not convert csv due to the following errror:\n{e}")
            
        for ind, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = MolFromSmiles(mol['smiles'])
            # get node features
            node_features = self._get_node_features(mol_obj)
            # get edge features
            edge_features = self._get_edge_features(mol_obj)
            # get adjacency matrix
            edge_index = self._get_adjacency_info(mol_obj)
            # get labels
            label = self._get_labels(mol['HIV_active'])
            
            # create data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                y=label,
                smiles=mol['smiles']
            )
            
            # save to processed dir
            pathname = os.path.join(self.processed_dir, f"data_{ind}.pt")
            torch.save(data, pathname)    
            
    def _get_node_features(self, mol: Mol) -> torch.Tensor:
        """Returns a 2d tensor with [# nodes, # of features per node] using molecule smile

        Args:
            `mol` (`rdkit.Chem.rdchem.Mol`): Mol object
            
        Returns:
            `torch.Tensor`: tensor object with nodes and features
        """
        all_node_features = []
        
        for atom in mol.GetAtoms():
            node_features = []
            # get atomic number
            node_features.append(atom.GetAtomicNum())
            # get degree
            node_features.append(atom.GetDegree())
            # get formal charge
            node_features.append(atom.GetFormalCharge())
            # get hybridization
            node_features.append(atom.GetHybridization())
            # get aromaticity
            node_features.append(atom.GetIsAromatic())
            
            all_node_features.append(node_features)
            
        all_node_features = np.asarray(all_node_features)
        
        return torch.tensor(all_node_features, dtype=torch.float)
    
    def _get_edge_features(self, mol: Mol) -> torch.Tensor:
        """Returns a 2d matrix with [# edges, # of features per edge] using the bonds

        Args:
            `mol` (`Mol`): Mol object

        Returns:
            `torch.Tensor`: tensor object with edges and features
        """
        all_edge_features = []
        for bond in mol.GetBonds():
            edge_features = []
            # get bond type as double
            edge_features.append(bond.GetBondTypeAsDouble())
            # get if bond is in a ring
            edge_features.append(bond.IsInRing())
            
            all_edge_features.append(edge_features)
            
        all_edge_features = np.asarray(all_edge_features)
        
        return torch.tensor(all_edge_features, dtype=torch.float)
    
    def _get_adjacency_info(self, mol: Mol) -> torch.Tensor:
        """Generate adjacency matrix for each molecule

        Args:
            `mol` (`Mol`): Mol object

        Returns:
            `torch.Tensor`: adjacency matrix
        """
        adj_mat = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_mat)
        coo = np.array(list(zip(row, col)))
        coo = np.reshape(coo, (2, -1))
        
        return torch.tensor(coo, dtype=torch.long)
    
    def _get_labels(self, label: Any) -> torch.Tensor:
        """Get labels from molecules

        Args:   
            `label` (`Any`): labels from features   

        Returns:
            `torch.Tensor`: tensor of labels
        """
        label = np.asarray([label])
        
        return torch.tensor(label, dtype=torch.int64)
    
    def len(self):
        return self.data.shape[0]
    
    def get(self, idx: int) -> Any:
        """Get data object based on index from stored directory

        Args:
            `idx` (`int`): index of data

        Returns:
            `Any`: data object
        """
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        
        return data
