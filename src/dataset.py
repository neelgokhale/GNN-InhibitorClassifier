# ../src/dataset.py

import os
import torch

import pandas as pd
import numpy as np

from typing import Optional, Callable, Any
from torch_geometric.data import Dataset, download_url
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Mol

from constants import Constant as c


class MoleculeDataset(Dataset):
    def __init__(self, root: str | None, 
                 transform: Callable | None=None, 
                 pre_transform: Callable | None=None):
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
        pass
    
    def _get_node_features(self, mol: Mol) -> torch.Tensor:
        """Returns a 2d tensor with [# nodes, # of features per node] using molecule smile

        Args:
            `mol` (`rdkit.Chem.rdchem.Mol`): Mol object
            
        Returns:
            `torch.Tensor`: tensor object with nodes and features
        """
        
        all_node_features = []
        
        for atom in mol.GetAtoms(mol):
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
    
    
            