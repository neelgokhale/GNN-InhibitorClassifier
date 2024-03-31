# ../src/dataset.py

import os
import torch

import pandas as pd
import numpy as np

from typing import Callable, Any
from torch_geometric.data import Dataset, Data, download_url
from deepchem.feat import MolGraphConvFeaturizer
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
            `root` (`str`): root folder for data storage
            `transform` (`Callable`, optional): transforation protocol. Defaults to None.
            `pre_transform` (`Callable`, optional): pre-transformation protocol. Defaults to None.
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
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        
        return [f"data_{i}.pt" for i in list(self.data.index)]
        
    def download(self) -> None:
        """Download the HIV dataset from MoleculeNet
        """
        url = c.DATASET_URL
        path = download_url(url, self.raw_dir)
        
    def process(self) -> None:
        """Process raw data
        """
        try:
            self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        except Exception as e:
            raise e("Could not convert csv")
        
        # using deepchem's molgraph featurizer
        featurizer = MolGraphConvFeaturizer(use_edges=True)
            
        for ind, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            
            # TODO work around for featurizer required
            # I had to add the following modification to the GraphData obj @ line 150 from the
            # deepchem library as below:
            # class GraphData:
            #       ...
            #   def to_pyg_graph(self):
            #       ...
            #       for key, value in self.kwargs.items():
            #           << if key != 'pos':  # Exclude 'pos' from kwargs >>
            
            feat = featurizer.featurize(row['smiles'])
            data = feat[0].to_pyg_graph()
            data.y = self._get_labels(row['HIV_active'])
            data.smiles = row['smiles']
        
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
