<<<<<<< HEAD
import os
import scipy.io as sio
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def download_and_convert_pubmed():
    print("⏳ Downloading PubMed via PyTorch Geometric...")
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    data = dataset[0]
    
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Features: {data.num_features}")

    adj = to_scipy_sparse_matrix(data.edge_index)
    X = data.x.numpy()
    labels = data.y.numpy()
    
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    mat_data = {
        'network': adj,
        'Attributes': X,
        'group': labels
    }
    
    output_path = '../data/pubmed.mat'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sio.savemat(output_path, mat_data)
    print(f"✅ Success! Saved to {output_path}")

if __name__ == "__main__":
    download_and_convert_pubmed()
=======
import os
import numpy as np
import scipy.io as sio
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def download_and_convert_pubmed():
    print("⏳ Downloading PubMed via PyTorch Geometric...")
    # Downloads to a temp folder (no permanent storage in repo needed)
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    data = dataset[0]

    print(f"   Nodes:    {data.num_nodes}")
    print(f"   Edges:    {data.num_edges}")
    print(f"   Features: {data.num_features}")
    print(f"   Classes:  {dataset.num_classes}")

    # 1. Convert Edge Index to Sparse Adjacency Matrix (undirected, unweighted)
    adj = to_scipy_sparse_matrix(data.edge_index)

    # 2. Get Features (dense)
    X = data.x.numpy()

    # 3. Get Labels
    # PyG labels are 1D tensors (0..2 for PubMed)
    labels = data.y.numpy()
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # 4. Save as .mat file using keys expected by loaders.py:
    #    'network'    → adjacency
    #    'Attributes' → node features
    #    'group'      → labels
    mat_data = {
        'network':    adj,
        'Attributes': X,
        'group':      labels,
    }

    output_path = '../data/pubmed.mat'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sio.savemat(output_path, mat_data)
    print(f"✅ Success! Saved to {output_path}")

if __name__ == "__main__":
    download_and_convert_pubmed()
>>>>>>> 18ea314 (perf: fine-tuning w/ 150 trials)
