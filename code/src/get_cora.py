import os
import numpy as np
import scipy.io as sio
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def download_and_convert_cora():
    print("⏳ Downloading Cora via PyTorch Geometric...")
    # This downloads to a temp folder
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Features: {data.num_features}")

    # 1. Convert Edge Index to Sparse Matrix (Adjacency)
    # This ensures it's an undirected, unweighted adjacency matrix
    adj = to_scipy_sparse_matrix(data.edge_index)
    
    # 2. Get Features (Dense)
    X = data.x.numpy()
    
    # 3. Get Labels
    # PyG labels are 1D tensors (0..6 for Cora)
    labels = data.y.numpy()
    # Ensure shape is (N, 1) for consistency with .mat format
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # 4. Save as .mat file
    # We use keys that match your loaders.py expectations:
    # 'network' -> Adjacency
    # 'Attributes' -> Features (X)
    # 'group' -> Labels
    mat_data = {
        'network': adj,
        'Attributes': X,
        'group': labels
    }
    
    output_path = '../data/cora.mat'
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sio.savemat(output_path, mat_data)
    print(f"✅ Success! Saved to {output_path}")

if __name__ == "__main__":
    download_and_convert_cora()