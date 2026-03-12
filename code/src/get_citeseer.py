import os
import scipy.io as sio
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def download_and_convert_citeseer():
    print("⏳ Downloading CiteSeer via PyTorch Geometric...")
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
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
    
    output_path = '../data/citeseer.mat'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sio.savemat(output_path, mat_data)
    print(f"✅ Success! Saved to {output_path}")

if __name__ == "__main__":
    download_and_convert_citeseer()
