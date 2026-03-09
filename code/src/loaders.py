import os
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from ogb.nodeproppred import NodePropPredDataset

def load_dataset(name, root_dir='./data'):
    """
    Universal Data Loader (Adapter Pattern).
    
    Args:
        name (str): Name of the dataset ('blogcatalog', 'cora', 'ogbn-arxiv').
        root_dir (str): Path to the data storage folder.
        
    Returns:
        tuple: (adj_matrix, features, labels)
            - adj_matrix: scipy.sparse.csr_matrix (N x N)
            - features: numpy array or None (N x F)
            - labels: numpy array (N x 1)
    """
    name = name.lower()
    
    if name in ['blogcatalog', 'cora', 'citeseer', 'pubmed', 'flickr', 'www10k', 'www200k']:
        return _load_mat_data(name, root_dir)
    elif name.startswith('ogb'):
        return _load_ogb_data(name, root_dir)
    else:
        raise ValueError(f"Dataset '{name}' not supported.")

def _load_mat_data(name, root_dir):
    """
    Loads legacy .mat files (standard in network embedding research).
    Expects keys: 'network' (sparse adj) and 'group'/'label' (labels).
    """
    file_path = os.path.join(root_dir, f"{name}.mat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please upload it to {root_dir}")
        
    print(f"Loading {name} from .mat file...")
    mat = sio.loadmat(file_path)
    
    # 1. Adjacency Matrix
    if 'network' in mat:
        adj = mat['network']
    elif 'A' in mat:
        adj = mat['A']
    else:
        raise KeyError(f"Could not find adjacency matrix in {name}.mat")
    
    # Ensure it is CSR and Symmetric
    adj = adj.tocsr()
    # Check for symmetry roughly (optional, but good for embedding)
    if (adj != adj.T).sum() > 0:
        print("Note: Graph is directed. Symmetrizing for embedding...")
        adj = adj + adj.T
    
    # 2. Labels
    if 'group' in mat:
        labels = mat['group']
    elif 'label' in mat:
        labels = mat['label']
    elif 'gnd' in mat:
        labels = mat['gnd']
    else:
        labels = None
        
    # 3. Features (Usually .mat files for these tasks don't have features, or they are separate)
    # We return None for features for now unless found
    features = None
    if 'Attributes' in mat:
        features = mat['Attributes']
    elif 'X' in mat:
        features = mat['X']

    return adj, features, labels

def _load_ogb_data(name, root_dir):
    """
    Loads OGB datasets and adapts them to sparse matrices.
    """
    print(f"Loading {name} via OGB library...")
    dataset = NodePropPredDataset(name=name, root=root_dir)
    
    graph = dataset[0][0]
    labels = dataset[0][1]
    
    num_nodes = graph['num_nodes']
    edge_index = graph['edge_index']
    
    # 1. Convert OGB Edge List to Sparse CSR Matrix
    # edge_index is [2, E], we treat edges as binary (weight=1)
    data = np.ones(edge_index.shape[1])
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    
    # 2. Symmetrize (OGB Arxiv is directed, but embeddings usually treat it as undirected)
    # Note: A + A.T might double count edges, so we convert to boolean then back to float to fix weights
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data) # Binarize
    
    # 3. Features
    features = graph['node_feat'] # Already numpy array
    
    return adj, features, labels

if __name__ == "__main__":
    # Quick Test
    try:
        # Test OGB (since it downloads automatically)
        A, X, Y = load_dataset('ogbn-arxiv', root_dir='../data')
        print("Success! OGB Data Shapes:")
        print(f"  Adj: {A.shape} (CSR)")
        print(f"  Feat: {X.shape if X is not None else 'None'}")
        print(f"  Labels: {Y.shape}")
    except Exception as e:
        print(f"Error during test: {e}")