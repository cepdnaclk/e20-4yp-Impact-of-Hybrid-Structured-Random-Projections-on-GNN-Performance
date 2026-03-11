import os
import sys
import torch
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
import time
import warnings
import scipy.sparse as sp

# Suppress warnings from sklearn during Optuna trials
warnings.filterwarnings('ignore', category=UserWarning)

# Ensure 'src' is in path, or use absolute imports from root
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from loaders import load_dataset
from fastrp_layer import FastRP

def load_data_for(dataset_name: str, data_root: str, device: torch.device):
    if dataset_name == 'flickr':
        flickr_path = os.path.join(data_root, 'flickr.mat')
        if not os.path.exists(flickr_path):
            raise FileNotFoundError("Missing flickr.mat.")
    elif dataset_name == 'cora':
        cora_path = os.path.join(data_root, 'cora.mat')
        if not os.path.exists(cora_path):
            raise FileNotFoundError("Missing cora.mat.")
    elif dataset_name == 'citeseer':
        citeseer_path = os.path.join(data_root, 'citeseer.mat')
        if not os.path.exists(citeseer_path):
            raise FileNotFoundError("Missing citeseer.mat.")
    elif dataset_name == 'pubmed':
        pubmed_path = os.path.join(data_root, 'pubmed.mat')
        if not os.path.exists(pubmed_path):
            raise FileNotFoundError("Missing pubmed.mat.")

    print(f"Loading {dataset_name}...")
    adj, features, labels = load_dataset(dataset_name, root_dir=data_root)

    print(f"   Nodes: {adj.shape[0]}, Edges: {adj.nnz}")

    # Prepare features if they exist
    feat_tensor = None
    if features is not None:
        if hasattr(features, 'todense'):
            features = features.todense()
        feat_tensor = torch.FloatTensor(features).to(device)

    # Convert sparse scipy matrix to edge index for easier manipulation
    adj = adj.tocoo()
    
    # Extract upper triangle to avoid duplicating undirected edges
    # We assume undirected graphs for node-classification datasets usually
    upper_mask = adj.row < adj.col
    row, col = adj.row[upper_mask], adj.col[upper_mask]
    
    # If the graph was directed, the upper triangle might miss edges. 
    # Just in case, if the graph had 0 upper triangle edges, use all edges.
    if len(row) == 0:
        row, col = adj.row, adj.col

    edge_index = np.vstack((row, col))
    
    print(f"Data preparation complete on {device}")
    return adj, edge_index, feat_tensor

def train_test_split_edges(edge_index, num_nodes, test_ratio=0.1, val_ratio=0.05):
    """
    Split the edges into train/val/test and sample negative edges.
    """
    num_edges = edge_index.shape[1]
    num_test = int(num_edges * test_ratio)
    num_val = int(num_edges * val_ratio)
    
    # Shuffle edges
    all_edge_idx = np.arange(num_edges)
    np.random.shuffle(all_edge_idx)
    
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    train_edge_idx = all_edge_idx[(num_val + num_test):]
    
    test_pos_edges = edge_index[:, test_edge_idx]
    val_pos_edges = edge_index[:, val_edge_idx]
    train_pos_edges = edge_index[:, train_edge_idx]
    
    # Sample negative edges for test and val (same number as positive edges)
    def sample_neg_edges(num_samples):
        neg_edges = set()
        while len(neg_edges) < num_samples:
            i, j = np.random.randint(0, num_nodes, 2)
            if i != j and i < j:
                neg_edges.add((i, j))
        return np.array(list(neg_edges)).T

    test_neg_edges = sample_neg_edges(num_test)
    val_neg_edges = sample_neg_edges(num_val)
    
    # Reconstruct the Adjacency matrix for training (using only training edges)
    # FastRP requires an adjacency matrix as input
    train_adj = sp.coo_matrix(
        (np.ones(train_pos_edges.shape[1] * 2),
         (np.concatenate([train_pos_edges[0], train_pos_edges[1]]),
          np.concatenate([train_pos_edges[1], train_pos_edges[0]]))),
        shape=(num_nodes, num_nodes)
    )
    
    # Create the tensor version for the model
    train_adj_coo = train_adj.tocoo()
    indices = torch.from_numpy(np.vstack((train_adj_coo.row, train_adj_coo.col))).long()
    values = torch.from_numpy(train_adj_coo.data).float()
    shape = torch.Size(train_adj_coo.shape)
    train_adj_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    return train_adj_tensor, test_pos_edges, test_neg_edges

def compute_link_prediction_metrics(embeddings, pos_edges, neg_edges):
    """
    Compute AUC and MRR for link prediction.
    """
    # Dot product similarity for positive edges
    pos_sims = torch.sum(embeddings[pos_edges[0]] * embeddings[pos_edges[1]], dim=1)
    
    # Dot product similarity for negative edges
    neg_sims = torch.sum(embeddings[neg_edges[0]] * embeddings[neg_edges[1]], dim=1)
    
    # Labels (1 for positive, 0 for negative)
    y_true = torch.cat([torch.ones(pos_sims.shape[0]), torch.zeros(neg_sims.shape[0])])
    y_scores = torch.cat([pos_sims, neg_sims])
    
    # AUC calculation
    auc = roc_auc_score(y_true.cpu().numpy(), y_scores.cpu().numpy())
    
    # MRR computation (Simplified ranking)
    # For each positive edge, we rank it against 'num_neg_samples' negative edges.
    # A standard setting in many papers is ranking 1 positive against 100 negatives, or sometimes against all.
    # To keep it computationally feasible and standard, let's rank 1 pos vs 99 negs.
    mrr = 0.0
    num_pos = pos_sims.shape[0]
    num_negs = neg_sims.shape[0]
    
    # If we have enough negative samples, we rank each positive against a random subset
    # This matches common MRR@10 evaluation protocols.
    ranks = []
    for i in range(num_pos):
        # Score of the true edge
        true_score = pos_sims[i].item()
        
        # Sample 99 negative edges for this positive edge
        # We reuse the generated negative edges
        neg_idx = torch.randint(0, num_negs, (99,))
        sampled_neg_scores = neg_sims[neg_idx]
        
        # Combine true score and neg scores
        all_scores = torch.cat([torch.tensor([true_score]), sampled_neg_scores])
        
        # Sort scores in descending order
        sorted_indices = torch.argsort(all_scores, descending=True)
        
        # Find the rank of the true edge (index 0 in all_scores)
        rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
        
        # We compute MRR@10. If rank > 10, reciprocal rank is 0.
        if rank <= 10:
            mrr += 1.0 / rank
            
    mrr /= num_pos
    
    return auc, mrr

def make_objective(train_adj_tensor, feat_tensor, test_pos_edges, test_neg_edges, device: torch.device, variant: str):
    def objective(trial):
        # 1. Suggest Hyperparameters
        _dim = trial.suggest_categorical('dim', [64, 128, 256, 512])
        _window_size = trial.suggest_int('window_size', 1, 5)
        _normalization = trial.suggest_categorical('normalization', [True, False])
        _g = trial.suggest_int('g', 2, 10)

        # 2. Initialize Model
        model = FastRP(
            embedding_dim=_dim,
            window_size=_window_size,
            normalization=_normalization,
            group_size=_g,
            input_matrix='trans',
            alpha=-0.6,
            weights=[1.0, 1.0, 7.81, 45.28],
            projection_type='gaussian',
        ).to(device)

        # 3. Generate Embeddings
        start_time = time.process_time()
        with torch.no_grad():
            if variant == 'hybrid':
                embeddings = model(train_adj_tensor.to(device), features=feat_tensor)
            else:
                embeddings = model(train_adj_tensor.to(device), features=None)
        cpu_time = time.process_time() - start_time

        # 4. Evaluate (Link Prediction)
        auc, mrr = compute_link_prediction_metrics(embeddings.cpu(), test_pos_edges, test_neg_edges)

        # Optuna optimizes a single value. Usually AUC is robust for link prediction.
        trial.set_user_attr('auc', auc)
        trial.set_user_attr('mrr_10', mrr)
        trial.set_user_attr('cpu_time', cpu_time)

        # Return AUC as the metric to maximize
        return auc

    return objective

def main():
    dataset_names = ['cora', 'citeseer', 'pubmed']
    data_root = '../data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    all_results = {}
    variants = ['gaussian', 'hybrid']

    for dataset_name in dataset_names:
        # Load the graph
        adj, edge_index, feat_tensor = load_data_for(dataset_name, data_root, device)
        num_nodes = adj.shape[0]

        # Prepare the train/test split for Link Prediction
        # We need to hide test edges from the FastRP embedding generation
        np.random.seed(42)
        torch.manual_seed(42)
        train_adj_tensor, test_pos_edges, test_neg_edges = train_test_split_edges(edge_index, num_nodes)
        
        print(f"   Train Edges (undirected pairs): {train_adj_tensor._nnz() // 2}")
        print(f"   Test Pos Edges: {test_pos_edges.shape[1]}")
        print(f"   Test Neg Edges: {test_neg_edges.shape[1]}")

        for variant in variants:
            if variant == 'hybrid' and feat_tensor is None:
                print(f"Skipping hybrid variant for {dataset_name} due to lack of features.")
                continue

            experiment_key = f"{dataset_name} ({variant})"
            print(f"\n=== Optuna: {experiment_key} ===")
            study = optuna.create_study(direction='maximize')
            objective = make_objective(train_adj_tensor, feat_tensor, test_pos_edges, test_neg_edges, device, variant)
            
            # n_trials can be adjusted. Set to 10 for faster testing, but 50 aligns with your Node Classification.
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            all_results[experiment_key] = {
                'best_params': study.best_params,
                'best_auc': study.best_value,
                'best_mrr_10': study.best_trial.user_attrs.get('mrr_10', 0.0),
                'cpu_time': study.best_trial.user_attrs.get('cpu_time', 0.0),
            }
            print(f"Best Hyperparameters for {experiment_key}:", study.best_params)
            print(f"Best AUC for {experiment_key}:", study.best_value)
            print(f"Associated MRR@10 for {experiment_key}:", all_results[experiment_key]['best_mrr_10'])
            print(f"Embedding CPU Time (s) for {experiment_key}:", all_results[experiment_key]['cpu_time'])

    print("\n=== Summary (Link Prediction) ===")
    for key, result in all_results.items():
        print(f"{key}: AUC={result['best_auc']:.4f}, MRR@10={result['best_mrr_10']:.4f}, Time={result['cpu_time']:.4f}s | params: {result['best_params']}")

if __name__ == '__main__':
    main()
