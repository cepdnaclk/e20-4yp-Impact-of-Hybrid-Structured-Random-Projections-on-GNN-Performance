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

def build_edge_set(edge_index):
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge_set.add((min(u, v), max(u, v)))
    return edge_set

def sample_negatives_per_positive(pos_edges, num_nodes, edge_set, k=500):
    num_pos = pos_edges.shape[1]
    neg_edges = np.zeros((2, num_pos, k), dtype=np.int64)
    
    for i in range(num_pos):
        assigned_negs = set()
        while len(assigned_negs) < k:
            # Batch sample random nodes to speed up
            u_batch = np.random.randint(0, num_nodes, size=k * 2)
            v_batch = np.random.randint(0, num_nodes, size=k * 2)
            for u, v in zip(u_batch, v_batch):
                if u == v:
                    continue
                canonical = (int(min(u, v)), int(max(u, v)))
                if canonical not in edge_set and canonical not in assigned_negs:
                    assigned_negs.add(canonical)
                if len(assigned_negs) == k:
                    break
                    
        assigned_negs_list = list(assigned_negs)
        neg_edges[0, i, :] = [e[0] for e in assigned_negs_list]
        neg_edges[1, i, :] = [e[1] for e in assigned_negs_list]
        
    return neg_edges

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
    
    # Build complete edge set to avoid sampling true edges
    edge_set = build_edge_set(edge_index)
    
    # Sample 500 negative edges specifically for each test positive edge
    test_neg_edges = sample_negatives_per_positive(test_pos_edges, num_nodes, edge_set, k=500)
    
    # Sample 500 negative edges specifically for each val positive edge
    val_neg_edges = sample_negatives_per_positive(val_pos_edges, num_nodes, edge_set, k=500)
    
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
    
    return train_adj_tensor, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges

def compute_link_prediction_metrics(embeddings, pos_edges, neg_edges):
    """
    Compute AUC and MRR for link prediction.
    pos_edges: shape (2, num_pos)
    neg_edges: shape (2, num_pos, 500)
    """
    num_pos = pos_edges.shape[1]
    
    # Process in batches to prevent Out-Of-Memory / slow thrashing 
    # especially for large graphs like pubmed where (num_pos, 500, dim) is huge.
    batch_size = 500
    pos_sims_list = []
    neg_sims_list = []
    
    for i in range(0, num_pos, batch_size):
        end = min(i + batch_size, num_pos)
        
        # Positive batch
        p_u = embeddings[pos_edges[0, i:end]]
        p_v = embeddings[pos_edges[1, i:end]]
        batch_pos_sims = torch.sum(p_u * p_v, dim=1)
        pos_sims_list.append(batch_pos_sims)
        
        # Negative batch
        n_u = embeddings[neg_edges[0, i:end]] # shape (batch_size, 500, dim)
        n_v = embeddings[neg_edges[1, i:end]] # shape (batch_size, 500, dim)
        batch_neg_sims = torch.sum(n_u * n_v, dim=2) # shape (batch_size, 500)
        neg_sims_list.append(batch_neg_sims)
        
    pos_sims = torch.cat(pos_sims_list, dim=0)
    neg_sims = torch.cat(neg_sims_list, dim=0)
    
    # --- AUC calculation ---
    # Flatten everything into single 1D arrays for sklearn AUC
    pos_sims_flat = pos_sims.cpu().numpy()
    neg_sims_flat = neg_sims.cpu().numpy().flatten()
    
    y_true = np.concatenate([np.ones(pos_sims_flat.shape[0]), np.zeros(neg_sims_flat.shape[0])])
    y_scores = np.concatenate([pos_sims_flat, neg_sims_flat])
    
    auc = roc_auc_score(y_true, y_scores)
    
    # --- MRR@10 computation ---
    # Expand pos_sims to shape (num_pos, 1) and concatenate with neg_sims (num_pos, 500)
    # to form an all_scores matrix of shape (num_pos, 501), where index 0 is the true edge's score.
    pos_sims_exp = pos_sims.unsqueeze(1)
    all_scores = torch.cat([pos_sims_exp, neg_sims], dim=1) # shape (num_pos, 501)
    
    # Sort descending along each row (dim=1)
    sorted_indices = torch.argsort(all_scores, dim=1, descending=True)
    
    # Find the rank of the true edge (index 0). We can do this efficiently without a loop.
    # sorted_indices == 0 gives a boolean matrix, nonzero gives the true edge's position.
    ranks_0_indexed = (sorted_indices == 0).nonzero(as_tuple=True)[1]
    ranks = ranks_0_indexed + 1 # Convert from 0-indexed to 1-indexed
    
    # Only points ranked in top 10 contribute to MRR@10, others count as 0
    mask = ranks <= 10
    
    # Sum the inverted ranks of valid entries and divide by total number of positive edges
    mrr = (1.0 / ranks[mask].float()).sum().item() / num_pos
    
    return auc, mrr

def make_objective(train_adj_tensor, feat_tensor, val_pos_edges, val_neg_edges, device: torch.device, variant: str):
    def objective(trial):
        # 1. Suggest Hyperparameters
        _dim = trial.suggest_categorical('dim', [64, 128, 256, 512])
        _window_size = trial.suggest_int('window_size', 1, 5)
        _normalization = trial.suggest_categorical('normalization', [True, False])
        _g = trial.suggest_int('g', 2, 10)

        # Parse variant
        proj_type = 'striped' if 'striped' in variant else 'gaussian'
        use_features = feat_tensor if 'hybrid' in variant else None

        # 2. Initialize Model
        model = FastRP(
            embedding_dim=_dim,
            window_size=_window_size,
            normalization=_normalization,
            group_size=_g,
            input_matrix='trans',
            alpha=-0.6,
            weights=[1.0, 1.0, 7.81, 45.28],
            projection_type=proj_type,
        ).to(device)

        # 3. Generate Embeddings
        start_time = time.process_time()
        with torch.no_grad():
            embeddings = model(train_adj_tensor.to(device), features=use_features)
        cpu_time = time.process_time() - start_time

        # 4. Evaluate (Link Prediction)
        auc, mrr = compute_link_prediction_metrics(embeddings.cpu(), val_pos_edges, val_neg_edges)

        trial.set_user_attr('auc', auc)
        trial.set_user_attr('mrr_10', mrr)
        trial.set_user_attr('cpu_time', cpu_time)

        # Return MRR@10 as the metric to maximize during validation
        return mrr

    return objective

def evaluate_best_model(study, train_adj_tensor, feat_tensor, test_pos_edges, test_neg_edges, device, variant, num_seeds=5):
    best_params = study.best_params
    proj_type = 'striped' if 'striped' in variant else 'gaussian'
    use_features = feat_tensor if 'hybrid' in variant else None

    test_aucs = []
    test_mrrs = []
    test_times = []

    seeds = [42, 100, 200, 300, 400]
    for seed in seeds[:num_seeds]:
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = FastRP(
            embedding_dim=best_params['dim'],
            window_size=best_params['window_size'],
            normalization=best_params['normalization'],
            group_size=best_params['g'],
            input_matrix='trans',
            alpha=-0.6,
            weights=[1.0, 1.0, 7.81, 45.28],
            projection_type=proj_type,
        ).to(device)

        start_time = time.process_time()
        with torch.no_grad():
            embeddings = model(train_adj_tensor.to(device), features=use_features)
        cpu_time = time.process_time() - start_time

        auc, mrr = compute_link_prediction_metrics(embeddings.cpu(), test_pos_edges, test_neg_edges)
        test_aucs.append(auc)
        test_mrrs.append(mrr)
        test_times.append(cpu_time)

    return {
        'auc_mean': np.mean(test_aucs),
        'auc_std': np.std(test_aucs),
        'mrr_mean': np.mean(test_mrrs),
        'mrr_std': np.std(test_mrrs),
        'time_mean': np.mean(test_times),
        'time_std': np.std(test_times)
    }

def main():
    dataset_names = ['cora', 'citeseer', 'pubmed']
    data_root = '../data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    all_results = {}
    variants = ['gaussian', 'striped', 'hybrid_gaussian', 'hybrid_striped']

    for dataset_name in dataset_names:
        # Load the graph
        adj, edge_index, feat_tensor = load_data_for(dataset_name, data_root, device)
        num_nodes = adj.shape[0]

        # Prepare the train/val/test split for Link Prediction
        np.random.seed(42)
        torch.manual_seed(42)
        train_adj_tensor, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges = train_test_split_edges(edge_index, num_nodes, test_ratio=0.1, val_ratio=0.05)
        
        print(f"   Train Edges (undirected pairs): {train_adj_tensor._nnz() // 2}")
        print(f"   Val Pos Edges: {val_pos_edges.shape[1]}")
        print(f"   Test Pos Edges: {test_pos_edges.shape[1]}")

        for variant in variants:
            if 'hybrid' in variant and feat_tensor is None:
                print(f"Skipping {variant} for {dataset_name} due to lack of features.")
                continue

            experiment_key = f"{dataset_name} ({variant})"
            print(f"\n=== Optuna: {experiment_key} ===")
            study = optuna.create_study(direction='maximize')
            objective = make_objective(train_adj_tensor, feat_tensor, val_pos_edges, val_neg_edges, device, variant)
            
            # n_trials can be adjusted. Set to 10 for faster testing, but 50 aligns with your Node Classification.
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            print(f"Best Val Params for {experiment_key}:", study.best_params)
            print(f"Best Val MRR@10 for {experiment_key}: {study.best_value * 100:.2f}%")

            # Final Evaluation on Test Set across multiple seeds
            test_results = evaluate_best_model(study, train_adj_tensor, feat_tensor, test_pos_edges, test_neg_edges, device, variant, num_seeds=5)

            all_results[experiment_key] = {
                'best_params': study.best_params,
                'val_mrr': study.best_value,
                'test_auc_mean': test_results['auc_mean'],
                'test_auc_std': test_results['auc_std'],
                'test_mrr_mean': test_results['mrr_mean'],
                'test_mrr_std': test_results['mrr_std'],
                'test_time_mean': test_results['time_mean'],
                'test_time_std': test_results['time_std'],
            }
            print(f"Test AUC for {experiment_key}: {test_results['auc_mean']*100:.2f} ± {test_results['auc_std']*100:.2f}%")
            print(f"Test MRR@10 for {experiment_key}: {test_results['mrr_mean']*100:.2f} ± {test_results['mrr_std']*100:.2f}%")

    print("\n=== Summary (Link Prediction Final Test Results) ===")
    for key, result in all_results.items():
        print(f"{key}: AUC={result['test_auc_mean']*100:.2f}±{result['test_auc_std']*100:.2f}%, MRR={result['test_mrr_mean']*100:.2f}±{result['test_mrr_std']*100:.2f}%, Time={result['test_time_mean']:.4f}±{result['test_time_std']:.4f}s | params: {result['best_params']}")

if __name__ == '__main__':
    main()
