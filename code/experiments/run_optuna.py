import os
import sys
import torch
import numpy as np
import optuna

# Ensure 'src' is in path, or use absolute imports from root
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from loaders import load_dataset
from fastrp_layer import FastRP
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import warnings

# Suppress warnings from sklearn during Optuna trials
warnings.filterwarnings('ignore', category=UserWarning)

def load_data_for(dataset_name: str, data_root: str, device: torch.device):
    if dataset_name == 'flickr':
        flickr_path = os.path.join(data_root, 'flickr.mat')
        if not os.path.exists(flickr_path):
            raise FileNotFoundError(
                "Missing flickr.mat. Run: python ../src/get_flickr.py to download and convert it. "
                "If you already have a .mat file, place it at ../data/flickr.mat."
            )
    elif dataset_name == 'cora':
        cora_path = os.path.join(data_root, 'cora.mat')
        if not os.path.exists(cora_path):
            raise FileNotFoundError(
                "Missing cora.mat. Run: python ../src/get_cora.py to download and convert it. "
                "If you already have a .mat file, place it at ../data/cora.mat."
            )
    elif dataset_name == 'citeseer':
        citeseer_path = os.path.join(data_root, 'citeseer.mat')
        if not os.path.exists(citeseer_path):
            raise FileNotFoundError(
                "Missing citeseer.mat. Run: python -m src.get_citeseer to download and convert it."
            )
    elif dataset_name == 'pubmed':
        pubmed_path = os.path.join(data_root, 'pubmed.mat')
        if not os.path.exists(pubmed_path):
            raise FileNotFoundError(
                "Missing pubmed.mat. Run: python -m src.get_pubmed to download and convert it."
            )

    print(f"Loading {dataset_name}...")
    adj, features, labels = load_dataset(dataset_name, root_dir=data_root)

    # BlogCatalog labels are often a sparse matrix. Densify for slicing.
    if hasattr(labels, 'toarray'):
        labels = labels.toarray()

    # If labels are single-column (N, 1), flatten them for easier use
    if labels.shape[1] == 1:
        labels = labels.flatten()

    print(f"   Nodes: {adj.shape[0]}, Edges: {adj.nnz}")
    print(f"   Labels Shape: {labels.shape}")

    adj_coo = adj.tocoo()
    indices = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long()
    values = torch.from_numpy(adj_coo.data).float()
    shape = torch.Size(adj_coo.shape)
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device).coalesce()

    feat_tensor = None
    if features is not None:
        if hasattr(features, 'todense'):
            features = features.todense()
        feat_tensor = torch.FloatTensor(features).to(device)
        print(f"   Features loaded on GPU: {feat_tensor.shape}")

    print(f"Data preparation complete on {device}")
    return adj, features, labels, adj_tensor, feat_tensor


def make_objective(adj_tensor, feat_tensor, labels, device: torch.device, variant: str):
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
        with torch.no_grad():
            if variant == 'hybrid':
                embeddings = model(adj_tensor, features=feat_tensor)
            else:
                embeddings = model(adj_tensor, features=None)

        # 4. Evaluate (Downstream Task)
        X = embeddings.cpu().numpy()
        Y = labels

        indices = np.arange(X.shape[0])
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(0.8 * X.shape[0])

        clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=100))
        clf.fit(X[indices[:split]], Y[indices[:split]])
        y_pred = clf.predict(X[indices[split:]])
        y_prob = clf.predict_proba(X[indices[split:]])

        macro_f1 = f1_score(Y[indices[split:]], y_pred, average='macro')
        
        # Calculate ROC-AUC (One-vs-Rest, Macro)
        try:
            auc = roc_auc_score(Y[indices[split:]], y_prob, average='macro', multi_class='ovr')
        except ValueError:
            auc = 0.0 # Fallback if a class is entirely missing from the split

        # Calculate MRR@10
        mrr = 0.0
        for i, true_label in enumerate(Y[indices[split:]]):
            # Get the top 10 predicted classes sorted by probability (descending)
            top_10_indices = np.argsort(y_prob[i])[::-1][:10]
            # Find the rank of the true label
            if true_label in top_10_indices:
                rank = np.where(top_10_indices == true_label)[0][0] + 1
                mrr += 1.0 / rank
        mrr /= len(Y[indices[split:]])

        # Store additional metrics in the Optuna trial
        trial.set_user_attr('macro_f1', macro_f1)
        trial.set_user_attr('auc', auc)
        trial.set_user_attr('mrr_10', mrr)

        return macro_f1

    return objective


def main():
    dataset_names = ['cora', 'citeseer', 'pubmed']
    data_root = '../data'

    device = torch.device('cpu')
    print(f"Running on: {device}")

    all_results = {}
    variants = ['gaussian', 'hybrid']

    for dataset_name in dataset_names:
        adj, features, labels, adj_tensor, feat_tensor = load_data_for(dataset_name, data_root, device)

        for variant in variants:
            if variant == 'hybrid' and feat_tensor is None:
                print(f"Skipping hybrid variant for {dataset_name} due to lack of features.")
                continue

            experiment_key = f"{dataset_name} ({variant})"
            print(f"\n=== Optuna: {experiment_key} ===")
            study = optuna.create_study(direction='maximize')
            objective = make_objective(adj_tensor, feat_tensor, labels, device, variant)
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            all_results[experiment_key] = {
                'best_params': study.best_params,
                'best_macro_f1': study.best_value,
                'best_auc': study.best_trial.user_attrs.get('auc', 0.0),
                'best_mrr_10': study.best_trial.user_attrs.get('mrr_10', 0.0),
            }
            print(f"Best Hyperparameters for {experiment_key}:", study.best_params)
            print(f"Best Macro-F1 for {experiment_key}:", study.best_value)
            print(f"Associated AUC for {experiment_key}:", all_results[experiment_key]['best_auc'])
            print(f"Associated MRR@10 for {experiment_key}:", all_results[experiment_key]['best_mrr_10'])

    print("\n=== Summary ===")
    for key, result in all_results.items():
        print(f"{key}: F1={result['best_macro_f1']:.4f}, AUC={result['best_auc']:.4f}, MRR@10={result['best_mrr_10']:.4f} | params: {result['best_params']}")


if __name__ == '__main__':
    main()
