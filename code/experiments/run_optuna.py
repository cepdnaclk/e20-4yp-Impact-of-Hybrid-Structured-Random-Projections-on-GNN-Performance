import os
import sys
import torch
import numpy as np
import optuna

# Add 'src' to path so we can import FastRP
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from loaders import load_dataset
from fastrp_layer import FastRP
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


def load_data_for(dataset_name: str, data_root: str, device: torch.device):
    if dataset_name == 'flickr':
        flickr_path = os.path.join(data_root, 'flickr.mat')
        if not os.path.exists(flickr_path):
            raise FileNotFoundError(
                "Missing flickr.mat. Run: python ../src/get_flickr.py to download and convert it. "
                "If you already have a .mat file, place it at ../data/flickr.mat."
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


def make_objective(adj_tensor, labels, device: torch.device):
    def objective(trial):
        # 1. Suggest Hyperparameters
        _dim = trial.suggest_categorical('dim', [64, 128, 256, 512])
        _window_size = trial.suggest_int('window_size', 1, 5)
        _normalization = trial.suggest_categorical('normalization', [True, False])
        _g = trial.suggest_int('g', 2, 10)

        # 2. Initialize Model
        model = FastRP(
            embedding_dim=512,
            window_size=4,
            normalization=True,
            group_size=3,
            input_matrix='trans',
            alpha=-0.6,
            weights=[1.0, 1.0, 7.81, 45.28],
            projection_type='gaussian',
        ).to(device)

        # 3. Generate Embeddings
        with torch.no_grad():
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

        macro_f1 = f1_score(Y[indices[split:]], y_pred, average='macro')
        return macro_f1

    return objective


def main():
    dataset_names = ['blogcatalog', 'www10k', 'www200k']
    data_root = '../data'

    device = torch.device('cpu')
    print(f"Running on: {device}")

    all_results = {}

    for dataset_name in dataset_names:
        adj, features, labels, adj_tensor, feat_tensor = load_data_for(dataset_name, data_root, device)

        print(f"\n=== Optuna: {dataset_name} ===")
        study = optuna.create_study(direction='maximize')
        objective = make_objective(adj_tensor, labels, device)
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        all_results[dataset_name] = {
            'best_params': study.best_params,
            'best_macro_f1': study.best_value,
        }
        print("Best Hyperparameters:", study.best_params)
        print("Best Macro-F1:", study.best_value)

    print("\n=== Summary ===")
    for name, result in all_results.items():
        print(f"{name}: {result['best_macro_f1']:.4f} | params: {result['best_params']}")


if __name__ == '__main__':
    main()
