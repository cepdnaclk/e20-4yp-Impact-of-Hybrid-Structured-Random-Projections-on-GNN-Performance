import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import scipy.sparse as sp
import torch
from ogb.nodeproppred import NodePropPredDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add src path for FastRP import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastrp_layer import FastRP


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ogbn_arxiv(data_root: str, device: torch.device):
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=data_root)
    graph, labels_raw = dataset[0]
    split_idx = dataset.get_idx_split()

    labels = labels_raw.flatten()
    num_nodes = graph["num_nodes"]
    edge_index = graph["edge_index"]

    # Symmetrize directed citation graph for diffusion-style propagation.
    vals = np.ones(edge_index.shape[1], dtype=np.float32)
    adj_dir = sp.csr_matrix((vals, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    adj_sym = adj_dir + adj_dir.T
    adj_sym.data = np.ones_like(adj_sym.data, dtype=np.float32)

    coo = adj_sym.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data).float()
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce().to(device)

    feat_tensor = torch.FloatTensor(graph["node_feat"]).to(device)

    print("Loaded ogbn-arxiv")
    print(f"  Nodes: {num_nodes}")
    print(f"  Directed edges: {edge_index.shape[1]}")
    print(f"  Symmetrized edges: {adj_sym.nnz}")
    print(f"  Features: {tuple(graph['node_feat'].shape)}")
    print(f"  Split sizes: train={len(split_idx['train'])}, valid={len(split_idx['valid'])}, test={len(split_idx['test'])}")

    return adj_tensor, feat_tensor, labels, split_idx


def fit_probe_and_scores(embeddings_np, labels_np, split_idx, c_value: float, max_iter: int):
    idx_train = split_idx["train"]
    idx_valid = split_idx["valid"]
    idx_test = split_idx["test"]

    clf = LogisticRegression(
        solver="saga",
        C=c_value,
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,
        tol=1e-3,
    )
    clf.fit(embeddings_np[idx_train], labels_np[idx_train])

    y_train = clf.predict(embeddings_np[idx_train])
    y_valid = clf.predict(embeddings_np[idx_valid])
    y_test = clf.predict(embeddings_np[idx_test])

    return {
        "train": float(accuracy_score(labels_np[idx_train], y_train)),
        "valid": float(accuracy_score(labels_np[idx_valid], y_valid)),
        "test": float(accuracy_score(labels_np[idx_test], y_test)),
    }


def make_objective(adj_tensor, feat_tensor, labels, split_idx, device: torch.device):
    def objective(trial: optuna.trial.Trial):
        embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512, 768])
        window_size = trial.suggest_int("window_size", 2, 6)
        normalization = trial.suggest_categorical("normalization", [True, False])
        group_size = trial.suggest_int("group_size", 2, 16)
        alpha = trial.suggest_float("alpha", -1.0, 0.0)

        logreg_c = trial.suggest_float("logreg_C", 1e-3, 1e2, log=True)
        logreg_max_iter = trial.suggest_categorical("logreg_max_iter", [500, 1000, 2000])

        params = {
            "embedding_dim": embedding_dim,
            "window_size": window_size,
            "normalization": normalization,
            "group_size": group_size,
            "input_matrix": "trans",
            "alpha": alpha,
            "weights": [1.0, 1.0, 7.81, 45.28],
        }

        set_seeds(42)
        model = FastRP(**params, projection_type="striped").to(device)

        start = time.time()
        with torch.no_grad():
            embeddings = model(adj_tensor, features=feat_tensor)
        emb_time = time.time() - start

        x = embeddings.detach().cpu().numpy()
        scores = fit_probe_and_scores(x, labels, split_idx, c_value=logreg_c, max_iter=logreg_max_iter)

        trial.set_user_attr("train_acc", scores["train"])
        trial.set_user_attr("valid_acc", scores["valid"])
        trial.set_user_attr("test_acc_snapshot", scores["test"])
        trial.set_user_attr("embedding_time_sec", emb_time)

        return scores["valid"]

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuning for FastRP Hybrid on ogbn-arxiv")
    parser.add_argument("--data-root", type=str, default="../data", help="Path containing ogbn-arxiv cache")
    parser.add_argument("--trials", type=int, default=150, help="Target total trial budget")
    parser.add_argument("--study-name", type=str, default="fastrp_ogbn_arxiv_hybrid_tuning")
    parser.add_argument("--db", type=str, default="optuna_ogbn_arxiv.db", help="SQLite file path")
    parser.add_argument("--artifact", type=str, default="optuna_ogbn_arxiv_best.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-startup-trials", type=int, default=20)
    return parser.parse_args()


def make_trial_logger():
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        state = trial.state.name
        msg = f"[trial {trial.number:04d}] state={state}"
        if trial.value is not None:
            msg += f" valid={trial.value:.4f}"
        train_acc = trial.user_attrs.get("train_acc")
        test_snap = trial.user_attrs.get("test_acc_snapshot")
        if train_acc is not None:
            msg += f" train={train_acc:.4f}"
        if test_snap is not None:
            msg += f" test_snapshot={test_snap:.4f}"
        if study.best_trial is not None:
            msg += f" | best_valid={study.best_value:.4f} (trial {study.best_trial.number})"
        print(msg, flush=True)

    return _callback


def main():
    args = parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    db_path = Path(args.db).resolve()
    artifact_path = Path(args.artifact).resolve()
    storage = f"sqlite:///{db_path}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=0)

    adj_tensor, feat_tensor, labels, split_idx = load_ogbn_arxiv(args.data_root, device)
    objective = make_objective(adj_tensor, feat_tensor, labels, split_idx, device)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    completed = len(study.trials)
    remaining = max(0, args.trials - completed)
    print(f"Study: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Completed trials: {completed}")
    print(f"Remaining to target: {remaining}")

    if remaining > 0:
        callback = make_trial_logger()
        study.optimize(objective, n_trials=remaining, show_progress_bar=True, callbacks=[callback])
    else:
        print("Target trial budget already reached. Skipping optimization.")

    best = study.best_trial
    bp = best.params

    best_fastrp_params = {
        "embedding_dim": bp["embedding_dim"],
        "window_size": bp["window_size"],
        "normalization": bp["normalization"],
        "group_size": bp["group_size"],
        "input_matrix": "trans",
        "alpha": bp["alpha"],
        "weights": [1.0, 1.0, 7.81, 45.28],
    }

    set_seeds(args.seed)
    best_model = FastRP(**best_fastrp_params, projection_type="striped").to(device)
    with torch.no_grad():
        best_embeddings = best_model(adj_tensor, features=feat_tensor)

    best_scores = fit_probe_and_scores(
        best_embeddings.detach().cpu().numpy(),
        labels,
        split_idx,
        c_value=bp["logreg_C"],
        max_iter=bp["logreg_max_iter"],
    )

    print("\n=== Best Trial (Selected by Validation Accuracy) ===")
    print(f"Trial: {best.number}")
    print(f"Best valid acc: {best_scores['valid']:.4f}")
    print(f"Train acc: {best_scores['train']:.4f}")
    print(f"Test acc (final one-time eval): {best_scores['test']:.4f}")
    print(f"Params: {bp}")

    artifact = {
        "study_name": args.study_name,
        "storage": storage,
        "n_trials_total": len(study.trials),
        "best_trial_number": best.number,
        "best_params": bp,
        "metrics": {
            "train_acc": best_scores["train"],
            "valid_acc": best_scores["valid"],
            "test_acc": best_scores["test"],
        },
    }
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"Saved artifact: {artifact_path}")


if __name__ == "__main__":
    main()
