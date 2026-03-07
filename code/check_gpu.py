import torch
import numpy as np
from ogb.nodeproppred import NodePropPredDataset

print(f"🔥 PyTorch Version: {torch.__version__}")
print(f"🔎 CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cpu")
    print(f"✅ Selected Device: cpu")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test a small sparse matrix multiplication on GPU
    try:
        # Create a random sparse tensor
        i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        v = torch.FloatTensor([3, 4, 5])
        sparse_tensor = torch.sparse_coo_tensor(i, v, (2, 3)).to(device)
        dense_matrix = torch.randn(3, 4).to(device)
        
        # Perform multiplication
        result = torch.sparse.mm(sparse_tensor, dense_matrix)
        print("✅ GPU Sparse Multiplication Test: PASSED")
    except Exception as e:
        print(f"❌ GPU Sparse Math Failed: {e}")
else:
    print("❌ GPU NOT Detected. Check your drivers.")

print("\n--- OGB Data Check ---")
try:
    # Use a small dataset just to test the download/load mechanism
    # We will save it to the 'data' folder we planned
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='data')
    print("✅ OGB-Arxiv Loaded Successfully.")
    graph = dataset[0][0]
    print(f"   Nodes: {graph['num_nodes']}")
    print(f"   Edges: {graph['edge_index'].shape[1]}")
except Exception as e:
    print(f"❌ OGB Load Failed: {e}")