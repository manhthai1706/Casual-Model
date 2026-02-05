"""
CausalMLP Training Script / Kịch bản huấn luyện CausalMLP

Train CausalMLP on various datasets including Sachs.
Huấn luyện CausalMLP trên nhiều tập dữ liệu khác nhau bao gồm Sachs.

Usage / Cách dùng:
    python train.py                          # Synthetic data / Dữ liệu tổng hợp
    python train.py --dataset sachs          # Sachs dataset / Dữ liệu Sachs
    python train.py --num_nodes 20 --samples 2000  # Custom / Tùy chỉnh
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import sys

# Add current directory to path / Thêm thư mục hiện tại vào đường dẫn
sys.path.insert(0, str(Path(__file__).parent))

from config import CausalMLPConfig
from core.model import CausalMLPModel
from training.trainer import CurriculumTrainer
from utils.dag_utils import compute_metrics, to_dag


# Sachs ground truth (consensus network) / Ground truth của Sachs (mạng đồng thuận)
SACHS_GROUND_TRUTH = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Raf
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Mek
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Plcg
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # PIP2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # PIP3
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Erk
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Akt
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],  # PKA
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # PKC
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # P38
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Jnk
], dtype=np.float32)

SACHS_NODE_NAMES = ['Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk']


def generate_synthetic_data(
    adjacency: np.ndarray,
    n_samples: int = 1000,
    noise_std: float = 0.3,
    seed: int = 42
) -> np.ndarray:
    """Generate data from linear SEM. / Tạo dữ liệu từ SEM tuyến tính."""
    np.random.seed(seed)
    n = adjacency.shape[0]
    
    # Random weights / Trọng số ngẫu nhiên
    W = adjacency * (np.random.randn(n, n) * 0.5 + 0.5)
    
    # Topological sort / Sắp xếp topo
    in_degree = adjacency.sum(axis=0).astype(int)
    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []
    temp_in = in_degree.copy()
    
    while queue:
        node = queue.pop(0)
        order.append(node)
        for j in range(n):
            if adjacency[node, j]:
                temp_in[j] -= 1
                if temp_in[j] == 0:
                    queue.append(j)
    
    if len(order) != n:
        order = list(range(n))
    
    # Generate samples / Tạo mẫu
    X = np.zeros((n_samples, n))
    for node in order:
        X[:, node] = X @ W[:, node] + np.random.randn(n_samples) * noise_std
    
    # Standardize / Chuẩn hóa
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return X


def load_sachs_data():
    """Load real Sachs dataset if available. / Tải tập dữ liệu Sachs thực nếu có."""
    sachs_path = Path(__file__).parent.parent / 'MLP-DAG' / 'data' / 'sachs' / 'continuous' / 'data1.npy'
    
    if sachs_path.exists():
        data = np.load(sachs_path)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data
    else:
        print("Real Sachs data not found, generating synthetic...")
        return generate_synthetic_data(SACHS_GROUND_TRUTH)


def main():
    parser = argparse.ArgumentParser(description='Train CausalMLP / Huấn luyện CausalMLP')
    
    # Dataset / Tập dữ liệu
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'sachs', 'custom'])
    parser.add_argument('--num_nodes', type=int, default=10)
    parser.add_argument('--samples', type=int, default=1000)
    
    # Architecture / Kiến trúc
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--adjacency_type', type=str, default='soft',
                       choices=['soft', 'enco', 'dual'])
    parser.add_argument('--noise_type', type=str, default='gaussian',
                       choices=['gaussian', 'heteroscedastic', 'adaptive'])
    
    # Training / Huấn luyện
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--max_outer', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--sparsity', type=float, default=0.001)
    
    # Other / Khác
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    # Set seeds / Thiết lập seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("CausalMLP: Advanced Hybrid Causal Discovery")
    print("CausalMLP: Khám phá nhân quả lai tiên tiến")
    print("=" * 60)
    
    # Load data / Tải dữ liệu
    if args.dataset == 'sachs':
        data = load_sachs_data()
        true_adj = torch.tensor(SACHS_GROUND_TRUTH)
        num_nodes = 11
        print(f"Dataset: Sachs ({data.shape[0]} samples, {num_nodes} nodes)")
    elif args.dataset == 'synthetic':
        # Random DAG / DAG ngẫu nhiên
        n = args.num_nodes
        prob = 2 / n  # Sparse graph / Đồ thị thưa
        adj = np.triu(np.random.rand(n, n) < prob, k=1).astype(np.float32)
        data = generate_synthetic_data(adj, args.samples, seed=args.seed)
        true_adj = torch.tensor(adj)
        num_nodes = n
        print(f"Dataset: Synthetic ({args.samples} samples, {num_nodes} nodes, {int(adj.sum())} edges)")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    data = torch.tensor(data, dtype=torch.float32)
    
    # Create config / Tạo cấu hình
    config = CausalMLPConfig(
        num_nodes=num_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        adjacency_type=args.adjacency_type,
        noise_type=args.noise_type,
        warmup_steps=args.warmup_steps,
        max_outer_iter=args.max_outer,
        warmup_lr=args.lr,
        main_lr=args.lr * 0.7,
        sparsity_lambda=args.sparsity,
    )
    
    print(f"\nConfiguration / Cấu hình:")
    print(f"  Adjacency: {config.adjacency_type}")
    print(f"  MLP: {config.hidden_dim}x{config.num_layers}")
    print(f"  Noise: {config.noise_type}")
    print(f"  Warmup: {config.warmup_steps} steps")
    print(f"  Max outer: {config.max_outer_iter}")
    
    # Create model / Tạo mô hình
    model = CausalMLPModel(config)
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Train / Huấn luyện
    trainer = CurriculumTrainer(model, config)
    result = trainer.fit(data, true_adj, verbose=True)
    
    # Final evaluation / Đánh giá cuối cùng
    print("\n" + "=" * 60)
    print("FINAL RESULTS / KẾT QUẢ CUỐI CÙNG")
    print("=" * 60)
    
    with torch.no_grad():
        adj = model.adjacency.probs.cpu().numpy()
    
    print("\nContinuous adjacency (rounded) / Ma trận kề liên tục (làm tròn):")
    print(np.round(adj, 2))
    
    for thresh in [0.2, 0.3, 0.5]:
        metrics = compute_metrics(
            torch.tensor(adj),
            true_adj,
            threshold=thresh
        )
        print(f"\nThreshold {thresh}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  SHD: {metrics['shd']}")
    
    print(f"\nBest F1 (during training): {result['best_f1']:.3f}")
    
    # Save results / Lưu kết quả
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'adjacency.npy', adj)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'best_f1': result['best_f1'],
            'history': result['history'],
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
