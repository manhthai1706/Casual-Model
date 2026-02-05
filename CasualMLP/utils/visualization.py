"""
Visualization for CausalMLP / Trực quan hóa cho CausalMLP

Provides / Cung cấp:
- Graph plotting with NetworkX / Vẽ đồ thị với NetworkX
- Training curves / Đường cong huấn luyện
- Adjacency heatmaps / Bản đồ nhiệt ma trận kề
- Comparison with ground truth / So sánh với ground truth
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path


def plot_adjacency_heatmap(
    adjacency: np.ndarray,
    node_names: Optional[List[str]] = None,
    title: str = 'Learned Adjacency',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
):
    """
    Plot adjacency matrix as heatmap.
    Vẽ ma trận kề dưới dạng bản đồ nhiệt.
    
    Args:
        adjacency: Adjacency matrix / Ma trận kề
        node_names: Names for nodes / Tên cho các nút
        title: Plot title / Tiêu đề biểu đồ
        save_path: Path to save figure / Đường dẫn lưu hình
        figsize: Figure size / Kích thước hình
        cmap: Colormap / Bảng màu
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for visualization")
        return
    
    if torch.is_tensor(adjacency):
        adjacency = adjacency.numpy()
    
    n = adjacency.shape[0]
    
    if node_names is None:
        node_names = [f'X{i}' for i in range(n)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        adjacency,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        xticklabels=node_names,
        yticklabels=node_names,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    
    ax.set_title(title)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    
    plt.close()
    return fig


def plot_graph(
    adjacency: np.ndarray,
    node_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    true_adjacency: Optional[np.ndarray] = None,
    title: str = 'Learned DAG',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot the learned graph structure.
    Vẽ cấu trúc đồ thị đã học.
    
    Args:
        adjacency: Adjacency matrix / Ma trận kề
        node_names: Names for nodes / Tên cho các nút
        threshold: Binarization threshold / Ngưỡng nhị phân hóa
        true_adjacency: Ground truth for color coding / Ground truth cho mã màu
        title: Plot title / Tiêu đề biểu đồ
        save_path: Path to save figure / Đường dẫn lưu hình
        figsize: Figure size / Kích thước hình
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("matplotlib and networkx required for visualization")
        return
    
    if torch.is_tensor(adjacency):
        adjacency = adjacency.numpy()
    if true_adjacency is not None and torch.is_tensor(true_adjacency):
        true_adjacency = true_adjacency.numpy()
    
    n = adjacency.shape[0]
    
    if node_names is None:
        node_names = [f'X{i}' for i in range(n)]
    
    adj_bin = (adjacency > threshold).astype(float)
    
    # Create graph / Tạo đồ thị
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # Add edges with colors / Thêm cạnh với màu sắc
    edge_colors = []
    edge_widths = []
    
    for i in range(n):
        for j in range(n):
            if adj_bin[i, j] > 0:
                G.add_edge(i, j)
                weight = adjacency[i, j]
                edge_widths.append(weight * 3)
                
                if true_adjacency is not None:
                    if true_adjacency[i, j] > 0:
                        edge_colors.append('green')  # True positive / Dương tính thật
                    else:
                        edge_colors.append('red')    # False positive / Dương tính giả
                else:
                    edge_colors.append('steelblue')
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Plot / Vẽ
    fig, ax = plt.subplots(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, 
                          node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels={i: node_names[i] for i in range(n)}, 
                           ax=ax, font_size=10, font_weight='bold')
    
    if edge_colors:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                              width=edge_widths if edge_widths else 1.5,
                              arrows=True, arrowsize=20,
                              connectionstyle='arc3,rad=0.1',
                              alpha=0.8)
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    if true_adjacency is not None:
        legend_text = 'Green = True Positive, Red = False Positive'
        ax.text(0.5, -0.05, legend_text, transform=ax.transAxes,
               fontsize=10, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    
    plt.close()
    return fig


def plot_training_curves(
    history: List[Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot training curves from history.
    Vẽ đường cong huấn luyện từ lịch sử.
    
    Args:
        history: List of dicts with training metrics / Danh sách dict chứa các chỉ số huấn luyện
        save_path: Path to save figure / Đường dẫn lưu hình
        figsize: Figure size / Kích thước hình
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    if not history:
        print("Empty history")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract metrics / Trích xuất chỉ số
    outer_iters = [h.get('outer', i) for i, h in enumerate(history)]
    
    # Plot DAG constraint / Vẽ ràng buộc DAG
    ax = axes[0, 0]
    h_values = [h.get('h', 0) for h in history]
    ax.semilogy(outer_iters, h_values, 'b-o')
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('DAG Constraint h(A)')
    ax.set_title('DAG Constraint / Ràng buộc DAG')
    ax.grid(True, alpha=0.3)
    
    # Plot F1 scores / Vẽ điểm F1
    ax = axes[0, 1]
    if 'f1_0.3' in history[0]:
        f1_03 = [h.get('f1_0.3', 0) for h in history]
        ax.plot(outer_iters, f1_03, 'g-o', label='threshold=0.3')
    if 'f1_0.5' in history[0]:
        f1_05 = [h.get('f1_0.5', 0) for h in history]
        ax.plot(outer_iters, f1_05, 'r-s', label='threshold=0.5')
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score / Điểm F1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot alpha and rho / Vẽ alpha và rho
    ax = axes[1, 0]
    alpha = [h.get('alpha', 0) for h in history]
    rho = [h.get('rho', 0) for h in history]
    ax.semilogy(outer_iters, alpha, 'b-o', label='alpha')
    ax.semilogy(outer_iters, rho, 'r-s', label='rho')
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Augmented Lagrangian Parameters / Tham số Lagrangian tăng cường')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot loss / Vẽ mất mát
    ax = axes[1, 1]
    if 'mean_loss' in history[0]:
        mean_loss = [h.get('mean_loss', 0) for h in history]
        ax.plot(outer_iters, mean_loss, 'b-o')
        ax.set_ylabel('Mean Loss')
    if 'nll' in history[0]:
        nll = [h.get('nll', 0) for h in history]
        ax.plot(outer_iters, nll, 'g-s', label='NLL')
        ax.legend()
    ax.set_xlabel('Outer Iteration')
    ax.set_title('Training Loss / Mất mát huấn luyện')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    
    plt.close()
    return fig


def compare_adjacencies(
    predicted: np.ndarray,
    true: np.ndarray,
    node_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    Compare predicted and true adjacency matrices side by side.
    So sánh ma trận kề dự đoán và thực tế cạnh nhau.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for visualization")
        return
    
    if torch.is_tensor(predicted):
        predicted = predicted.numpy()
    if torch.is_tensor(true):
        true = true.numpy()
    
    n = predicted.shape[0]
    
    if node_names is None:
        node_names = [f'X{i}' for i in range(n)]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Predicted / Dự đoán
    sns.heatmap(predicted, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=node_names, yticklabels=node_names,
               ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Predicted')
    
    # True / Thực tế
    sns.heatmap(true, annot=True, fmt='.0f', cmap='Greens',
               xticklabels=node_names, yticklabels=node_names,
               ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    
    # Difference / Khác biệt
    pred_bin = (predicted > threshold).astype(float)
    diff = np.zeros_like(predicted)
    diff[(pred_bin == 1) & (true == 1)] = 2    # TP
    diff[(pred_bin == 1) & (true == 0)] = 1    # FP
    diff[(pred_bin == 0) & (true == 1)] = -1   # FN
    
    cmap = plt.cm.RdYlGn
    sns.heatmap(diff, annot=False, cmap=cmap, center=0,
               xticklabels=node_names, yticklabels=node_names,
               ax=axes[2], vmin=-1, vmax=2)
    axes[2].set_title('Comparison (Green=TP, Yellow=FP, Red=FN)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    
    plt.close()
    return fig
