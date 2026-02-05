"""
DAG Utilities / Tiện ích DAG

Implements / Triển khai:
- GPU-accelerated DAG constraint (NOTEARS) / Ràng buộc DAG tăng tốc GPU (NOTEARS)
- Polynomial approximation fallback / Phương pháp dự phòng xấp xỉ đa thức
- to_dag() cycle removal / to_dag() loại bỏ chu trình
- Graph metrics (F1, SHD) / Các chỉ số đồ thị (F1, SHD)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


def calculate_dag_constraint(
    adjacency: torch.Tensor,
    method: str = 'exact'
) -> torch.Tensor:
    """
    Calculate DAG constraint h(A) = tr(e^A) - d.
    Tính ràng buộc DAG h(A) = tr(e^A) - d.
    
    h(A) = 0 iff A is a DAG.
    h(A) = 0 khi và chỉ khi A là một DAG.
    
    Args:
        adjacency: Weighted adjacency matrix (n, n) / Ma trận kề có trọng số
        method: 'exact' (matrix_exp) or 'polynomial' (faster, approximate)
                'chính xác' (hàm mũ ma trận) hoặc 'đa thức' (nhanh hơn, xấp xỉ)
        
    Returns:
        Scalar constraint value / Giá trị ràng buộc vô hướng
    """
    n = adjacency.shape[0]
    
    # Element-wise square for weighted version / Bình phương từng phần tử cho phiên bản có trọng số
    A = adjacency * adjacency
    
    if method == 'exact':
        # Exact matrix exponential (GPU-accelerated) / Hàm mũ ma trận chính xác (tăng tốc GPU)
        try:
            E = torch.linalg.matrix_exp(A)
            h = torch.trace(E) - n
        except RuntimeError:
            # Fallback to polynomial if matrix_exp fails / Dự phòng sang đa thức nếu matrix_exp thất bại
            h = _dag_constraint_polynomial(A, n)
    else:
        h = _dag_constraint_polynomial(A, n)
    
    return h


def _dag_constraint_polynomial(A: torch.Tensor, n: int, k: int = 10) -> torch.Tensor:
    """
    Polynomial approximation of matrix exponential trace.
    Xấp xỉ đa thức của vết hàm mũ ma trận.
    
    tr(e^A) ≈ tr(I + A + A²/2! + A³/3! + ... + Aᵏ/k!)
    
    Faster for large matrices but less accurate.
    Nhanh hơn đối với ma trận lớn nhưng kém chính xác hơn.
    """
    trace = float(n)  # tr(I) = n
    A_power = A.clone()
    factorial = 1.0
    
    for i in range(1, k + 1):
        factorial *= i
        trace = trace + torch.trace(A_power) / factorial
        if i < k:
            A_power = A_power @ A
    
    return trace - n


def to_dag(
    adjacency: torch.Tensor,
    threshold: float = 0.5,
    method: str = 'greedy'
) -> torch.Tensor:
    """
    Convert soft adjacency to strict DAG by removing cycles.
    Chuyển đổi ma trận kề mềm thành DAG nghiêm ngặt bằng cách loại bỏ chu trình.
    
    Args:
        adjacency: Soft adjacency matrix / Ma trận kề mềm
        threshold: Binarization threshold / Ngưỡng nhị phân hóa
        method: 'greedy' (fast) or 'optimal' (slower, better)
                'tham lam' (nhanh) hoặc 'tối ưu' (chậm hơn, tốt hơn)
        
    Returns:
        Binary DAG adjacency / Ma trận kề DAG nhị phân
    """
    n = adjacency.shape[0]
    device = adjacency.device
    
    # Binarize / Nhị phân hóa
    adj = (adjacency > threshold).float()
    adj = adj * (1 - torch.eye(n, device=device))  # No self-loops / Không có vòng lặp tự thân
    
    # Remove cycles / Loại bỏ chu trình
    max_iter = n * n
    for _ in range(max_iter):
        h = calculate_dag_constraint(adj, method='polynomial')
        if h < 1e-6:
            break
        
        cycle = _find_cycle(adj)
        if cycle is None:
            break
        
        # Remove weakest edge in cycle / Loại bỏ cạnh yếu nhất trong chu trình
        min_weight = float('inf')
        min_edge = None
        
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            weight = adjacency[u, v].item()
            if weight < min_weight:
                min_weight = weight
                min_edge = (u, v)
        
        if min_edge:
            adj[min_edge[0], min_edge[1]] = 0
    
    return adj


def _find_cycle(adj: torch.Tensor) -> Optional[list]:
    """Find a cycle using DFS. / Tìm một chu trình sử dụng DFS."""
    n = adj.shape[0]
    adj_np = (adj > 0.5).cpu().numpy()
    
    color = [0] * n  # 0=white, 1=gray, 2=black
    
    def dfs(node, path):
        color[node] = 1
        path.append(node)
        
        for neighbor in range(n):
            if adj_np[node, neighbor]:
                if color[neighbor] == 1:
                    idx = path.index(neighbor)
                    return path[idx:] + [neighbor]
                elif color[neighbor] == 0:
                    result = dfs(neighbor, path)
                    if result:
                        return result
        
        color[node] = 2
        path.pop()
        return None
    
    for start in range(n):
        if color[start] == 0:
            cycle = dfs(start, [])
            if cycle:
                return cycle
    
    return None


def compute_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Compute evaluation metrics.
    Tính toán các chỉ số đánh giá.
    
    Args:
        pred: Predicted adjacency / Ma trận kề dự đoán
        true: Ground truth adjacency / Ma trận kề thực tế
        threshold: Binarization threshold / Ngưỡng nhị phân hóa
        
    Returns:
        Dict with precision, recall, f1, shd, etc. / Dict chứa độ chính xác, độ phủ, f1, shd, v.v.
    """
    pred_bin = (pred > threshold).float()
    true_bin = (true > 0.5).float()
    
    # Remove diagonal / Loại bỏ đường chéo
    n = pred.shape[0]
    mask = 1 - torch.eye(n, device=pred.device)
    pred_bin = pred_bin * mask
    true_bin = true_bin * mask
    
    # Counts / Đếm
    tp = ((pred_bin == 1) & (true_bin == 1)).sum().float()
    fp = ((pred_bin == 1) & (true_bin == 0)).sum().float()
    fn = ((pred_bin == 0) & (true_bin == 1)).sum().float()
    tn = ((pred_bin == 0) & (true_bin == 0)).sum().float()
    
    # Metrics / Các chỉ số
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # SHD: structural hamming distance / Khoảng cách Hamming cấu trúc
    shd = (fp + fn).item()
    
    # FDR: false discovery rate / Tỷ lệ phát hiện sai
    fdr = fp / (tp + fp + 1e-10)
    
    # TPR: true positive rate (same as recall) / Tỷ lệ dương tính thật (giống độ phủ)
    tpr = recall
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'shd': int(shd),
        'tp': int(tp.item()),
        'fp': int(fp.item()),
        'fn': int(fn.item()),
        'fdr': fdr.item(),
        'tpr': tpr.item(),
        'n_edges_pred': int(pred_bin.sum().item()),
        'n_edges_true': int(true_bin.sum().item()),
    }


def compute_shd(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: float = 0.5
) -> int:
    """Compute Structural Hamming Distance. / Tính khoảng cách Hamming cấu trúc."""
    return compute_metrics(pred, true, threshold)['shd']


class AugmentedLagrangian(nn.Module):
    """
    Augmented Lagrangian for DAG constraint optimization.
    Lagrangian tăng cường cho tối ưu hóa ràng buộc DAG.
    
    L = f(θ) + α·h(A) + ρ/2·h(A)²
    
    Update rules / Quy tắc cập nhật:
    - α ← α + ρ·h(A)
    - ρ ← ρ·mult if h(A) > prev_h * threshold
    """
    
    def __init__(
        self,
        init_alpha: float = 0.0,
        init_rho: float = 0.01,
        rho_mult: float = 2.0,
        rho_max: float = 1e4,
        alpha_max: float = 1e10,
    ):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(init_alpha))
        self.register_buffer('rho', torch.tensor(init_rho))
        self.rho_mult = rho_mult
        self.rho_max = rho_max
        self.alpha_max = alpha_max
    
    def compute_penalty(self, h: torch.Tensor) -> torch.Tensor:
        """Compute augmented Lagrangian penalty. / Tính phạt Lagrangian tăng cường."""
        return self.alpha * h + 0.5 * self.rho * h * h
    
    def update_alpha(self, h: torch.Tensor):
        """Update Lagrange multiplier. / Cập nhật nhân tử Lagrange."""
        new_alpha = self.alpha + self.rho * h
        self.alpha = torch.clamp(new_alpha, max=self.alpha_max)
    
    def update_rho(self, factor: Optional[float] = None):
        """Increase penalty parameter. / Tăng tham số phạt."""
        factor = factor or self.rho_mult
        new_rho = self.rho * factor
        self.rho = torch.clamp(new_rho, max=self.rho_max)
    
    def reset(self, alpha: float = 0.0, rho: Optional[float] = None):
        """Reset parameters. / Đặt lại tham số."""
        self.alpha.fill_(alpha)
        if rho is not None:
            self.rho.fill_(rho)
