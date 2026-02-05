"""
CAM Pruning for CausalMLP / Cắt tỉa CAM cho CausalMLP

Post-hoc pruning using Causal Additive Models:
Cắt tỉa hậu nghiệm sử dụng Mô hình Cộng tính Nhân quả (CAM):
- Correlation-based pruning / Cắt tỉa dựa trên tương quan
- Regression-based pruning (linear and non-linear) / Cắt tỉa dựa trên hồi quy (tuyến tính và phi tuyến tính)
- Statistical significance testing / Kiểm tra ý nghĩa thống kê
- Iterative edge removal / Loại bỏ cạnh lặp đi lặp lại

Based on GraN-DAG's CAM pruning (Python implementation).
Dựa trên cắt tỉa CAM của GraN-DAG (triển khai Python).
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy import stats


def cam_pruning(
    data: np.ndarray,
    adjacency: np.ndarray,
    threshold: float = 0.5,
    method: str = 'regression',
    alpha: float = 0.05,
    verbose: bool = False,
) -> np.ndarray:
    """
    CAM-style pruning to remove spurious edges.
    Cắt tỉa kiểu CAM để loại bỏ các cạnh giả mạo.
    
    Tests each edge for statistical significance using residual analysis.
    Kiểm tra ý nghĩa thống kê của từng cạnh sử dụng phân tích phần dư.
    
    Args:
        data: Data matrix (n_samples, n_nodes) / Ma trận dữ liệu
        adjacency: Initial adjacency matrix (n_nodes, n_nodes) / Ma trận kề ban đầu
        threshold: Binarization threshold for adjacency / Ngưỡng nhị phân hóa cho ma trận kề
        method: 'correlation', 'regression', or 'partial_correlation' / Phương pháp: 'tương quan', 'hồi quy', hoặc 'tương quan riêng'
        alpha: Significance level for tests / Mức ý nghĩa cho các kiểm định
        verbose: Print pruning progress / In tiến trình cắt tỉa
        
    Returns:
        Pruned binary adjacency matrix / Ma trận kề nhị phân đã được cắt tỉa
    """
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(adjacency):
        adjacency = adjacency.numpy()
    
    # Binarize / Nhị phân hóa
    adj = (adjacency > threshold).astype(float)
    n = adj.shape[0]
    
    if method == 'correlation':
        pruned = _correlation_pruning(data, adj, alpha, verbose)
    elif method == 'regression':
        pruned = _regression_pruning(data, adj, alpha, verbose)
    elif method == 'partial_correlation':
        pruned = _partial_correlation_pruning(data, adj, alpha, verbose)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return pruned


def _correlation_pruning(
    data: np.ndarray,
    adj: np.ndarray,
    alpha: float,
    verbose: bool,
) -> np.ndarray:
    """Prune edges with low correlation. / Cắt tỉa các cạnh có tương quan thấp."""
    n = adj.shape[0]
    pruned = adj.copy()
    
    for i in range(n):
        for j in range(n):
            if adj[j, i] > 0:  # Edge j -> i exists / Cạnh j -> i tồn tại
                corr, p_value = stats.pearsonr(data[:, j], data[:, i])
                
                if p_value > alpha:
                    pruned[j, i] = 0
                    if verbose:
                        print(f"Pruned edge {j}->{i}: corr={corr:.3f}, p={p_value:.3f}")
    
    return pruned


def _regression_pruning(
    data: np.ndarray,
    adj: np.ndarray,
    alpha: float,
    verbose: bool,
) -> np.ndarray:
    """
    Prune edges using regression significance.
    Cắt tỉa các cạnh sử dụng ý nghĩa hồi quy.
    
    For each node i with parents PA_i:
    Với mỗi nút i có cha mẹ PA_i:
    1. Fit regression: X_i ~ PA_i / Khớp hồi quy: X_i ~ PA_i
    2. For each parent j, test if coefficient is significant / Với mỗi cha mẹ j, kiểm tra xem hệ số có ý nghĩa không
    3. Remove edge if not significant / Loại bỏ cạnh nếu không có ý nghĩa
    """
    n = adj.shape[0]
    pruned = adj.copy()
    n_samples = data.shape[0]
    
    for i in range(n):
        parents = np.where(adj[:, i] > 0)[0]
        
        if len(parents) == 0:
            continue
        
        # Get parent data
        X = data[:, parents]
        y = data[:, i]
        
        # Add intercept / Thêm hệ số chắn
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        
        try:
            # OLS regression / Hồi quy OLS
            beta, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            
            # Compute standard errors / Tính sai số chuẩn
            y_pred = X_with_intercept @ beta
            residual = y - y_pred
            mse = np.sum(residual ** 2) / (n_samples - len(beta))
            
            # Covariance matrix of coefficients / Ma trận hiệp phương sai của các hệ số
            try:
                var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                se = np.sqrt(np.diag(var_beta))
            except np.linalg.LinAlgError:
                continue
            
            # t-statistics (skip intercept) / thống kê t (bỏ qua hệ số chắn)
            t_stats = beta[1:] / (se[1:] + 1e-10)
            
            # p-values (two-tailed) / giá trị p (hai đuôi)
            dof = n_samples - len(beta)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
            
            # Prune non-significant edges / Cắt tỉa các cạnh không có ý nghĩa
            for idx, j in enumerate(parents):
                if p_values[idx] > alpha:
                    pruned[j, i] = 0
                    if verbose:
                        print(f"Pruned edge {j}->{i}: t={t_stats[idx]:.3f}, p={p_values[idx]:.3f}")
        
        except Exception as e:
            if verbose:
                print(f"Warning: Regression failed for node {i}: {e}")
            continue
    
    return pruned


def _partial_correlation_pruning(
    data: np.ndarray,
    adj: np.ndarray,
    alpha: float,
    verbose: bool,
) -> np.ndarray:
    """
    Prune using partial correlations.
    Cắt tỉa sử dụng tương quan riêng.
    
    Tests if X_j is independent of X_i given other parents.
    Kiểm tra xem X_j có độc lập với X_i khi biết các cha mẹ khác không.
    """
    n = adj.shape[0]
    pruned = adj.copy()
    
    for i in range(n):
        parents = np.where(adj[:, i] > 0)[0]
        
        if len(parents) <= 1:
            continue
        
        for j in parents:
            # Other parents (conditioning set) / Cha mẹ khác (tập điều kiện)
            other_parents = [p for p in parents if p != j]
            
            if len(other_parents) == 0:
                continue
            
            # Compute partial correlation / Tính tương quan riêng
            pcorr, p_value = _partial_corr(data[:, j], data[:, i], data[:, other_parents])
            
            if p_value > alpha:
                pruned[j, i] = 0
                if verbose:
                    print(f"Pruned edge {j}->{i}: partial_corr={pcorr:.3f}, p={p_value:.3f}")
    
    return pruned


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Compute partial correlation between x and y given z.
    Tính tương quan riêng giữa x và y khi biết z.
    
    Uses regression residuals method.
    Sử dụng phương pháp phần dư hồi quy.
    """
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    
    # Add intercept / Thêm hệ số chắn
    z_with_intercept = np.column_stack([np.ones(len(x)), z])
    
    try:
        # Residuals of x ~ z / Phần dư của x ~ z
        beta_x = np.linalg.lstsq(z_with_intercept, x, rcond=None)[0]
        residual_x = x - z_with_intercept @ beta_x
        
        # Residuals of y ~ z / Phần dư của y ~ z
        beta_y = np.linalg.lstsq(z_with_intercept, y, rcond=None)[0]
        residual_y = y - z_with_intercept @ beta_y
        
        # Correlation of residuals / Tương quan của phần dư
        corr, p_value = stats.pearsonr(residual_x, residual_y)
        
        return corr, p_value
    
    except Exception:
        return 0.0, 1.0


def iterative_pruning(
    data: np.ndarray,
    adjacency: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.05,
    max_iter: int = 10,
    verbose: bool = False,
) -> np.ndarray:
    """
    Iteratively prune edges until convergence.
    Cắt tỉa cạnh lặp đi lặp lại cho đến khi hội tụ.
    
    Args:
        data: Data matrix / Ma trận dữ liệu
        adjacency: Initial adjacency / Ma trận kề ban đầu
        threshold: Binarization threshold / Ngưỡng nhị phân hóa
        alpha: Significance level / Mức ý nghĩa
        max_iter: Maximum iterations / Số lần lặp tối đa
        verbose: Print progress / In tiến trình
        
    Returns:
        Pruned adjacency matrix / Ma trận kề đã cắt tỉa
    """
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(adjacency):
        adjacency = adjacency.numpy()
    
    adj = (adjacency > threshold).astype(float)
    
    for iteration in range(max_iter):
        n_edges_before = adj.sum()
        
        # Apply pruning / Áp dụng cắt tỉa
        adj = cam_pruning(data, adj, threshold=0.5, method='regression', alpha=alpha, verbose=verbose)
        
        n_edges_after = adj.sum()
        
        if verbose:
            print(f"Iteration {iteration + 1}: {int(n_edges_before)} -> {int(n_edges_after)} edges")
        
        if n_edges_after == n_edges_before:
            break
    
    return adj


def threshold_search(
    data: np.ndarray,
    adjacency: np.ndarray,
    true_adjacency: np.ndarray,
    thresholds: Optional[List[float]] = None,
    use_pruning: bool = True,
    alpha: float = 0.05,
) -> Dict[float, Dict]:
    """
    Search for optimal threshold with optional pruning.
    Tìm kiếm ngưỡng tối ưu với tùy chọn cắt tỉa.
    
    Args:
        data: Data matrix / Ma trận dữ liệu
        adjacency: Soft adjacency matrix / Ma trận kề mềm
        true_adjacency: Ground truth / Ground truth
        thresholds: List of thresholds to try / Danh sách các ngưỡng để thử
        use_pruning: Apply CAM pruning after thresholding / Áp dụng cắt tỉa CAM sau khi phân ngưỡng
        alpha: Significance level for pruning / Mức ý nghĩa cho cắt tỉa
        
    Returns:
        Dict mapping threshold to metrics / Dict ánh xạ ngưỡng đến các chỉ số
    """
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(adjacency):
        adjacency = adjacency.numpy()
    if torch.is_tensor(true_adjacency):
        true_adjacency = true_adjacency.numpy()
    
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = {}
    
    for thresh in thresholds:
        adj = (adjacency > thresh).astype(float)
        
        if use_pruning:
            adj = cam_pruning(data, adj, threshold=0.5, method='regression', alpha=alpha)
        
        # Compute metrics
        true_bin = (true_adjacency > 0.5).astype(float)
        
        tp = ((adj == 1) & (true_bin == 1)).sum()
        fp = ((adj == 1) & (true_bin == 0)).sum()
        fn = ((adj == 0) & (true_bin == 1)).sum()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        shd = fp + fn
        
        results[thresh] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'shd': int(shd),
            'n_edges': int(adj.sum()),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'adjacency': adj,
        }
    
    return results


def pns_selection(
    data: np.ndarray,
    num_neighbors: int = 5,
) -> np.ndarray:
    """
    Preliminary Neighborhood Selection (PNS) from GraN-DAG.
    Lựa chọn Vùng lân cận Sơ bộ (PNS) từ GraN-DAG.
    
    Pre-filters potential parents for each node based on correlations.
    Lọc trước các cha mẹ tiềm năng cho mỗi nút dựa trên tương quan.
    
    Args:
        data: Data matrix (n_samples, n_nodes) / Ma trận dữ liệu
        num_neighbors: Maximum number of neighbors per node / Số lượng hàng xóm tối đa mỗi nút
        
    Returns:
        Binary mask indicating potential edges / Mặt nạ nhị phân chỉ ra các cạnh tiềm năng
    """
    if torch.is_tensor(data):
        data = data.numpy()
    
    n = data.shape[1]
    
    # Compute correlation matrix / Tính ma trận tương quan
    corr_matrix = np.corrcoef(data.T)
    np.fill_diagonal(corr_matrix, 0)
    
    # For each node, keep top-k most correlated / Với mỗi nút, giữ lại top-k tương quan nhất
    mask = np.zeros((n, n))
    
    for i in range(n):
        corrs = np.abs(corr_matrix[i, :])
        top_k = np.argsort(corrs)[-num_neighbors:]
        mask[top_k, i] = 1  # These nodes could be parents of i / Những nút này có thể là cha mẹ của i
    
    # Remove diagonal / Loại bỏ đường chéo
    np.fill_diagonal(mask, 0)
    
    return mask
