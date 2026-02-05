"""
Temporal/Time-Series Causal Discovery for CausalMLP / Khám phá nhân quả chuỗi thời gian cho CausalMLP

Implements time-lagged causal discovery for time series data.
Triển khai khám phá nhân quả có độ trễ thời gian cho dữ liệu chuỗi thời gian.
Based on concepts from PCMCI, Granger causality, and neural approaches.
Dựa trên các khái niệm từ PCMCI, nhân quả Granger và các phương pháp mạng nơ-ron.

Features / Tính năng:
- Multiple time lags / Đa độ trễ thời gian
- Contemporaneous and lagged effects / Hiệu ứng đồng thời và có độ trễ
- Summary and full-time graphs / Đồ thị tổng hợp và toàn thời gian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class TemporalAdjacency(nn.Module):
    """
    Temporal adjacency for time-lagged causal discovery.
    Ma trận kề thời gian cho khám phá nhân quả có độ trễ.
    
    Learns / Học:
    - A_0: Contemporaneous effects (same time step) / Hiệu ứng đồng thời (cùng bước thời gian)
    - A_1, A_2, ..., A_L: Lagged effects (past → present) / Hiệu ứng trễ (quá khứ → hiện tại)
    
    The full causal graph is a tensor of shape (num_lags+1, num_nodes, num_nodes)
    Đồ thị nhân quả đầy đủ là một tensor có kích thước (num_lags+1, num_nodes, num_nodes)
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_lags: int = 2,
        init_prob: float = 0.2,
        temperature: float = 1.0,
        allow_contemporaneous: bool = True,
    ):
        """
        Args:
            num_nodes: Number of nodes / Số lượng nút
            num_lags: Number of past time steps to consider / Số lượng bước thời gian quá khứ cần xem xét
            allow_contemporaneous: Allow edges at lag 0 / Cho phép cạnh tại độ trễ 0
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_lags = num_lags
        self.temperature = temperature
        self.allow_contemporaneous = allow_contemporaneous
        
        # Total matrices: 1 contemporaneous + num_lags
        # Tổng số ma trận: 1 đồng thời + num_lags
        num_matrices = 1 + num_lags if allow_contemporaneous else num_lags
        
        init_logit = np.log(init_prob / (1 - init_prob))
        self.logits = nn.Parameter(
            init_logit * torch.ones(num_matrices, num_nodes, num_nodes)
        )
        
        # Diagonal mask for contemporaneous (no self-loops at same time)
        # Mặt nạ đường chéo cho đồng thời (không có vòng lặp tự thân tại cùng thời điểm)
        self.register_buffer('diag_mask', 1 - torch.eye(num_nodes))
    
    @property
    def probs(self) -> torch.Tensor:
        """
        Get edge probabilities for all lags.
        Lấy xác suất cạnh cho tất cả các độ trễ.
        """
        probs = torch.sigmoid(self.logits / self.temperature)
        
        if self.allow_contemporaneous:
            # Apply diagonal mask only to contemporaneous (index 0)
            # Áp dụng mặt nạ đường chéo chỉ cho đồng thời (chỉ số 0)
            probs[0] = probs[0] * self.diag_mask
        
        return probs
    
    def get_contemporaneous(self) -> torch.Tensor:
        """
        Get contemporaneous adjacency A_0.
        Lấy ma trận kề đồng thời A_0.
        """
        if self.allow_contemporaneous:
            return self.probs[0]
        return torch.zeros(self.num_nodes, self.num_nodes, device=self.logits.device)
    
    def get_lagged(self, lag: int = 1) -> torch.Tensor:
        """
        Get lagged adjacency A_lag.
        Lấy ma trận kề có độ trễ A_lag.
        """
        if self.allow_contemporaneous:
            idx = lag
        else:
            idx = lag - 1
        
        if idx >= self.probs.shape[0] or idx < 0:
            return torch.zeros(self.num_nodes, self.num_nodes, device=self.logits.device)
        
        return self.probs[idx]
    
    def sample(self, hard: bool = False) -> torch.Tensor:
        """Sample all adjacency matrices. / Lấy mẫu tất cả các ma trận kề."""
        u = torch.rand_like(self.logits).clamp(1e-8, 1 - 1e-8)
        gumbel = -torch.log(-torch.log(u))
        
        soft = torch.sigmoid((self.logits + gumbel) / self.temperature)
        
        if self.allow_contemporaneous:
            soft[0] = soft[0] * self.diag_mask
        
        if hard:
            hard_sample = (soft > 0.5).float()
            return (hard_sample - soft).detach() + soft
        
        return soft
    
    def forward(self, hard: bool = False) -> torch.Tensor:
        """Get adjacency matrices. / Lấy các ma trận kề."""
        if self.training:
            return self.sample(hard)
        return self.probs
    
    def get_summary_graph(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get summary graph (any lag).
        Lấy đồ thị tổng hợp (bất kỳ độ trễ nào).
        
        Summary[i,j] = 1 if there exists any lag l such that A_l[i,j] > threshold
        Summary[i,j] = 1 nếu tồn tại bất kỳ độ trễ l nào sao cho A_l[i,j] > ngưỡng
        """
        all_adj = self.probs
        
        # Max over all lags / Max trên tất cả các độ trễ
        summary = (all_adj > threshold).float().max(dim=0)[0]
        
        return summary


class TemporalDAGConstraint(nn.Module):
    """
    DAG constraint for temporal graphs.
    Ràng buộc DAG cho đồ thị thời gian.
    
    For temporal graphs: / Đối với đồ thị thời gian:
    - Lagged effects (A_1, ..., A_L) don't need DAG constraint (past can't cause past)
      Hiệu ứng trễ không cần ràng buộc DAG (quá khứ không thể gây ra quá khứ)
    - Only contemporaneous (A_0) needs acyclicity constraint
      Chỉ có đồng thời (A_0) cần ràng buộc không chu trình
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, adjacencies: torch.Tensor) -> torch.Tensor:
        """
        Compute DAG constraint on contemporaneous part only.
        Tính toán ràng buộc DAG chỉ trên phần đồng thời.
        
        Args:
            adjacencies: Shape (num_lags+1, num_nodes, num_nodes)
            
        Returns:
            DAG constraint value / Giá trị ràng buộc DAG
        """
        # Only contemporaneous (first matrix) needs constraint
        # Chỉ phần đồng thời (ma trận đầu tiên) cần ràng buộc
        A = adjacencies[0]
        n = A.shape[0]
        
        # Standard NOTEARS
        A_sq = A * A
        E = torch.linalg.matrix_exp(A_sq)
        h = torch.trace(E) - n
        
        return h


class TemporalMLP(nn.Module):
    """
    MLP for temporal causal discovery.
    MLP cho khám phá nhân quả thời gian.
    
    Predicts X_t from: / Dự đoán X_t từ:
    - X_t (contemporaneous, masked by A_0) / X_t (đồng thời, được che bởi A_0)
    - X_{t-1}, X_{t-2}, ..., X_{t-L} (lagged, masked by A_1, ..., A_L) / X độ trễ
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_lags: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_lags = num_lags
        
        # Input dim: num_nodes * (1 + num_lags)
        input_dim = num_nodes * (1 + num_lags)
        
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
        
        layers.append(nn.Linear(hidden_dim, num_nodes * 2))  # Mean + log_std per node
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        x_history: torch.Tensor,  # (batch, num_lags+1, num_nodes)
        adjacencies: torch.Tensor,  # (num_lags+1, num_nodes, num_nodes)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. / Lan truyền tiến.
        
        Args:
            x_history: Time series history [X_t, X_{t-1}, ..., X_{t-L}]
                       Lịch sử chuỗi thời gian.
            adjacencies: Adjacency matrices [A_0, A_1, ..., A_L]
                         Các ma trận kề.
            
        Returns:
            (means, log_stds) for X_t
            (trung bình, log độ lệch chuẩn) cho X_t
        """
        batch_size = x_history.shape[0]
        
        # Apply adjacency masks / Áp dụng mặt nạ kề
        masked_inputs = []
        
        for lag in range(1 + self.num_lags):
            # x_history[:, lag] is X_{t-lag}
            # adjacencies[lag] is A_lag
            # A_lag[j, i] = 1 means X_{t-lag, j} causes X_{t, i}
            
            # For each node i, mask its inputs from lag
            # Với mỗi nút i, che đầu vào của nó từlag
            masked = adjacencies[lag].T.unsqueeze(0) * x_history[:, lag].unsqueeze(1)
            # Shape: (batch, num_nodes, num_nodes)
            # Sum over source nodes to get input per target node
            # Tổng qua các nút nguồn để lấy đầu vào cho mỗi nút đích
            masked_sum = masked.sum(dim=2)  # (batch, num_nodes)
            masked_inputs.append(masked_sum)
        
        # Concatenate all lagged inputs / Nối tất cả đầu vào trễ
        mlp_input = torch.cat(masked_inputs, dim=1)  # (batch, num_nodes * (num_lags+1))
        
        output = self.network(mlp_input)  # (batch, num_nodes * 2)
        
        means = output[:, :self.num_nodes]
        log_stds = output[:, self.num_nodes:]
        
        return means, log_stds


class TemporalCausalModel(nn.Module):
    """
    Full temporal causal discovery model.
    Mô hình khám phá nhân quả thời gian đầy đủ.
    
    Learns causal structure from time series data.
    Học cấu trúc nhân quả từ dữ liệu chuỗi thời gian.
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_lags: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        allow_contemporaneous: bool = True,
        sparsity_lambda: float = 0.001,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_lags = num_lags
        self.sparsity_lambda = sparsity_lambda
        
        self.adjacency = TemporalAdjacency(
            num_nodes, num_lags,
            allow_contemporaneous=allow_contemporaneous
        )
        self.mlp = TemporalMLP(num_nodes, num_lags, hidden_dim, num_layers)
        self.constraint = TemporalDAGConstraint()
        
        # Augmented Lagrangian parameters / Tham số Lagrangian tăng cường
        self.register_buffer('alpha', torch.tensor(0.0))
        self.register_buffer('rho', torch.tensor(0.01))
    
    def forward(
        self,
        x_history: torch.Tensor,  # (batch, num_lags+1, num_nodes)
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass. / Lan truyền tiến.
        
        Args:
            x_history: Time series windows [X_t, X_{t-1}, ..., X_{t-L}]
                       Các cửa sổ chuỗi thời gian.
        """
        adjacencies = self.adjacency()
        
        means, log_stds = self.mlp(x_history, adjacencies)
        
        # Target is X_t (first in sequence)
        # Mục tiêu là X_t (đầu tiên trong chuỗi)
        x_t = x_history[:, 0]
        
        # NLL
        var = torch.exp(2 * log_stds).clamp(0.01, 4.0)
        nll = 0.5 * ((x_t - means) ** 2 / var + torch.log(var)).sum(dim=1).mean()
        
        # DAG constraint / Ràng buộc DAG
        h = self.constraint(adjacencies)
        dag_penalty = self.alpha * h + 0.5 * self.rho * h ** 2
        
        # Sparsity / Tính thưa thớt
        sparsity = adjacencies.abs().sum()
        
        loss = nll + dag_penalty + self.sparsity_lambda * sparsity
        
        result = {'loss': loss}
        
        if return_components:
            result.update({
                'nll': nll,
                'h': h,
                'sparsity': sparsity,
                'adjacencies': adjacencies,
            })
        
        return result
    
    def update_auglag(self, h: float):
        """Update augmented Lagrangian parameters. / Cập nhật tham số Lagrangian tăng cường."""
        self.alpha = self.alpha + self.rho * h
        if h > 0.25 * h:  # Not improving enough / Không cải thiện đủ
            self.rho = torch.clamp(self.rho * 2, max=1e4)
    
    def get_causal_graph(
        self,
        threshold: float = 0.5,
        lag: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get learned causal graph.
        Lấy đồ thị nhân quả đã học.
        
        Args:
            threshold: Binarization threshold / Ngưỡng nhị phân hóa
            lag: Specific lag (None for all) / Độ trễ cụ thể (None cho tất cả)
            
        Returns:
            Adjacency matrix or tensor of matrices
            Ma trận kề hoặc tensor các ma trận
        """
        with torch.no_grad():
            if lag is not None:
                if lag == 0:
                    return (self.adjacency.get_contemporaneous() > threshold).float()
                else:
                    return (self.adjacency.get_lagged(lag) > threshold).float()
            else:
                return (self.adjacency.probs > threshold).float()


def create_temporal_windows(
    time_series: torch.Tensor,  # (time_steps, num_nodes)
    num_lags: int = 2,
) -> torch.Tensor:
    """
    Create sliding windows from time series.
    Tạo các cửa sổ trượt từ chuỗi thời gian.
    
    Args:
        time_series: Shape (T, num_nodes)
        num_lags: Number of lags / Số lượng độ trễ
        
    Returns:
        Windows of shape (T - num_lags, num_lags + 1, num_nodes)
        Each window is [X_t, X_{t-1}, ..., X_{t-L}]
    """
    T = time_series.shape[0]
    windows = []
    
    for t in range(num_lags, T):
        window = torch.stack([time_series[t - l] for l in range(num_lags + 1)])
        windows.append(window)
    
    return torch.stack(windows)


class GrangerCausalityTest:
    """
    Classical Granger causality test for comparison.
    Kiểm định nhân quả Granger cổ điển để so sánh.
    
    Tests whether past values of X help predict Y beyond Y's own past.
    Kiểm tra xem giá trị quá khứ của X có giúp dự đoán Y vượt ra ngoài quá khứ của chính Y không.
    """
    
    def __init__(self, max_lag: int = 5, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha = alpha
    
    def test(
        self,
        time_series: np.ndarray,  # (T, num_nodes)
    ) -> np.ndarray:
        """
        Run pairwise Granger causality tests.
        Chạy kiểm định nhân quả Granger theo cặp.
        
        Returns:
            Adjacency matrix where [i,j] = 1 if i Granger-causes j
            Ma trận kề nơi [i,j] = 1 nếu i gây ra j theo Granger
        """
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("scipy required for Granger causality test")
        
        T, n = time_series.shape
        adjacency = np.zeros((n, n))
        
        for j in range(n):  # Target / Đích
            for i in range(n):  # Source / Nguồn
                if i == j:
                    continue
                
                # Restricted model: Y ~ Y_past / Mô hình hạn chế
                y = time_series[self.max_lag:, j]
                X_restricted = np.column_stack([
                    time_series[self.max_lag - l:-l if l > 0 else T, j]
                    for l in range(1, self.max_lag + 1)
                ])
                
                # Unrestricted model: Y ~ Y_past + X_past / Mô hình không hạn chế
                X_unrestricted = np.column_stack([
                    X_restricted,
                    *[time_series[self.max_lag - l:-l if l > 0 else T, i:i+1]
                      for l in range(1, self.max_lag + 1)]
                ])
                
                try:
                    # OLS
                    beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                    rss_r = np.sum((y - X_restricted @ beta_r) ** 2)
                    
                    beta_u = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
                    rss_u = np.sum((y - X_unrestricted @ beta_u) ** 2)
                    
                    # F-test
                    df1 = self.max_lag
                    df2 = len(y) - 2 * self.max_lag - 1
                    
                    if df2 > 0 and rss_u > 0:
                        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                        
                        if p_value < self.alpha:
                            adjacency[i, j] = 1
                except:
                    pass
        
        return adjacency
