"""
ADMG Support for CausalMLP / Hỗ trợ ADMG cho CausalMLP

Implements Acyclic Directed Mixed Graphs for handling latent confounders.
Triển khai đồ thị hỗn hợp có hướng không chu trình (ADMG) để xử lý các biến gây nhiễu ẩn.
Based on DECI's ADMG capabilities. / Dựa trên khả năng ADMG của DECI.

ADMGs extend DAGs with: / ADMG mở rộng DAG với:
- Directed edges (→): Direct causal effects / Các cạnh có hướng: Hiệu quả nhân quả trực tiếp
- Bidirected edges (↔): Latent confounders / Các cạnh hai chiều: Biến gây nhiễu ẩn

This allows modeling scenarios where unmeasured confounders affect multiple observed variables.
Điều này cho phép mô hình hóa các kịch bản nơi các biến gây nhiễu không đo lường được ảnh hưởng đến nhiều biến quan sát.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class ADMGAdjacency(nn.Module):
    """
    ADMG adjacency with both directed and bidirected edges.
    Ma trận kề ADMG với cả cạnh có hướng và hai chiều.
    
    Directed edges: Standard DAG structure / Cạnh có hướng: Cấu trúc DAG chuẩn
    Bidirected edges: Latent confounders (symmetric) / Cạnh hai chiều: Biến gây nhiễu ẩn (đối xứng)
    
    The model learns: / Mô hình học:
    - D[i,j]: Directed edge i → j (asymmetric) / Cạnh có hướng i → j (bất đối xứng)
    - B[i,j]: Bidirected edge i ↔ j (symmetric) / Cạnh hai chiều i ↔ j (đối xứng)
    """
    
    def __init__(
        self,
        num_nodes: int,
        init_directed_prob: float = 0.3,
        init_bidirected_prob: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.temperature = temperature
        
        # Directed edges (asymmetric)
        # Các cạnh có hướng (bất đối xứng)
        init_logit_d = np.log(init_directed_prob / (1 - init_directed_prob))
        self.directed_logits = nn.Parameter(
            init_logit_d * torch.ones(num_nodes, num_nodes)
        )
        
        # Bidirected edges (symmetric, only upper triangle)
        # Các cạnh hai chiều (đối xứng, chỉ tam giác trên)
        n_bidirected = num_nodes * (num_nodes - 1) // 2
        init_logit_b = np.log(init_bidirected_prob / (1 - init_bidirected_prob))
        self.bidirected_logits = nn.Parameter(
            init_logit_b * torch.ones(n_bidirected)
        )
        
        # Masks
        self.register_buffer('diag_mask', 1 - torch.eye(num_nodes))
        self._build_bidirected_indices()
    
    def _build_bidirected_indices(self):
        """
        Build index mapping for symmetric bidirected edges.
        Xây dựng ánh xạ chỉ mục cho các cạnh hai chiều đối xứng.
        """
        n = self.num_nodes
        row_idx = []
        col_idx = []
        
        for i in range(n):
            for j in range(i + 1, n):
                row_idx.append(i)
                col_idx.append(j)
        
        self.register_buffer('bi_row', torch.tensor(row_idx))
        self.register_buffer('bi_col', torch.tensor(col_idx))
    
    @property
    def directed_probs(self) -> torch.Tensor:
        """
        Get directed edge probabilities.
        Lấy xác suất cạnh có hướng.
        """
        return torch.sigmoid(self.directed_logits / self.temperature) * self.diag_mask
    
    @property
    def bidirected_probs(self) -> torch.Tensor:
        """
        Get bidirected edge probabilities (symmetric matrix).
        Lấy xác suất cạnh hai chiều (ma trận đối xứng).
        """
        n = self.num_nodes
        probs_flat = torch.sigmoid(self.bidirected_logits / self.temperature)
        
        # Build symmetric matrix
        B = torch.zeros(n, n, device=probs_flat.device)
        B[self.bi_row, self.bi_col] = probs_flat
        B[self.bi_col, self.bi_row] = probs_flat  # Symmetric
        
        return B
    
    def sample_directed(self, hard: bool = False) -> torch.Tensor:
        """
        Sample directed adjacency using Gumbel-Softmax.
        Lấy mẫu ma trận kề có hướng sử dụng Gumbel-Softmax.
        """
        u = torch.rand_like(self.directed_logits).clamp(1e-8, 1 - 1e-8)
        gumbel = -torch.log(-torch.log(u))
        
        soft = torch.sigmoid((self.directed_logits + gumbel) / self.temperature)
        
        if hard:
            hard_sample = (soft > 0.5).float()
            return (hard_sample - soft).detach() + soft
        
        return soft * self.diag_mask
    
    def sample_bidirected(self, hard: bool = False) -> torch.Tensor:
        """
        Sample bidirected adjacency.
        Lấy mẫu ma trận kề hai chiều.
        """
        u = torch.rand_like(self.bidirected_logits).clamp(1e-8, 1 - 1e-8)
        gumbel = -torch.log(-torch.log(u))
        
        soft = torch.sigmoid((self.bidirected_logits + gumbel) / self.temperature)
        
        if hard:
            hard_sample = (soft > 0.5).float()
            soft = (hard_sample - soft).detach() + soft
        
        # Build symmetric matrix / Xây dựng ma trận đối xứng
        n = self.num_nodes
        B = torch.zeros(n, n, device=soft.device)
        B[self.bi_row, self.bi_col] = soft
        B[self.bi_col, self.bi_row] = soft
        
        return B
    
    def forward(self, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both adjacency matrices.
        Lấy cả hai ma trận kề.
        
        Returns:
            (directed_adj, bidirected_adj)
        """
        if self.training:
            return self.sample_directed(hard), self.sample_bidirected(hard)
        return self.directed_probs, self.bidirected_probs


class ADMGConstraint(nn.Module):
    """
    ADMG acyclicity constraint.
    Ràng buộc không chu trình ADMG.
    
    For ADMGs, we need: / Đối với ADMG, chúng ta cần:
    1. Directed part must be acyclic: h(D) = 0 / Phần có hướng phải không chu trình: h(D) = 0
    2. "Bow-free" constraint / Ràng buộc "không cung":
       D[i,j] > 0 implies B[i,j] = 0 (No both directed and bidirected between same pair)
       D[i,j] > 0 ngụ ý B[i,j] = 0 (Không có cả hướng và hai chiều giữa cùng một cặp)
    """
    
    def __init__(self, bow_free_weight: float = 10.0):
        super().__init__()
        self.bow_free_weight = bow_free_weight
    
    def dag_constraint(self, D: torch.Tensor) -> torch.Tensor:
        """
        Standard NOTEARS DAG constraint on directed edges.
        Ràng buộc DAG chuẩn của NOTEARS trên các cạnh có hướng.
        """
        n = D.shape[0]
        A = D * D  # Element-wise square
        E = torch.linalg.matrix_exp(A)
        return torch.trace(E) - n
    
    def bow_free_constraint(self, D: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Bow-free constraint: Penalize simultaneous D[i,j] and B[i,j].
        Ràng buộc không cung: Phạt sự xuất hiện đồng thời của D[i,j] và B[i,j].
        """
        # Check both directions for directed / Kiểm tra cả hai hướng cho cạnh có hướng
        D_or = torch.max(D, D.T)
        
        # Penalize product / Phạt tích
        bow_violation = (D_or * B).sum()
        
        return bow_violation
    
    def forward(
        self,
        directed: torch.Tensor,
        bidirected: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ADMG constraints.
        Tính toán các ràng buộc ADMG.
        """
        h_dag = self.dag_constraint(directed)
        h_bow = self.bow_free_constraint(directed, bidirected)
        
        return {
            'h_dag': h_dag,
            'h_bow': h_bow,
            'h_total': h_dag + self.bow_free_weight * h_bow,
        }


class ADMGMLP(nn.Module):
    """
    MLP for ADMG that handles both direct effects and confounding.
    MLP cho ADMG xử lý cả hiệu ứng trực tiếp và biến gây nhiễu.
    
    For node i: / Đối với nút i:
    - Direct effects: from parents (nodes j where D[j,i] = 1)
      Hiệu ứng trực tiếp: từ cha mẹ (các nút j nơi D[j,i] = 1)
    - Confounded effects: from nodes sharing bidirected edge (B[i,j] = 1)
      Hiệu ứng gây nhiễu: từ các nút chia sẻ cạnh hai chiều (B[i,j] = 1)
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Main MLP for each node (direct effects)
        # MLP chính cho mỗi nút (hiệu ứng trực tiếp)
        self.direct_weights = nn.ParameterList()
        self.direct_biases = nn.ParameterList()
        
        dims = [num_nodes] + [hidden_dim] * num_layers + [2]  # Output: mean, log_std
        
        for layer in range(len(dims) - 1):
            weight = torch.empty(num_nodes, dims[layer + 1], dims[layer])
            nn.init.xavier_uniform_(weight.view(-1, dims[layer]))
            self.direct_weights.append(nn.Parameter(weight))
            self.direct_biases.append(nn.Parameter(torch.zeros(num_nodes, dims[layer + 1])))
        
        # Confounding correlation matrix (Cholesky factor)
        # Ma trận tương quan gây nhiễu (Thừa số Cholesky)
        # L such that Sigma = L @ L.T
        self.noise_cholesky = nn.Parameter(torch.eye(num_nodes) * 0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        directed: torch.Tensor,
        bidirected: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass. / Lan truyền tiến.
        
        Returns:
            (means, log_stds, noise_cov)
            (trung bình, log độ lệch chuẩn, hiệp phương sai nhiễu)
        """
        batch_size = x.shape[0]
        
        # Compute direct effects / Tính hiệu ứng trực tiếp
        h = directed.unsqueeze(0) * x.unsqueeze(1)  # (batch, num_nodes, num_nodes)
        
        for layer, (weight, bias) in enumerate(zip(self.direct_weights, self.direct_biases)):
            if layer == 0:
                h = torch.einsum('noh,bno->bnh', weight, h)
            else:
                h = torch.einsum('noh,bnh->bno', weight, h)
            h = h + bias.unsqueeze(0)
            
            if layer < len(self.direct_weights) - 1:
                h = F.leaky_relu(h, 0.2)
        
        means = h[:, :, 0]
        log_stds = h[:, :, 1]
        
        # Compute noise covariance from bidirected edges
        # Tính hiệp phương sai nhiễu từ các cạnh hai chiều
        # The bidirected structure induces correlated noise
        # Cấu trúc hai chiều gây ra nhiễu tương quan
        L = torch.tril(self.noise_cholesky)
        
        # Mask by bidirected structure (plus diagonal)
        # Che mặt nạ theo cấu trúc hai chiều (cộng đường chéo)
        mask = bidirected + torch.eye(self.num_nodes, device=bidirected.device)
        L_masked = L * mask
        
        # Covariance = L @ L.T
        noise_cov = L_masked @ L_masked.T
        
        return means, log_stds, noise_cov
    
    def log_prob(
        self,
        x: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        noise_cov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability with correlated noise.
        Tính log xác suất với nhiễu tương quan.
        
        Uses multivariate Gaussian with learned covariance.
        Sử dụng phân phối Gaussian đa biến với hiệp phương sai học được.
        """
        residuals = x - means  # (batch, num_nodes)
        
        if self.num_nodes <= 20:
            # Full multivariate normal / Gaussian đa biến đầy đủ
            diag_var = torch.exp(2 * log_stds)  # (batch, num_nodes)
            
            # Total covariance per sample / Tổng hiệp phương sai mỗi mẫu
            batch_size = x.shape[0]
            log_probs = []
            
            for b in range(min(batch_size, 100)):  # Limit for memory
                cov = torch.diag(diag_var[b]) + noise_cov * 0.1
                cov = cov + 1e-4 * torch.eye(self.num_nodes, device=cov.device)  # Stability/Sự ổn định
                
                try:
                    mvn = torch.distributions.MultivariateNormal(
                        means[b], covariance_matrix=cov
                    )
                    log_probs.append(mvn.log_prob(x[b]))
                except RuntimeError:
                    # Fallback to independent / Dự phòng về độc lập
                    log_p = -0.5 * (residuals[b] ** 2 / diag_var[b] + torch.log(diag_var[b]))
                    log_probs.append(log_p.sum())
            
            return torch.stack(log_probs)
        else:
            # Independent approximation for large graphs
            # Xấp xỉ độc lập cho đồ thị lớn
            var = torch.exp(2 * log_stds)
            log_prob = -0.5 * (residuals ** 2 / var + torch.log(var) + np.log(2 * np.pi))
            return log_prob.sum(dim=1)


class ADMGModel(nn.Module):
    """
    Full ADMG model for CausalMLP.
    Mô hình ADMG đầy đủ cho CausalMLP.
    
    Extends the basic DAG model to handle latent confounders.
    Mở rộng mô hình DAG cơ bản để xử lý các biến gây nhiễu ẩn.
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bow_free_weight: float = 10.0,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        self.adjacency = ADMGAdjacency(num_nodes)
        self.mlp = ADMGMLP(num_nodes, hidden_dim, num_layers)
        self.constraint = ADMGConstraint(bow_free_weight)
    
    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing loss. / Lan truyền tiến tính toán mất mát."""
        directed, bidirected = self.adjacency()
        
        means, log_stds, noise_cov = self.mlp(x, directed, bidirected)
        
        # Log likelihood
        log_lik = self.mlp.log_prob(x, means, log_stds, noise_cov)
        nll = -log_lik.mean()
        
        # Constraints / Các ràng buộc
        constraints = self.constraint(directed, bidirected)
        
        # Sparsity / Tính thưa thớt
        sparsity = directed.abs().sum() + bidirected.abs().sum()
        
        loss = nll + constraints['h_total'] + 0.001 * sparsity
        
        result = {'loss': loss}
        
        if return_components:
            result.update({
                'nll': nll,
                'h_dag': constraints['h_dag'],
                'h_bow': constraints['h_bow'],
                'directed': directed,
                'bidirected': bidirected,
            })
        
        return result
    
    def get_directed(self, threshold: float = 0.5) -> torch.Tensor:
        """Get directed adjacency. / Lấy ma trận kề có hướng."""
        with torch.no_grad():
            return (self.adjacency.directed_probs > threshold).float()
    
    def get_bidirected(self, threshold: float = 0.5) -> torch.Tensor:
        """Get bidirected adjacency. / Lấy ma trận kề hai chiều."""
        with torch.no_grad():
            return (self.adjacency.bidirected_probs > threshold).float()
