"""
Piecewise Linear Networks for CausalMLP / Mạng tuyến tính từng đoạn cho CausalMLP

Alternative to MLP using piecewise linear functions.
Phương án thay thế cho MLP sử dụng các hàm tuyến tính từng đoạn.
From GraN-DAG paper.
Từ báo cáo GraN-DAG.

Benefits / Lợi ích:
- Simpler than full MLP / Đơn giản hơn MLP đầy đủ
- Easier to interpret / Dễ diễn giải hơn
- Fewer parameters / Ít tham số hơn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PiecewiseLinear(nn.Module):
    """
    Piecewise linear function with learnable breakpoints.
    Hàm tuyến tính từng đoạn với các điểm ngắt có thể học được.
    
    f(x) = sum_k w_k * max(0, x - b_k)
    
    This is essentially a sum of ReLU functions with different breakpoints.
    Đây về cơ bản là tổng của các hàm ReLU với các điểm ngắt khác nhau.
    """
    
    def __init__(
        self,
        num_pieces: int = 5,
        init_range: float = 2.0,
    ):
        super().__init__()
        
        self.num_pieces = num_pieces
        
        # Breakpoints (uniformly initialized)
        # Điểm ngắt (khởi tạo đồng đều)
        breakpoints = torch.linspace(-init_range, init_range, num_pieces)
        self.breakpoints = nn.Parameter(breakpoints)
        
        # Weights for each piece / Trọng số cho mỗi đoạn
        self.weights = nn.Parameter(torch.randn(num_pieces) * 0.1)
        
        # Bias / Độ lệch
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply piecewise linear function.
        Áp dụng hàm tuyến tính từng đoạn.
        
        Args:
            x: Input (any shape) / Đầu vào (bất kỳ hình dạng nào)
            
        Returns:
            Output (same shape) / Đầu ra (cùng hình dạng)
        """
        # Expand for broadcasting / Mở rộng để phát sóng
        x_expanded = x.unsqueeze(-1)  # (..., 1)
        breakpoints = self.breakpoints.view(1, -1)  # (1, num_pieces)
        
        # Compute ReLU at each breakpoint / Tính ReLU tại mỗi điểm ngắt
        activated = F.relu(x_expanded - breakpoints)  # (..., num_pieces)
        
        # Weighted sum / Tổng có trọng số
        output = (activated * self.weights).sum(dim=-1)  # (...)
        
        return output + self.bias


class PiecewiseLinearMLP(nn.Module):
    """
    MLP using piecewise linear functions instead of standard activations.
    MLP sử dụng hàm tuyến tính từng đoạn thay vì hàm kích hoạt tiêu chuẩn.
    
    Each node has a piecewise linear function for each input.
    Mỗi nút có một hàm tuyến tính từng đoạn cho mỗi đầu vào.
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_pieces: int = 5,
        hidden_dim: int = 32,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_pieces = num_pieces
        
        # Piecewise linear for each (input, output) pair
        # Tuyến tính từng đoạn cho mỗi cặp (đầu vào, đầu ra)
        # pw_functions[i][j] = piecewise linear from node j to node i
        self.pw_functions = nn.ModuleList([
            nn.ModuleList([
                PiecewiseLinear(num_pieces) for _ in range(num_nodes)
            ]) for _ in range(num_nodes)
        ])
        
        # Output layer for mean and log_std
        # Lớp đầu ra cho trung bình và log độ lệch chuẩn
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            ) for _ in range(num_nodes)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. / Lan truyền tiến.
        
        Args:
            x: Input (batch, num_nodes) / Đầu vào
            adjacency: Adjacency matrix (num_nodes, num_nodes) / Ma trận kề
            
        Returns:
            (means, log_stds)
        """
        batch_size = x.shape[0]
        
        means = []
        log_stds = []
        
        for i in range(self.num_nodes):
            # Sum piecewise linear contributions from parents
            # Tổng các đóng góp tuyến tính từng đoạn từ cha mẹ
            contribution = torch.zeros(batch_size, device=x.device)
            
            for j in range(self.num_nodes):
                if i == j:
                    continue
                
                # Weight by adjacency / Trọng số hóa bởi ma trận kề
                weight = adjacency[j, i]
                if weight > 0.01:  # Skip near-zero edges / Bỏ qua các cạnh gần bằng 0
                    pw_output = self.pw_functions[i][j](x[:, j])
                    contribution = contribution + weight * pw_output
            
            # Output layer / Lớp đầu ra
            out = self.output_layers[i](contribution.unsqueeze(-1))
            means.append(out[:, 0])
            log_stds.append(out[:, 1])
        
        return torch.stack(means, dim=1), torch.stack(log_stds, dim=1)


class PathWeightExtractor:
    """
    Extract interpretable path weights from model.
    Trích xuất trọng số đường dẫn dễ diễn giải từ mô hình.
    
    Based on GraN-DAG's path weight normalization.
    Dựa trên chuẩn hóa trọng số đường dẫn của GraN-DAG.
    """
    
    def __init__(self, model):
        self.model = model
    
    def compute_path_weights(
        self,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute path-based edge weights.
        Tính trọng số cạnh dựa trên đường dẫn.
        
        The weight of edge j → i is computed from the product of weights along the path.
        Trọng số của cạnh j → i được tính từ tích các trọng số dọc theo đường dẫn.
        
        For MLP: W_ji ∝ |W^(1)_i * W^(0)_ij|
        
        Returns:
            Path weight matrix (num_nodes, num_nodes)
            Ma trận trọng số đường dẫn
        """
        if adjacency is None:
            adjacency = self.model.adjacency.probs.detach()
        
        n = adjacency.shape[0]
        
        if hasattr(self.model, 'mlp') and hasattr(self.model.mlp, 'weights'):
            weights = self.model.mlp.weights
            
            if len(weights) >= 1:
                # First layer weights: (num_nodes, hidden_dim, num_nodes)
                # Trọng số lớp đầu tiên
                W0 = weights[0].detach().abs()
                
                # Sum over hidden dimension / Tổng qua chiều ẩn
                path_weights = W0.sum(dim=1)  # (num_nodes, num_nodes)
                
                # Apply adjacency mask / Áp dụng mặt nạ kề
                path_weights = path_weights * adjacency
                
                # Normalize / Chuẩn hóa
                max_val = path_weights.max()
                if max_val > 0:
                    path_weights = path_weights / max_val
                
                return path_weights
        
        # Fallback: use adjacency directly / Dự phòng: dùng trực tiếp ma trận kề
        return adjacency
    
    def normalized_edge_weights(
        self,
        method: str = 'softmax',
    ) -> torch.Tensor:
        """
        Get normalized edge weights.
        Lấy trọng số cạnh đã chuẩn hóa.
        
        Args:
            method: 'softmax', 'l1', or 'l2'
            
        Returns:
            Normalized path weights / Trọng số đường dẫn đã chuẩn hóa
        """
        raw = self.compute_path_weights()
        
        if method == 'softmax':
            # Softmax per target node / Softmax cho mỗi nút đích
            return F.softmax(raw, dim=0)
        elif method == 'l1':
            # L1 normalization per target / Chuẩn hóa L1 mỗi đích
            norm = raw.sum(dim=0, keepdim=True) + 1e-8
            return raw / norm
        elif method == 'l2':
            # L2 normalization per target / Chuẩn hóa L2 mỗi đích
            norm = (raw ** 2).sum(dim=0, keepdim=True).sqrt() + 1e-8
            return raw / norm
        else:
            return raw
    
    def edge_importance_ranking(self) -> list:
        """
        Rank edges by importance.
        Xếp hạng các cạnh theo độ quan trọng.
        
        Returns:
            List of (source, target, weight) sorted by weight
            Danh sách (nguồn, đích, trọng số) được sắp xếp theo trọng số
        """
        weights = self.compute_path_weights().cpu().numpy()
        n = weights.shape[0]
        
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and weights[i, j] > 0.01:
                    edges.append((i, j, weights[i, j]))
        
        edges.sort(key=lambda x: -x[2])
        return edges


class LeakyIntegrate(nn.Module):
    """
    Leaky integration for signal processing in causal models.
    Tích hợp rò rỉ cho xử lý tín hiệu trong mô hình nhân quả.
    
    Models temporal dynamics in a simple way.
    Mô hình hóa động lực học thời gian theo cách đơn giản.
    """
    
    def __init__(
        self,
        num_nodes: int,
        leak_rate: float = 0.1,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.leak_rate = nn.Parameter(torch.full((num_nodes,), leak_rate))
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leaky integration step.
        Bước tích hợp rò rỉ.
        
        state_new = (1 - leak) * state + leak * x
        """
        if state is None:
            state = torch.zeros_like(x)
        
        leak = torch.sigmoid(self.leak_rate)
        new_state = (1 - leak) * state + leak * x
        
        return new_state, new_state


def create_alternative_mlp(
    num_nodes: int,
    mlp_type: str = 'standard',
    **kwargs,
) -> nn.Module:
    """
    Factory for different MLP types.
    Hàm nhà máy cho các loại MLP khác nhau.
    
    Args:
        num_nodes: Number of nodes / Số lượng nút
        mlp_type: 'standard', 'piecewise', or 'efficient'
    """
    if mlp_type == 'piecewise':
        return PiecewiseLinearMLP(num_nodes, **kwargs)
    else:
        # Import standard MLP from core / Import MLP tiêu chuẩn từ core
        from core.mlp import EfficientCausalMLP
        return EfficientCausalMLP(num_nodes, **kwargs)
