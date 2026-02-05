"""
Missing Values Handling for CausalMLP / Xử lý giá trị bị thiếu cho CausalMLP

Implements strategies for learning with missing data:
Triển khai các chiến lược học tập với dữ liệu bị thiếu:
- Mask-based training (ignore missing) / Huấn luyện dựa trên mặt nạ (bỏ qua phần thiếu)
- Imputation during training / Nội suy trong lúc huấn luyện
- MCAR, MAR, MNAR handling / Xử lý MCAR, MAR, MNAR

Based on DECI's partial observation support.
Dựa trên hỗ trợ quan sát một phần của DECI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class MissingValueHandler:
    """
    Handles missing values in data.
    Xử lý các giá trị bị thiếu trong dữ liệu.
    
    Strategies / Các chiến lược:
    - mask: Create observation mask, compute loss only on observed / Tạo mặt nạ quan sát, chỉ tính mất mát trên phần được quan sát
    - mean: Impute with column means / Nội suy bằng trung bình cột
    - zero: Impute with zeros / Nội suy bằng số 0
    - learned: Learn imputation values / Học các giá trị nội suy
    """
    
    def __init__(
        self,
        strategy: str = 'mask',
        missing_indicator: float = float('nan'),
    ):
        self.strategy = strategy
        self.missing_indicator = missing_indicator
        self._column_means = None
        self._column_stds = None
        self._learned_values = None
    
    def fit(self, data: torch.Tensor):
        """
        Compute statistics from data with missing values.
        Tính toán thống kê từ dữ liệu với các giá trị bị thiếu.
        
        Args:
            data: Data tensor (may contain NaN) / Tensor dữ liệu (có thể chứa NaN)
        """
        if torch.is_tensor(data):
            data_np = data.numpy() if not data.requires_grad else data.detach().numpy()
        else:
            data_np = data
        
        # Compute column-wise statistics ignoring NaN / Tính thống kê theo cột bỏ qua NaN
        self._column_means = np.nanmean(data_np, axis=0)
        self._column_stds = np.nanstd(data_np, axis=0)
        self._column_stds[self._column_stds < 1e-8] = 1.0
        
        return self
    
    def get_mask(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get observation mask.
        Lấy mặt nạ quan sát.
        
        Returns:
            Binary mask where 1 = observed, 0 = missing
            Mặt nạ nhị phân trong đó 1 = được quan sát, 0 = bị thiếu
        """
        if torch.isnan(data).any():
            mask = ~torch.isnan(data)
        else:
            mask = torch.ones_like(data, dtype=torch.bool)
        
        return mask.float()
    
    def impute(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Impute missing values.
        Nội suy các giá trị bị thiếu.
        
        Args:
            data: Data with missing values / Dữ liệu có giá trị thiếu
            
        Returns:
            (imputed_data, mask) / (dữ liệu đã nội suy, mặt nạ)
        """
        mask = self.get_mask(data)
        imputed = data.clone()
        
        if self.strategy == 'zero':
            imputed = torch.where(mask.bool(), data, torch.zeros_like(data))
            
        elif self.strategy == 'mean':
            if self._column_means is None:
                self.fit(data)
            
            means = torch.tensor(self._column_means, dtype=data.dtype, device=data.device)
            imputed = torch.where(mask.bool(), data, means.unsqueeze(0).expand_as(data))
            
        elif self.strategy == 'mask':
            # Replace NaN with 0 for computation, use mask for loss
            # Thay thế NaN bằng 0 để tính toán, sử dụng mặt nạ cho mất mát
            imputed = torch.where(mask.bool(), data, torch.zeros_like(data))
            
        elif self.strategy == 'random':
            # Random normal based on observed statistics
            # Ngẫu nhiên chuẩn dựa trên thống kê quan sát được
            if self._column_means is None:
                self.fit(data)
            
            means = torch.tensor(self._column_means, dtype=data.dtype, device=data.device)
            stds = torch.tensor(self._column_stds, dtype=data.dtype, device=data.device)
            random_values = means + stds * torch.randn_like(data)
            imputed = torch.where(mask.bool(), data, random_values)
        
        return imputed, mask
    
    def compute_masked_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss only on observed values.
        Tính toán tổn thất chỉ trên các giá trị được quan sát.
        
        Args:
            predicted: Model predictions / Dự đoán của mô hình
            target: True values (with NaN for missing) / Giá trị thực (với NaN cho phần thiếu)
            mask: Observation mask / Mặt nạ quan sát
            
        Returns:
            Mean loss over observed values / Tổn thất trung bình trên các giá trị được quan sát
        """
        # Replace NaN with 0 in target for computation / Thay thế NaN bằng 0 trong đích để tính toán
        target_clean = torch.where(mask.bool(), target, torch.zeros_like(target))
        
        # Squared error / Sai số bình phương
        se = (predicted - target_clean) ** 2
        
        # Mask and average / Áp dụng mặt nạ và tính trung bình
        masked_se = se * mask
        
        # Avoid division by zero / Tránh chia cho 0
        n_observed = mask.sum()
        if n_observed > 0:
            return masked_se.sum() / n_observed
        else:
            return torch.tensor(0.0, device=predicted.device)


class MissingAwareMLP(nn.Module):
    """
    MLP that handles missing values.
    MLP xử lý các giá trị bị thiếu.
    
    Uses mask as additional input to inform model about missingness.
    Sử dụng mặt nạ như đầu vào bổ sung để thông báo cho mô hình về sự thiếu vắng dữ liệu.
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_mask_input: bool = True,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.use_mask_input = use_mask_input
        self.handler = MissingValueHandler(strategy='mean')
        
        # Input includes mask if use_mask_input / Đầu vào bao gồm mặt nạ nếu use_mask_input
        input_dim = num_nodes * 2 if use_mask_input else num_nodes
        
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [num_nodes * 2]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.LeakyReLU(0.2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with missing value handling.
        Lan truyền tiến với xử lý giá trị bị thiếu.
        
        Args:
            x: Input data (may contain NaN) / Dữ liệu đầu vào (có thể chứa NaN)
            adjacency: Adjacency matrix / Ma trận kề
            
        Returns:
            (means, log_stds, mask)
        """
        # Handle missing values / Xử lý giá trị bị thiếu
        x_imputed, mask = self.handler.impute(x)
        
        # Mask input by adjacency / Che đầu vào bằng ma trận kề
        masked_input = adjacency.T @ x_imputed.T  # (num_nodes, batch)
        masked_input = masked_input.T  # (batch, num_nodes)
        
        if self.use_mask_input:
            # Concatenate imputed data with mask / Nối dữ liệu đã nội suy với mặt nạ
            mlp_input = torch.cat([masked_input, mask], dim=1)
        else:
            mlp_input = masked_input
        
        output = self.network(mlp_input)
        
        means = output[:, :self.num_nodes]
        log_stds = output[:, self.num_nodes:]
        
        return means, log_stds, mask


class MissingAwareModel(nn.Module):
    """
    Full model with missing value support.
    Mô hình đầy đủ hỗ trợ giá trị bị thiếu.
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Adjacency / Ma trận kề
        self.adjacency_logits = nn.Parameter(-2.0 * torch.ones(num_nodes, num_nodes))
        self.register_buffer('diag_mask', 1 - torch.eye(num_nodes))
        
        # MLP with missing support / MLP hỗ trợ dữ liệu thiếu
        self.mlp = MissingAwareMLP(num_nodes, hidden_dim, num_layers)
    
    @property
    def adjacency(self) -> torch.Tensor:
        return torch.sigmoid(self.adjacency_logits) * self.diag_mask
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing loss with missing value handling.
        Lan truyền tiến tính toán tổn thất với xử lý giá trị bị thiếu.
        """
        means, log_stds, mask = self.mlp(x, self.adjacency)
        
        # Compute masked NLL
        var = torch.exp(2 * log_stds).clamp(0.01, 4.0)
        
        # Target (replace NaN for computation)
        target = torch.where(mask.bool(), x, means.detach())
        
        nll_per_sample = 0.5 * ((target - means) ** 2 / var + torch.log(var))
        nll_masked = (nll_per_sample * mask).sum() / (mask.sum() + 1e-8)
        
        # DAG constraint
        A = self.adjacency
        h = torch.trace(torch.linalg.matrix_exp(A * A)) - self.num_nodes
        
        # Sparsity / Tính thưa thớt
        sparsity = A.abs().sum()
        
        loss = nll_masked + 10 * h + 0.001 * sparsity
        
        return {
            'loss': loss,
            'nll': nll_masked,
            'h': h,
            'mask': mask,
            'means': means,
        }
    
    def impute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Impute missing values using learned model.
        Nội suy các giá trị bị thiếu sử dụng mô hình đã học.
        
        Uses iterative imputation:
        Sử dụng nội suy lặp:
        1. Initialize missing with mean / Khởi tạo giá trị thiếu bằng trung bình
        2. Predict using model / Dự đoán bằng mô hình
        3. Replace missing with predictions / Thay thế giá trị thiếu bằng dự đoán
        4. Repeat / Lặp lại
        """
        with torch.no_grad():
            # Get initial imputation / Lấy nội suy ban đầu
            x_imputed, mask = self.mlp.handler.impute(x)
            
            # Iterative refinement / Tinh chỉnh lặp lại
            for _ in range(5):
                means, _, _ = self.mlp(x_imputed, self.adjacency)
                
                # Replace missing with predictions / Thay thế giá trị thiếu bằng dự đoán
                x_imputed = torch.where(mask.bool(), x_imputed, means)
            
            return x_imputed


def create_missing_data(
    data: torch.Tensor,
    missing_rate: float = 0.1,
    mechanism: str = 'mcar',
) -> torch.Tensor:
    """
    Create data with missing values for testing.
    Tạo dữ liệu với các giá trị bị thiếu để kiểm thử.
    
    Args:
        data: Complete data / Dữ liệu đầy đủ
        missing_rate: Proportion to make missing / Tỷ lệ làm thiếu
        mechanism: 'mcar' (random), 'mar', or 'mnar' / Cơ chế: 'mcar' (ngẫu nhiên), 'mar', hoặc 'mnar'
        
    Returns:
        Data with NaN for missing values
        Dữ liệu với NaN cho các giá trị bị thiếu
    """
    data_missing = data.clone()
    
    if mechanism == 'mcar':
        # Missing Completely At Random / Thiếu Hoàn toàn Ngẫu nhiên
        mask = torch.rand_like(data) < missing_rate
        data_missing[mask] = float('nan')
        
    elif mechanism == 'mar':
        # Missing At Random (depends on observed data) / Thiếu Ngẫu nhiên (phụ thuộc dữ liệu quan sát được)
        # Higher values more likely to be missing / Giá trị cao hơn có khả năng bị thiếu cao hơn
        probs = (data - data.min()) / (data.max() - data.min() + 1e-8)
        probs = probs * missing_rate * 2  # Scale to get target rate / Tỷ lệ để đạt mục tiêu
        probs = probs.clamp(0, 0.5)
        mask = torch.rand_like(data) < probs
        data_missing[mask] = float('nan')
        
    elif mechanism == 'mnar':
        # Missing Not At Random (missing depends on missing value itself)
        # Thiếu Không Ngẫu nhiên (thiếu phụ thuộc vào chính giá trị bị thiếu)
        # Extreme values more likely missing / Các giá trị cực đoan có khả năng bị thiếu cao hơn
        z = (data - data.mean(0)) / (data.std(0) + 1e-8)
        probs = (z.abs() * missing_rate).clamp(0, 0.5)
        mask = torch.rand_like(data) < probs
        data_missing[mask] = float('nan')
    
    return data_missing


class ExpectationMaximization:
    """
    EM algorithm for missing data.
    Thuật toán EM cho dữ liệu bị thiếu.
    
    Alternates between / Luân phiên giữa:
    - E-step: Impute missing values given current parameters / Bước E: Nội suy giá trị thiếu dựa trên tham số hiện tại
    - M-step: Update model parameters given imputed data / Bước M: Cập nhật tham số mô hình dựa trên dữ liệu đã nội suy
    """
    
    def __init__(
        self,
        model: MissingAwareModel,
        lr: float = 0.003,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def fit(
        self,
        data: torch.Tensor,
        n_epochs: int = 1000,
        n_em_steps: int = 5,
        verbose: bool = True,
    ):
        """
        Train with EM algorithm.
        Huấn luyện với thuật toán EM.
        """
        # Initialize imputation / Khởi tạo nội suy
        self.model.mlp.handler.fit(data)
        
        for em_step in range(n_em_steps):
            if verbose:
                print(f"\nEM Step {em_step + 1}/{n_em_steps}")
            
            # E-step: Impute missing values / Bước E: Nội suy giá trị thiếu
            with torch.no_grad():
                data_imputed = self.model.impute(data)
            
            # M-step: Train model / Bước M: Huấn luyện mô hình
            epochs_per_step = n_epochs // n_em_steps
            
            for epoch in range(epochs_per_step):
                self.optimizer.zero_grad()
                
                # Use original data (with NaN) for training / Dùng dữ liệu gốc (với NaN) để huấn luyện
                # Model handles missing internally / Mô hình tự xử lý dữ liệu thiếu bên trong
                result = self.model(data)
                result['loss'].backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                if verbose and (epoch + 1) % 100 == 0:
                    mask = result['mask']
                    obs_rate = mask.mean().item()
                    print(f"  Epoch {epoch + 1}: loss={result['loss'].item():.3f}, "
                          f"h={result['h'].item():.3f}, obs_rate={obs_rate:.2%}")
