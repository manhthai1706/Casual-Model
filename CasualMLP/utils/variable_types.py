"""
Mixed Variable Types for CausalMLP / Các loại biến hỗn hợp cho CausalMLP

Supports / Hỗ trợ:
- Continuous variables (Gaussian) / Biến liên tục (Gaussian)
- Categorical variables (Softmax/Gumbel) / Biến phân loại (Softmax/Gumbel)
- Binary variables (Bernoulli) / Biến nhị phân (Bernoulli)

Based on DECI's variable type handling.
Dựa trên xử lý loại biến của DECI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class VariableType(Enum):
    """Types of variables. / Các loại biến."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"


@dataclass
class VariableSpec:
    """Specification for a single variable. / Đặc tả cho một biến đơn lẻ."""
    name: str
    var_type: VariableType
    dim: int = 1  # Dimension (>1 for categorical) / Chiều (>1 cho biến phân loại)
    categories: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.var_type == VariableType.CATEGORICAL and self.dim < 2:
            raise ValueError("Categorical variables must have dim >= 2")
        if self.var_type == VariableType.BINARY:
            self.dim = 1


class VariableHandler:
    """
    Handles encoding/decoding of mixed variable types.
    Xử lý mã hóa/giải mã các loại biến hỗn hợp.
    
    Converts between / Chuyển đổi giữa:
    - Original: Each variable is 1D (categorical as integer) / Gốc: Mỗi biến là 1D (biến phân loại dưới dạng số nguyên)
    - Encoded: Categorical expanded to one-hot / Mã hóa: Biến phân loại được mở rộng thành one-hot
    """
    
    def __init__(self, specs: List[VariableSpec]):
        self.specs = specs
        self.num_vars = len(specs)
        
        # Calculate dimensions / Tính toán các chiều
        self.original_dims = [1] * self.num_vars
        self.encoded_dims = [spec.dim for spec in specs]
        
        self._build_indices()
    
    def _build_indices(self):
        """Build index mappings. / Xây dựng ánh xạ chỉ mục."""
        self.original_idx = []
        self.encoded_idx = []
        
        orig_pos = 0
        enc_pos = 0
        
        for spec in self.specs:
            self.original_idx.append((orig_pos, orig_pos + 1))
            self.encoded_idx.append((enc_pos, enc_pos + spec.dim))
            
            orig_pos += 1
            enc_pos += spec.dim
        
        self.original_dim = orig_pos
        self.encoded_dim = enc_pos
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode from original to expanded format.
        Mã hóa từ định dạng gốc sang định dạng mở rộng.
        
        Categorical: integer -> one-hot / Phân loại: số nguyên -> one-hot
        Binary/Continuous: unchanged / Nhị phân/Liên tục: không thay đổi
        """
        batch_size = x.shape[0]
        encoded = torch.zeros(batch_size, self.encoded_dim, device=x.device)
        
        for i, spec in enumerate(self.specs):
            orig_start, orig_end = self.original_idx[i]
            enc_start, enc_end = self.encoded_idx[i]
            
            if spec.var_type == VariableType.CATEGORICAL:
                indices = x[:, orig_start].long()
                one_hot = F.one_hot(indices, num_classes=spec.dim).float()
                encoded[:, enc_start:enc_end] = one_hot
            else:
                encoded[:, enc_start:enc_end] = x[:, orig_start:orig_end]
        
        return encoded
    
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode from expanded to original format.
        Giải mã từ định dạng mở rộng về định dạng gốc.
        
        Categorical: argmax -> integer / Phân loại: argmax -> số nguyên
        Binary: threshold at 0.5 / Nhị phân: ngưỡng tại 0.5
        """
        batch_size = encoded.shape[0]
        decoded = torch.zeros(batch_size, self.original_dim, device=encoded.device)
        
        for i, spec in enumerate(self.specs):
            orig_start, orig_end = self.original_idx[i]
            enc_start, enc_end = self.encoded_idx[i]
            
            if spec.var_type == VariableType.CATEGORICAL:
                logits = encoded[:, enc_start:enc_end]
                decoded[:, orig_start] = logits.argmax(dim=1).float()
            elif spec.var_type == VariableType.BINARY:
                prob = torch.sigmoid(encoded[:, enc_start])
                decoded[:, orig_start] = (prob > 0.5).float()
            else:
                decoded[:, orig_start:orig_end] = encoded[:, enc_start:enc_end]
        
        return decoded


class MixedLikelihood(nn.Module):
    """
    Likelihood for mixed variable types.
    Hàm hợp lý cho các loại biến hỗn hợp.
    
    Combines / Kết hợp:
    - Gaussian for continuous / Gaussian cho biến liên tục
    - Categorical cross-entropy for multi-class / Cross-entropy phân loại cho đa lớp
    - Binary cross-entropy for binary / Cross-entropy nhị phân cho biến nhị phân
    """
    
    def __init__(self, specs: List[VariableSpec]):
        super().__init__()
        self.specs = specs
        self.handler = VariableHandler(specs)
        
        # Learnable noise for continuous / Nhiễu có thể học được cho biến liên tục
        continuous_idx = [i for i, s in enumerate(specs) 
                         if s.var_type == VariableType.CONTINUOUS]
        if continuous_idx:
            self.log_std = nn.Parameter(torch.zeros(len(continuous_idx)))
        else:
            self.register_buffer('log_std', torch.zeros(1))
        
        self.continuous_idx = continuous_idx
    
    def log_prob(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability.
        Tính log xác suất.
        
        Args:
            x: Encoded observations / Quan sát đã mã hóa
            pred: Predictions (mean for continuous, logits for categorical) / Dự đoán (trung bình cho liên tục, logits cho phân loại)
            
        Returns:
            Log probability per sample / Log xác suất trên mỗi mẫu
        """
        log_probs = []
        cont_idx = 0
        
        for i, spec in enumerate(self.specs):
            enc_start, enc_end = self.handler.encoded_idx[i]
            
            if spec.var_type == VariableType.CONTINUOUS:
                mean = pred[:, enc_start:enc_end]
                std = torch.exp(self.log_std[cont_idx]).clamp(0.01, 2.0)
                
                diff = x[:, enc_start:enc_end] - mean
                log_p = -0.5 * (diff / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)
                log_probs.append(log_p.sum(dim=1))
                
                cont_idx += 1
                
            elif spec.var_type == VariableType.CATEGORICAL:
                logits = pred[:, enc_start:enc_end]
                target = x[:, enc_start:enc_end].argmax(dim=1)
                log_p = -F.cross_entropy(logits, target, reduction='none')
                log_probs.append(log_p)
                
            elif spec.var_type == VariableType.BINARY:
                logit = pred[:, enc_start]
                target = x[:, enc_start]
                log_p = -F.binary_cross_entropy_with_logits(logit, target, reduction='none')
                log_probs.append(log_p)
        
        return torch.stack(log_probs, dim=1).sum(dim=1)


def infer_variable_types(
    data: Union[np.ndarray, torch.Tensor],
    var_names: Optional[List[str]] = None,
    categorical_threshold: int = 10,
) -> List[VariableSpec]:
    """
    Automatically infer variable types from data.
    Tự động suy luận loại biến từ dữ liệu.
    
    Args:
        data: Data array (n_samples, n_vars) / Mảng dữ liệu
        var_names: Optional variable names / Tên biến tùy chọn
        categorical_threshold: Max unique values for categorical / Số giá trị duy nhất tối đa cho biến phân loại
        
    Returns:
        List of VariableSpec / Danh sách VariableSpec
    """
    if torch.is_tensor(data):
        data = data.numpy()
    
    n_vars = data.shape[1]
    
    if var_names is None:
        var_names = [f'X{i}' for i in range(n_vars)]
    
    specs = []
    
    for i in range(n_vars):
        col = data[:, i]
        unique = np.unique(col[~np.isnan(col)])
        
        if len(unique) == 2 and set(unique).issubset({0, 1, 0.0, 1.0}):
            specs.append(VariableSpec(var_names[i], VariableType.BINARY))
        elif len(unique) <= categorical_threshold and np.allclose(unique, unique.astype(int)):
            n_cats = int(unique.max()) + 1
            specs.append(VariableSpec(var_names[i], VariableType.CATEGORICAL, dim=n_cats))
        else:
            specs.append(VariableSpec(var_names[i], VariableType.CONTINUOUS))
    
    return specs


class MixedTypeWrapper(nn.Module):
    """
    Wrapper to handle mixed variable types in CausalMLP.
    Lớp bao để xử lý các loại biến hỗn hợp trong CausalMLP.
    """
    
    def __init__(self, base_model, specs: List[VariableSpec]):
        super().__init__()
        self.base_model = base_model
        self.specs = specs
        self.handler = VariableHandler(specs)
        self.likelihood = MixedLikelihood(specs)
    
    def encode_data(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw data to model format. / Mã hóa dữ liệu thô sang định dạng mô hình."""
        return self.handler.encode(x)
    
    def decode_output(self, x: torch.Tensor) -> torch.Tensor:
        """Decode model output to raw format. / Giải mã đầu ra mô hình sang định dạng thô."""
        return self.handler.decode(x)
    
    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass (expects encoded input). / Lan truyền tiến (kỳ vọng đầu vào đã mã hóa)."""
        return self.base_model(x, **kwargs)
