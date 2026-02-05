"""
Neural CATE Estimator for CausalMLP / Bộ ước lượng CATE Nơ-ron cho CausalMLP

Deep learning-based Conditional Average Treatment Effect estimation.
Ước lượng Hiệu quả Xử lý Trung bình có Điều kiện dựa trên học sâu.

Implements / Triển khai:
- TARNet (Treatment-Agnostic Representation Network) / TARNet (Mạng biểu diễn bất khả tri điều trị)
- DragonNet-style architecture / Kiến trúc kiểu DragonNet
- Doubly Robust estimation / Ước lượng Doubly Robust

Reference: Shalit et al., "Estimating individual treatment effect: 
generalization bounds and algorithms" (ICML 2017)
Tham khảo: Shalit và cộng sự, "Ước lượng hiệu quả xử lý cá nhân: 
các giới hạn tổng quát hóa và thuật toán" (ICML 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class RepresentationNetwork(nn.Module):
    """
    Shared representation network.
    Mạng biểu diễn dùng chung.
    
    Maps covariates X to a balanced representation Φ(X).
    Ánh xạ các biến đồng lượng X sang một biểu diễn cân bằng Φ(X).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        rep_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [rep_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OutcomeHead(nn.Module):
    """
    Outcome prediction head.
    Đầu dự đoán kết quả.
    
    Predicts E[Y | X, T=t] from representation.
    Dự đoán E[Y | X, T=t] từ biểu diễn.
    """
    
    def __init__(
        self,
        rep_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        dims = [rep_dim] + [hidden_dim] * num_layers + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.network(rep).squeeze(-1)


class TARNet(nn.Module):
    """
    Treatment-Agnostic Representation Network.
    Mạng biểu diễn bất khả tri điều trị.
    
    Architecture / Kiến trúc:
    - Shared representation: X → Φ(X) / Biểu diễn dùng chung
    - Treatment head (T=1): Φ(X) → Y_1 / Đầu điều trị
    - Control head (T=0): Φ(X) → Y_0 / Đầu đối chứng
    
    CATE(x) = E[Y|X=x, T=1] - E[Y|X=x, T=0]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        rep_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.representation = RepresentationNetwork(
            input_dim, hidden_dim, rep_dim, num_layers
        )
        
        self.head_treated = OutcomeHead(rep_dim, hidden_dim // 2, num_layers)
        self.head_control = OutcomeHead(rep_dim, hidden_dim // 2, num_layers)
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Lan truyền tiến.
        
        Args:
            x: Covariates (batch, input_dim) / Biến đồng lượng
            t: Treatment indicator (batch,) - optional / Chỉ báo điều trị - tùy chọn
            
        Returns:
            Dict with y_0, y_1, cate predictions
            Dict chứa dự đoán y_0, y_1, cate
        """
        rep = self.representation(x)
        
        y_0 = self.head_control(rep)
        y_1 = self.head_treated(rep)
        
        cate = y_1 - y_0
        
        result = {
            'y_0': y_0,
            'y_1': y_1,
            'cate': cate,
            'representation': rep,
        }
        
        if t is not None:
            # Factual prediction / Dự đoán thực tế
            y_factual = t * y_1 + (1 - t) * y_0
            result['y_factual'] = y_factual
        
        return result
    
    def predict_cate(self, x: torch.Tensor) -> torch.Tensor:
        """Get CATE prediction. / Lấy dự đoán CATE."""
        with torch.no_grad():
            return self(x)['cate']


class DragonNet(nn.Module):
    """
    DragonNet: Joint propensity and outcome prediction.
    DragonNet: Dự đoán đồng thời xu hướng và kết quả.
    
    Architecture / Kiến trúc:
    - Shared representation: X → Φ(X) / Biểu diễn dùng chung
    - Propensity head: Φ(X) → P(T=1|X) / Đầu xu hướng
    - Outcome heads: Φ(X) → Y_0, Y_1 / Các đầu kết quả
    
    The propensity prediction acts as regularization for targeted learning.
    Dự đoán xu hướng hoạt động như một sự điều chuẩn cho việc học có mục tiêu.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        rep_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.representation = RepresentationNetwork(
            input_dim, hidden_dim, rep_dim, num_layers
        )
        
        self.head_treated = OutcomeHead(rep_dim, hidden_dim // 2, num_layers)
        self.head_control = OutcomeHead(rep_dim, hidden_dim // 2, num_layers)
        
        # Propensity head / Đầu xu hướng
        self.propensity_head = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass. / Lan truyền tiến."""
        rep = self.representation(x)
        
        y_0 = self.head_control(rep)
        y_1 = self.head_treated(rep)
        propensity = torch.sigmoid(self.propensity_head(rep).squeeze(-1))
        
        cate = y_1 - y_0
        
        result = {
            'y_0': y_0,
            'y_1': y_1,
            'cate': cate,
            'propensity': propensity,
            'representation': rep,
        }
        
        if t is not None:
            y_factual = t * y_1 + (1 - t) * y_0
            result['y_factual'] = y_factual
        
        return result


class CATETrainer:
    """
    Trainer for CATE estimators.
    Bộ huấn luyện cho các bộ ước lượng CATE.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.001,
        alpha: float = 1.0,  # Propensity loss weight / Trọng số mất mát xu hướng
        beta: float = 0.1,   # Representation balance weight / Trọng số cân bằng biểu diễn
    ):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        Tính toán mất mát huấn luyện.
        
        Args:
            x: Covariates (batch, dim) / Biến đồng lượng
            t: Treatment (batch,) - binary / Điều trị (nhị phân)
            y: Outcome (batch,) / Kết quả
        """
        out = self.model(x, t)
        
        # Factual outcome loss / Mất mát kết quả thực tế
        y_pred = out['y_factual']
        outcome_loss = F.mse_loss(y_pred, y)
        
        losses = {'outcome': outcome_loss}
        total_loss = outcome_loss
        
        # Propensity loss (DragonNet) / Mất mát xu hướng (DragonNet)
        if 'propensity' in out:
            prop_loss = F.binary_cross_entropy(out['propensity'], t)
            losses['propensity'] = prop_loss
            total_loss = total_loss + self.alpha * prop_loss
        
        # Representation balance (IPM) / Cân bằng biểu diễn (IPM)
        if self.beta > 0:
            rep = out['representation']
            rep_treated = rep[t > 0.5]
            rep_control = rep[t < 0.5]
            
            if len(rep_treated) > 0 and len(rep_control) > 0:
                # MMD (Maximum Mean Discrepancy) approximation
                # Xấp xỉ MMD (Sai biệt trung bình tối đa)
                mean_diff = (rep_treated.mean(0) - rep_control.mean(0)).pow(2).sum()
                losses['balance'] = mean_diff
                total_loss = total_loss + self.beta * mean_diff
        
        losses['total'] = total_loss
        
        return losses
    
    def fit(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 1000,
        batch_size: int = 256,
        verbose: bool = True,
    ):
        """
        Train the model.
        Huấn luyện mô hình.
        
        Args:
            x: Covariates / Biến đồng lượng
            t: Treatment indicator (binary) / Chỉ báo điều trị (nhị phân)
            y: Observed outcome / Kết quả quan sát được
            n_epochs: Number of epochs / Số lượng kỷ nguyên
            batch_size: Batch size / Kích thước lô
            verbose: Print progress / In tiến trình
        """
        n_samples = x.shape[0]
        
        for epoch in range(n_epochs):
            # Shuffle / Xáo trộn
            perm = torch.randperm(n_samples)
            
            epoch_losses = []
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                
                self.optimizer.zero_grad()
                losses = self.compute_loss(x[idx], t[idx], y[idx])
                losses['total'].backward()
                self.optimizer.step()
                
                epoch_losses.append(losses['total'].item())
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: Loss = {np.mean(epoch_losses):.4f}")
    
    def estimate_ate(
        self,
        x: torch.Tensor,
        n_bootstrap: int = 100,
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect with uncertainty.
        Ước lượng Hiệu quả Xử lý Trung bình với độ không chắc chắn.
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(x)
            cate = out['cate']
            ate = cate.mean().item()
        
        # Bootstrap confidence interval / Khoảng tin cậy Bootstrap
        ate_samples = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, len(x), (len(x),))
            ate_boot = cate[idx].mean().item()
            ate_samples.append(ate_boot)
        
        return {
            'ate': ate,
            'std': np.std(ate_samples),
            'ci_low': np.percentile(ate_samples, 2.5),
            'ci_high': np.percentile(ate_samples, 97.5),
        }


class DoublyRobustEstimator:
    """
    Doubly Robust CATE estimation.
    Ước lượng CATE Doubly Robust.
    
    Combines outcome model and propensity weighting for robustness.
    Kết hợp mô hình kết quả và trọng số xu hướng để tăng tính bền vững.
    DR estimator is consistent if either the outcome model OR the propensity model is correctly specified.
    Bộ ước lượng DR nhất quán nếu mô hình kết quả HOẶC mô hình xu hướng được chỉ định chính xác.
    """
    
    def __init__(
        self,
        outcome_model: nn.Module,
        propensity_model: Optional[nn.Module] = None,
    ):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
    
    def estimate_cate(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute doubly robust CATE estimate.
        Tính ước lượng CATE Doubly Robust.
        
        DR_i = (μ_1(x_i) - μ_0(x_i)) 
               + t_i/e(x_i) * (y_i - μ_1(x_i))
               - (1-t_i)/(1-e(x_i)) * (y_i - μ_0(x_i))
        """
        with torch.no_grad():
            out = self.outcome_model(x)
            mu_0 = out['y_0']
            mu_1 = out['y_1']
            
            if 'propensity' in out:
                e = out['propensity']
            elif self.propensity_model is not None:
                e = self.propensity_model(x)
            else:
                # Use empirical propensity / Sử dụng xu hướng thực nghiệm
                e = torch.full_like(t, t.mean())
            
            # Clip propensity for stability / Cắt bớt xu hướng để ổn định
            e = e.clamp(0.01, 0.99)
        
        # Doubly robust formula / Công thức Doubly Robust
        dr_1 = mu_1 + t / e * (y - mu_1)
        dr_0 = mu_0 + (1 - t) / (1 - e) * (y - mu_0)
        
        cate = dr_1 - dr_0
        
        return cate
    
    def estimate_ate(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Estimate ATE with doubly robust method. / Ước lượng ATE với phương pháp Doubly Robust."""
        cate = self.estimate_cate(x, t, y)
        
        return {
            'ate': cate.mean().item(),
            'std': cate.std().item(),
        }
