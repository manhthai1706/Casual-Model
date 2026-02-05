"""
Spline Flows for CausalMLP / Spline Flows cho CausalMLP

Implements Neural Spline Flows for flexible non-Gaussian noise modeling.
Triển khai Neural Spline Flows cho mô hình hóa nhiễu phi Gaussian linh hoạt.
Based on "Neural Spline Flows" (Durkan et al., 2019).
Dựa trên "Neural Spline Flows" (Durkan et al., 2019).

Benefits / Lợi ích:
- Can model arbitrary distributions / Có thể mô hình hóa các phân phối bất kỳ
- More expressive than Gaussian/heteroscedastic / Diễn đạt tốt hơn so với Gaussian/dị phương sai
- Invertible for density estimation / Có thể nghịch đảo để ước lượng mật độ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import math


def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary search for bin locations. / Tìm kiếm nhị phân cho vị trí thùng."""
    bin_locations = bin_locations.contiguous()
    inputs = inputs.contiguous()
    return torch.searchsorted(bin_locations, inputs.unsqueeze(-1), right=True).squeeze(-1) - 1


def rational_quadratic_spline(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rational Quadratic Spline transformation.
    Biến đổi Rational Quadratic Spline.
    
    Args:
        inputs: Input values / Giá trị đầu vào
        widths: Bin widths (K bins) / Chiều rộng thùng (K thùng)
        heights: Bin heights (K bins) / Chiều cao thùng (K thùng)
        derivatives: Derivatives at knots (K+1 values) / Đạo hàm tại các nút (K+1 giá trị)
        inverse: Whether to compute inverse / Có tính nghịch đảo hay không
        tail_bound: Bound for linear tails / Giới hạn cho phần đuôi tuyến tính
        
    Returns:
        Tuple of (outputs, log_abs_det_jacobian)
        Bộ (đầu ra, log định thức Jacobian)
    """
    # Handle tails (identity outside bounds)
    # Xử lý phần đuôi (đồng nhất bên ngoài giới hạn)
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask
    
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
    
    # Identity for tails / Đồng nhất cho phần đuôi
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
    
    if inside_interval_mask.sum() == 0:
        return outputs, logabsdet
    
    # Get inputs inside interval / Lấy đầu vào trong khoảng
    inputs_inside = inputs[inside_interval_mask]
    
    # Normalize widths and heights / Chuẩn hóa chiều rộng và chiều cao
    widths = F.softmax(widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * widths.shape[-1]) * widths
    
    heights = F.softmax(heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * heights.shape[-1]) * heights
    
    # Ensure positive derivatives / Đảm bảo đạo hàm dương
    derivatives = min_derivative + F.softplus(derivatives)
    
    # Scale to interval / Tỷ lệ theo khoảng
    widths = 2 * tail_bound * widths
    heights = 2 * tail_bound * heights
    
    # Cumulative widths and heights / Chiều rộng và chiều cao tích lũy
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=-tail_bound)
    cumwidths = (cumwidths[..., :-1] + cumwidths[..., 1:]) / 2  # Midpoints
    cumwidths[..., 0] = -tail_bound
    cumwidths[..., -1] = tail_bound
    
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=-tail_bound)
    cumheights = (cumheights[..., :-1] + cumheights[..., 1:]) / 2
    cumheights[..., 0] = -tail_bound
    cumheights[..., -1] = tail_bound
    
    # Find bin indices / Tìm chỉ số thùng
    if inverse:
        bin_idx = searchsorted(cumheights, inputs_inside)
    else:
        bin_idx = searchsorted(cumwidths, inputs_inside)
    
    bin_idx = bin_idx.clamp(0, widths.shape[-1] - 1)
    
    # Get bin parameters / Lấy tham số thùng
    input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    delta = input_bin_heights / input_bin_widths
    
    input_derivatives = derivatives.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_derivatives_plus_one = derivatives.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    
    # Compute spline / Tính toán spline
    if inverse:
        a = input_bin_heights * (input_derivatives + input_derivatives_plus_one - 2 * delta)
        b = input_bin_heights * (delta - input_derivatives) - a * (inputs_inside - input_cumheights)
        c = -delta * (inputs_inside - input_cumheights)
        
        discriminant = b.pow(2) - 4 * a * c
        discriminant = discriminant.clamp(min=0)
        
        root = (-b + torch.sqrt(discriminant)) / (2 * a + 1e-8)
        outputs_inside = root * input_bin_widths + input_cumwidths
        
        theta_one_minus_theta = root * (1 - root)
        denominator = delta + ((input_derivatives + input_derivatives_plus_one - 2 * delta) * theta_one_minus_theta)
        derivative_numerator = delta.pow(2) * (input_derivatives_plus_one * root.pow(2) + 2 * delta * theta_one_minus_theta + input_derivatives * (1 - root).pow(2))
        logabsdet_inside = torch.log(derivative_numerator + 1e-8) - 2 * torch.log(denominator.abs() + 1e-8)
    else:
        theta = (inputs_inside - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        
        numerator = input_bin_heights * (delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = delta + ((input_derivatives + input_derivatives_plus_one - 2 * delta) * theta_one_minus_theta)
        outputs_inside = input_cumheights + numerator / denominator
        
        derivative_numerator = delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2 * delta * theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
        logabsdet_inside = torch.log(derivative_numerator + 1e-8) - 2 * torch.log(denominator.abs() + 1e-8)
    
    outputs[inside_interval_mask] = outputs_inside
    logabsdet[inside_interval_mask] = logabsdet_inside
    
    return outputs, logabsdet


class SplineFlow(nn.Module):
    """
    Neural Spline Flow for one variable.
    Luồng Spline Nơ-ron cho một biến.
    
    Uses rational quadratic splines for flexible density estimation.
    Sử dụng rational quadratic splines để ước lượng mật độ linh hoạt.
    """
    
    def __init__(
        self,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        hidden_dim: int = 32,
        num_context: int = 0,  # Input dimension for conditional / Chiều đầu vào cho điều kiện
    ):
        super().__init__()
        
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        
        # Parameters: widths (K) + heights (K) + derivatives (K+1)
        # Tham số: chiều rộng (K) + chiều cao (K) + đạo hàm (K+1)
        num_params = 3 * num_bins + 1
        
        if num_context > 0:
            # Conditional spline / Spline có điều kiện
            self.param_net = nn.Sequential(
                nn.Linear(num_context, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_params),
            )
        else:
            # Unconditional spline / Spline không điều kiện
            self.params = nn.Parameter(torch.zeros(num_params))
            self.param_net = None
    
    def _get_params(self, context: Optional[torch.Tensor] = None):
        """Get spline parameters. / Lấy tham số spline."""
        if self.param_net is not None and context is not None:
            params = self.param_net(context)
        else:
            params = self.params.unsqueeze(0)
        
        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2*self.num_bins]
        derivatives = params[..., 2*self.num_bins:]
        
        return widths, heights, derivatives
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x -> z.
        
        Returns:
            (z, log_det_jacobian)
        """
        widths, heights, derivatives = self._get_params(context)
        
        # Broadcast if needed
        if widths.shape[0] == 1 and x.shape[0] > 1:
            widths = widths.expand(x.shape[0], -1)
            heights = heights.expand(x.shape[0], -1)
            derivatives = derivatives.expand(x.shape[0], -1)
        
        z, log_det = rational_quadratic_spline(
            x, widths, heights, derivatives,
            inverse=False, tail_bound=self.tail_bound
        )
        
        return z, log_det
    
    def inverse(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass: z -> x.
        Lan truyền ngược: z -> x.
        """
        widths, heights, derivatives = self._get_params(context)
        
        if widths.shape[0] == 1 and z.shape[0] > 1:
            widths = widths.expand(z.shape[0], -1)
            heights = heights.expand(z.shape[0], -1)
            derivatives = derivatives.expand(z.shape[0], -1)
        
        x, log_det = rational_quadratic_spline(
            z, widths, heights, derivatives,
            inverse=True, tail_bound=self.tail_bound
        )
        
        return x, -log_det
    
    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability.
        Tính log xác suất.
        
        log p(x) = log p(z) + log |det J|
        where z = f(x) and p(z) = N(0, 1)
        """
        z, log_det = self.forward(x, context)
        
        # Standard normal log prob / Log xác suất chuẩn tắc
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi))
        
        return log_pz + log_det


class SplineNoiseModel(nn.Module):
    """
    Spline-based noise model for CausalMLP.
    Mô hình nhiễu dựa trên Spline cho CausalMLP.
    
    Each node has its own conditional spline flow.
    Mỗi nút có một luồng spline có điều kiện riêng.
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_bins: int = 8,
        hidden_dim: int = 32,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # One spline per node, conditioned on predicted mean
        # Một spline cho mỗi nút, điều kiện hóa dựa trên trung bình dự đoán
        self.splines = nn.ModuleList([
            SplineFlow(num_bins=num_bins, hidden_dim=hidden_dim, num_context=1)
            for _ in range(num_nodes)
        ])
    
    def log_prob(
        self,
        x: torch.Tensor,          # Observations (batch, num_nodes) / Các quan sát
        mean: torch.Tensor,        # Predicted means (batch, num_nodes) / Trung bình dự đoán
        log_std: Optional[torch.Tensor] = None,  # Not used, for API compatibility / Không dùng, để tương thích API
    ) -> torch.Tensor:
        """
        Compute log probability for each node.
        Tính log xác suất cho mỗi nút.
        
        Returns:
            Log prob per sample (batch,) / Log xác suất mỗi mẫu
        """
        batch_size = x.shape[0]
        log_probs = []
        
        for i in range(self.num_nodes):
            # Residual / Phần dư
            residual = x[:, i] - mean[:, i]
            
            # Context is the predicted mean / Bối cảnh là trung bình dự đoán
            context = mean[:, i:i+1]
            
            # Log prob from spline / Log xác suất từ spline
            log_p = self.splines[i].log_prob(residual, context)
            log_probs.append(log_p)
        
        return torch.stack(log_probs, dim=1).sum(dim=1)
    
    def sample(
        self,
        mean: torch.Tensor,
        log_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the noise model. / Lấy mẫu từ mô hình nhiễu."""
        batch_size = mean.shape[0]
        samples = torch.zeros_like(mean)
        
        for i in range(self.num_nodes):
            # Sample from standard normal / Lấy mẫu từ chuẩn tắc
            z = torch.randn(batch_size, device=mean.device)
            
            # Transform through inverse spline / Biến đổi qua spline nghịch đảo
            context = mean[:, i:i+1]
            noise, _ = self.splines[i].inverse(z, context)
            
            samples[:, i] = mean[:, i] + noise
        
        return samples
