"""
Uncertainty Quantification for CausalMLP / Định lượng độ không chắc chắn cho CausalMLP

Provides:
Cung cấp:
- Posterior sampling over graphs (Gumbel sampling) / Lấy mẫu hậu nghiệm trên đồ thị (Lấy mẫu Gumbel)
- Bootstrap-based uncertainty / Độ không chắc chắn dựa trên Bootstrap
- Monte Carlo Dropout
- Confidence intervals for edges / Khoảng tin cậy cho các cạnh
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class EdgeUncertainty:
    """Uncertainty information for a single edge. / Thông tin độ không chắc chắn cho một cạnh đơn lẻ."""
    source: int
    target: int
    mean_prob: float
    std_prob: float
    ci_low: float
    ci_high: float
    confidence: float  # Proportion of samples with edge / Tỷ lệ mẫu có cạnh


class UncertaintyEstimator:
    """
    Uncertainty quantification for learned causal structure.
    Định lượng độ không chắc chắn cho cấu trúc nhân quả đã học.
    
    Methods / Các phương pháp:
    1. Gumbel sampling: Sample from adjacency distribution / Lấy mẫu Gumbel: Lấy mẫu từ phân phối kề
    2. Bootstrap: Retrain on resampled data / Bootstrap: Huấn luyện lại trên dữ liệu tái lấy mẫu
    3. MC Dropout: Multiple forward passes with dropout / MC Dropout: Nhiều lần lan truyền tiến với dropout
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.num_nodes = model.config.num_nodes
    
    def gumbel_samples(
        self,
        n_samples: int = 100,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """
        Sample graphs using Gumbel-Softmax reparameterization.
        Lấy mẫu đồ thị sử dụng tái tham số hóa Gumbel-Softmax.
        
        Args:
            n_samples: Number of graph samples / Số lượng mẫu đồ thị
            temperature: Gumbel temperature (lower = sharper) / Nhiệt độ Gumbel (thấp hơn = sắc nét hơn)
            
        Returns:
            Tensor of shape (n_samples, num_nodes, num_nodes)
        """
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                adj = self.model.adjacency.sample(hard=True, temperature=temperature)
                samples.append(adj.cpu())
        
        return torch.stack(samples)
    
    def edge_probabilities(
        self,
        n_samples: int = 100,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """
        Estimate edge probabilities from Gumbel sampling.
        Ước lượng xác suất cạnh từ lấy mẫu Gumbel.
        
        Returns:
            Edge probability matrix (num_nodes, num_nodes)
            Ma trận xác suất cạnh
        """
        samples = self.gumbel_samples(n_samples, temperature)
        return samples.mean(dim=0)
    
    def edge_uncertainty(
        self,
        n_samples: int = 100,
        temperature: float = 0.5,
        alpha: float = 0.05,
    ) -> Dict[Tuple[int, int], EdgeUncertainty]:
        """
        Compute uncertainty for each potential edge.
        Tính độ không chắc chắn cho mỗi cạnh tiềm năng.
        
        Args:
            n_samples: Number of samples / Số lượng mẫu
            temperature: Gumbel temperature / Nhiệt độ Gumbel
            alpha: Significance level for CI / Mức ý nghĩa cho khoảng tin cậy
            
        Returns:
            Dict mapping (i, j) to EdgeUncertainty
            Dict ánh xạ (i, j) đến EdgeUncertainty
        """
        samples = self.gumbel_samples(n_samples, temperature).numpy()
        
        uncertainties = {}
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                
                edge_samples = samples[:, i, j]
                
                uncertainties[(i, j)] = EdgeUncertainty(
                    source=i,
                    target=j,
                    mean_prob=edge_samples.mean(),
                    std_prob=edge_samples.std(),
                    ci_low=np.percentile(edge_samples, 100 * alpha / 2),
                    ci_high=np.percentile(edge_samples, 100 * (1 - alpha / 2)),
                    confidence=(edge_samples > 0.5).mean(),
                )
        
        return uncertainties
    
    def confident_edges(
        self,
        min_confidence: float = 0.8,
        n_samples: int = 100,
        temperature: float = 0.5,
    ) -> List[Tuple[int, int, float]]:
        """
        Get edges with high confidence.
        Lấy các cạnh có độ tin cậy cao.
        
        Args:
            min_confidence: Minimum proportion of samples with edge / Tỷ lệ mẫu tối thiểu có cạnh
            n_samples: Number of samples / Số lượng mẫu
            temperature: Gumbel temperature / Nhiệt độ Gumbel
            
        Returns:
            List of (source, target, confidence) sorted by confidence
            Danh sách (nguồn, đích, độ tin cậy) được sắp xếp theo độ tin cậy
        """
        samples = self.gumbel_samples(n_samples, temperature)
        
        confident = []
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                
                confidence = (samples[:, i, j] > 0.5).float().mean().item()
                
                if confidence >= min_confidence:
                    confident.append((i, j, confidence))
        
        # Sort by confidence (descending) / Sắp xếp theo độ tin cậy (giảm dần)
        confident.sort(key=lambda x: -x[2])
        
        return confident
    
    def bootstrap_uncertainty(
        self,
        data: torch.Tensor,
        n_bootstrap: int = 20,
        train_steps: int = 500,
        lr: float = 0.003,
    ) -> Dict[Tuple[int, int], EdgeUncertainty]:
        """
        Bootstrap-based uncertainty estimation.
        Ước lượng độ không chắc chắn dựa trên Bootstrap.
        
        Retrains model on bootstrap samples of data.
        Huấn luyện lại mô hình trên các mẫu bootstrap của dữ liệu.
        More accurate but much slower.
        Chính xác hơn nhưng chậm hơn nhiều.
        
        Args:
            data: Training data / Dữ liệu huấn luyện
            n_bootstrap: Number of bootstrap iterations / Số lần lặp bootstrap
            train_steps: Steps per bootstrap / Số bước cho mỗi bootstrap
            lr: Learning rate / Tốc độ học
            
        Returns:
            Edge uncertainties / Độ không chắc chắn của cạnh
        """
        n_samples = data.shape[0]
        adjacencies = []
        
        # Store original state / Lưu trạng thái gốc
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        print(f"Bootstrap uncertainty ({n_bootstrap} iterations)...")
        
        for boot in range(n_bootstrap):
            # Bootstrap sample / Lấy mẫu bootstrap
            indices = torch.randint(0, n_samples, (n_samples,))
            boot_data = data[indices].to(self.device)
            
            # Reset to original / Đặt lại về gốc
            self.model.load_state_dict(original_state)
            
            # Quick training / Huấn luyện nhanh
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
            for step in range(train_steps):
                optimizer.zero_grad()
                
                batch_idx = torch.randint(0, n_samples, (min(256, n_samples),))
                batch = boot_data[batch_idx]
                
                result = self.model(batch)
                result['loss'].backward()
                optimizer.step()
            
            # Get adjacency / Lấy ma trận kề
            with torch.no_grad():
                adj = self.model.adjacency.probs.cpu().numpy()
                adjacencies.append(adj)
            
            if (boot + 1) % 5 == 0:
                print(f"  Bootstrap {boot + 1}/{n_bootstrap}")
        
        # Restore original / Khôi phục gốc
        self.model.load_state_dict(original_state)
        
        # Compute uncertainties / Tính độ không chắc chắn
        adjacencies = np.stack(adjacencies)
        
        uncertainties = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                
                edge_samples = adjacencies[:, i, j]
                
                uncertainties[(i, j)] = EdgeUncertainty(
                    source=i,
                    target=j,
                    mean_prob=edge_samples.mean(),
                    std_prob=edge_samples.std(),
                    ci_low=np.percentile(edge_samples, 2.5),
                    ci_high=np.percentile(edge_samples, 97.5),
                    confidence=(edge_samples > 0.5).mean(),
                )
        
        return uncertainties
    
    def summary(self, n_samples: int = 100) -> Dict:
        """
        Get summary of uncertainty over graph structure.
        Lấy tóm tắt độ không chắc chắn trên cấu trúc đồ thị.
        """
        samples = self.gumbel_samples(n_samples)
        
        # Mean adjacency / Ma trận kề trung bình
        mean_adj = samples.mean(dim=0).numpy()
        
        # Standard deviation / Độ lệch chuẩn
        std_adj = samples.std(dim=0).numpy()
        
        # Number of edges per sample / Số lượng cạnh mỗi mẫu
        n_edges = (samples > 0.5).sum(dim=(1, 2)).float()
        
        # High confidence edges / Các cạnh có độ tin cậy cao
        confident = self.confident_edges(min_confidence=0.8, n_samples=n_samples)
        
        return {
            'mean_adjacency': mean_adj,
            'std_adjacency': std_adj,
            'mean_n_edges': n_edges.mean().item(),
            'std_n_edges': n_edges.std().item(),
            'min_n_edges': n_edges.min().item(),
            'max_n_edges': n_edges.max().item(),
            'n_confident_edges': len(confident),
        }


class MCDropoutEstimator:
    """
    Monte Carlo Dropout for uncertainty.
    Monte Carlo Dropout cho độ không chắc chắn.
    
    Requires model to have dropout layers enabled.
    Yêu cầu mô hình phải bật các lớp dropout.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def _enable_dropout(self):
        """Enable dropout during inference. / Bật dropout trong quá trình suy luận."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def _disable_dropout(self):
        """Disable dropout. / Tắt dropout."""
        self.model.eval()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty via MC Dropout.
        Đưa ra dự đoán với độ không chắc chắn thông qua MC Dropout.
        
        Returns:
            (mean_prediction, std_prediction)
        """
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                adj = self.model.adjacency.probs.to(self.device)
                mean, _ = self.model.mlp(x.to(self.device), adj)
                predictions.append(mean.cpu())
        
        self._disable_dropout()
        
        predictions = torch.stack(predictions)
        
        return predictions.mean(dim=0), predictions.std(dim=0)
