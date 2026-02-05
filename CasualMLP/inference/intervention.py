"""
Intervention Module for CausalMLP / Mô-đun Can thiệp cho CausalMLP

Implements do-calculus operations:
Triển khai các phép toán do-calculus:
- do() interventions / Can thiệp do()
- Average Treatment Effect (ATE) / Hiệu quả xử lý trung bình (ATE)
- Conditional Average Treatment Effect (CATE) / Hiệu quả xử lý trung bình có điều kiện (CATE)
- Individual Treatment Effect (ITE) / Hiệu quả xử lý cá nhân (ITE)
- Counterfactual reasoning / Suy luận phản thực tế

Based on DECI's intervention capabilities.
Dựa trên các khả năng can thiệp của DECI.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
import numpy as np


class CausalInference:
    """
    Causal inference engine for CausalMLP.
    Công cụ suy luận nhân quả cho CausalMLP.
    
    Enables causal queries after training:
    Cho phép các truy vấn nhân quả sau khi huấn luyện:
    - P(Y | do(X = x))
    - E[Y | do(X = x)] - E[Y | do(X = x')]  (ATE)
    - What would Y have been if X were different? (Counterfactual) / Y sẽ ra sao nếu X khác đi? (Phản thực tế)
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Args:
            model: Trained CausalMLPModel / Mô hình CausalMLP đã huấn luyện
            device: Computation device / Thiết bị tính toán
        """
        self.model = model
        self.device = device
        self.num_nodes = model.config.num_nodes
        self.model.eval()
    
    def _get_adjacency(self, threshold: float = 0.5) -> torch.Tensor:
        """Get learned adjacency matrix. / Lấy ma trận kề đã học."""
        with torch.no_grad():
            adj = self.model.adjacency.probs
            if threshold > 0:
                adj = (adj > threshold).float()
            return adj.to(self.device)
    
    def _topological_order(self, adjacency: torch.Tensor) -> List[int]:
        """
        Compute topological order using Kahn's algorithm.
        Tính thứ tự topo sử dụng thuật toán Kahn.
        """
        adj_np = adjacency.cpu().numpy()
        n = adj_np.shape[0]
        
        in_degree = (adj_np > 0.5).sum(axis=0).astype(int)
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        
        temp_adj = (adj_np > 0.5).copy()
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in range(n):
                if temp_adj[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        # Handle cycles by adding remaining nodes / Xử lý chu trình bằng cách thêm các nút còn lại
        if len(order) < n:
            remaining = [i for i in range(n) if i not in order]
            order.extend(remaining)
        
        return order
    
    def sample_observational(
        self,
        n_samples: int = 1000,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from observational distribution P(X).
        Lấy mẫu từ phân phối quan sát P(X).
        
        Uses ancestral sampling following topological order.
        Sử dụng lấy mẫu tổ tiên theo thứ tự topo.
        """
        if adjacency is None:
            adjacency = self._get_adjacency()
        
        order = self._topological_order(adjacency)
        samples = torch.zeros(n_samples, self.num_nodes, device=self.device)
        noise = torch.randn(n_samples, self.num_nodes, device=self.device)
        
        for node in order:
            parents = torch.where(adjacency[:, node] > 0.5)[0]
            
            if len(parents) == 0:
                # Root node / Nút gốc
                samples[:, node] = noise[:, node]
            else:
                # Compute mean from MLP / Tính trung bình từ MLP
                with torch.no_grad():
                    mean, log_std = self.model.mlp(samples, adjacency)
                    std = torch.exp(log_std[:, node]).clamp(0.01, 2.0) if log_std is not None else 0.5
                    samples[:, node] = mean[:, node] + noise[:, node] * std
        
        return samples
    
    def do(
        self,
        interventions: Dict[int, float],
        n_samples: int = 1000,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply do-calculus intervention: P(X | do(X_i = v)).
        Áp dụng can thiệp do-calculus: P(X | do(X_i = v)).
        
        The do() operator: / Toán tử do():
        1. Removes all incoming edges to intervened nodes / Loại bỏ tất cả các cạnh đến các nút bị can thiệp
        2. Sets intervened nodes to fixed values / Đặt các nút bị can thiệp thành các giá trị cố định
        3. Samples remaining variables causally / Lấy mẫu các biến còn lại theo nhân quả
        
        Args:
            interventions: Dict mapping node index to intervention value / Dict ánh xạ chỉ số nút đến giá trị can thiệp
                          e.g., {0: 1.5, 2: -0.5} means do(X_0=1.5, X_2=-0.5)
            n_samples: Number of samples to generate / Số lượng mẫu cần tạo
            adjacency: Optional custom adjacency matrix / Ma trận kề tùy chọn
            
        Returns:
            Samples from interventional distribution P(X | do(...))
            Các mẫu từ phân phối can thiệp
        """
        if adjacency is None:
            adjacency = self._get_adjacency()
        
        # Modify adjacency: remove incoming edges to intervened nodes
        # Sửa đổi ma trận kề: loại bỏ các cạnh đến các nút bị can thiệp
        intervened_adj = adjacency.clone()
        for node in interventions.keys():
            intervened_adj[:, node] = 0  # Remove all incoming edges
        
        order = self._topological_order(intervened_adj)
        samples = torch.zeros(n_samples, self.num_nodes, device=self.device)
        noise = torch.randn(n_samples, self.num_nodes, device=self.device)
        
        for node in order:
            if node in interventions:
                # Intervened: set to fixed value / Bị can thiệp: đặt giá trị cố định
                samples[:, node] = interventions[node]
            else:
                parents = torch.where(intervened_adj[:, node] > 0.5)[0]
                
                if len(parents) == 0:
                    samples[:, node] = noise[:, node]
                else:
                    with torch.no_grad():
                        mean, log_std = self.model.mlp(samples, intervened_adj)
                        std = torch.exp(log_std[:, node]).clamp(0.01, 2.0) if log_std is not None else 0.5
                        samples[:, node] = mean[:, node] + noise[:, node] * std
        
        return samples
    
    def ate(
        self,
        treatment: int,
        outcome: int,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
        n_samples: int = 5000,
    ) -> Dict[str, float]:
        """
        Compute Average Treatment Effect.
        Tính Hiệu quả Xử lý Trung bình (ATE).
        
        ATE = E[Y | do(T = treatment_value)] - E[Y | do(T = control_value)]
        
        Args:
            treatment: Index of treatment variable / Chỉ số biến điều trị
            outcome: Index of outcome variable / Chỉ số biến kết quả
            treatment_value: Value for treatment condition / Giá trị cho điều kiện điều trị
            control_value: Value for control condition / Giá trị cho điều kiện đối chứng
            n_samples: Number of samples for Monte Carlo estimation / Số lượng mẫu ước lượng Monte Carlo
            
        Returns:
            Dict with ATE, confidence interval, and E[Y] under each condition
            Dict chứa ATE, khoảng tin cậy và E[Y] dưới mỗi điều kiện
        """
        # Sample under treatment / Lấy mẫu dưới điều kiện điều trị
        samples_treat = self.do({treatment: treatment_value}, n_samples)
        y_treat = samples_treat[:, outcome]
        
        # Sample under control / Lấy mẫu dưới điều kiện đối chứng
        samples_control = self.do({treatment: control_value}, n_samples)
        y_control = samples_control[:, outcome]
        
        # Compute ATE / Tính ATE
        ate = (y_treat.mean() - y_control.mean()).item()
        
        # Bootstrap confidence interval / Khoảng tin cậy Bootstrap
        n_bootstrap = 1000
        ate_samples = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, n_samples, (n_samples,))
            ate_boot = y_treat[idx].mean() - y_control[idx].mean()
            ate_samples.append(ate_boot.item())
        
        ate_samples = np.array(ate_samples)
        
        return {
            'ate': ate,
            'std': np.std(ate_samples),
            'ci_low': np.percentile(ate_samples, 2.5),
            'ci_high': np.percentile(ate_samples, 97.5),
            'e_y_treat': y_treat.mean().item(),
            'e_y_control': y_control.mean().item(),
            'n_samples': n_samples,
        }
    
    def cate(
        self,
        treatment: int,
        outcome: int,
        condition: Dict[int, Tuple[float, float]],  # {node: (low, high)}
        treatment_value: float = 1.0,
        control_value: float = 0.0,
        n_samples: int = 5000,
    ) -> Dict[str, float]:
        """
        Compute Conditional Average Treatment Effect.
        Tính Hiệu quả Xử lý Trung bình có Điều kiện (CATE).
        
        CATE = E[Y | do(T=1), X in range] - E[Y | do(T=0), X in range]
        
        Args:
            treatment: Treatment node index / Chỉ số nút điều trị
            outcome: Outcome node index / Chỉ số nút kết quả
            condition: Dict mapping node to (low, high) range / Dict ánh xạ nút đến khoảng (thấp, cao)
            treatment_value: Treatment value / Giá trị điều trị
            control_value: Control value / Giá trị đối chứng
            n_samples: Number of samples / Số lượng mẫu
            
        Returns:
            Dict with CATE and statistics / Dict chứa CATE và thống kê
        """
        # Sample more to filter / Lấy mẫu nhiều hơn để lọc
        samples_treat = self.do({treatment: treatment_value}, n_samples * 3)
        samples_control = self.do({treatment: control_value}, n_samples * 3)
        
        # Filter by condition / Lọc theo điều kiện
        def filter_by_condition(samples):
            mask = torch.ones(samples.shape[0], dtype=torch.bool, device=self.device)
            for node, (low, high) in condition.items():
                mask &= (samples[:, node] >= low) & (samples[:, node] <= high)
            return samples[mask][:n_samples]
        
        filtered_treat = filter_by_condition(samples_treat)
        filtered_control = filter_by_condition(samples_control)
        
        if len(filtered_treat) < 100 or len(filtered_control) < 100:
            return {
                'cate': float('nan'),
                'error': 'Not enough samples satisfy condition / Không đủ mẫu thỏa mãn điều kiện',
                'n_treat': len(filtered_treat),
                'n_control': len(filtered_control),
            }
        
        cate = (filtered_treat[:, outcome].mean() - filtered_control[:, outcome].mean()).item()
        
        return {
            'cate': cate,
            'e_y_treat': filtered_treat[:, outcome].mean().item(),
            'e_y_control': filtered_control[:, outcome].mean().item(),
            'n_treat': len(filtered_treat),
            'n_control': len(filtered_control),
        }
    
    def ite(
        self,
        factual: torch.Tensor,
        treatment: int,
        outcome: int,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute Individual Treatment Effect.
        Tính Hiệu quả Xử lý Cá nhân (ITE).
        
        ITE = Y(T=treatment_value) - Y(T=control_value) for a specific individual.
        ITE cho một cá nhân cụ thể.
        
        Uses abduction-action-prediction:
        Sử dụng quy trình suy diễn-hành động-dự đoán:
        1. Abduction: Infer exogenous noise from observation / Suy diễn: Suy ra nhiễu ngoại sinh từ quan sát
        2. Action: Apply intervention / Hành động: Áp dụng can thiệp
        3. Prediction: Compute counterfactual outcome / Dự đoán: Tính kết quả phản thực tế
        
        Args:
            factual: Observed sample (1, num_nodes) or (num_nodes,) / Mẫu quan sát được
            treatment: Treatment node index / Chỉ số nút điều trị
            outcome: Outcome node index / Chỉ số nút kết quả
            treatment_value: Counterfactual treatment value / Giá trị điều trị phản thực tế
            control_value: Counterfactual control value / Giá trị đối chứng phản thực tế
            
        Returns:
            Dict with ITE and counterfactual outcomes / Dict chứa ITE và kết quả phản thực tế
        """
        if factual.dim() == 1:
            factual = factual.unsqueeze(0)
        factual = factual.to(self.device)
        
        adjacency = self._get_adjacency()
        
        # Abduction: Infer noise / Suy diễn: Suy ra nhiễu
        with torch.no_grad():
            mean, _ = self.model.mlp(factual, adjacency)
            noise = factual - mean
        
        # Counterfactual under treatment / Phản thực tế dưới điều kiện điều trị
        cf_treat = self._counterfactual_sample(factual, noise, {treatment: treatment_value}, adjacency)
        
        # Counterfactual under control / Phản thực tế dưới điều kiện đối chứng
        cf_control = self._counterfactual_sample(factual, noise, {treatment: control_value}, adjacency)
        
        ite = (cf_treat[:, outcome] - cf_control[:, outcome]).item()
        
        return {
            'ite': ite,
            'y_treat': cf_treat[:, outcome].item(),
            'y_control': cf_control[:, outcome].item(),
            'factual_y': factual[:, outcome].item(),
            'factual_treatment': factual[:, treatment].item(),
        }
    
    def counterfactual(
        self,
        factual: torch.Tensor,
        interventions: Dict[int, float],
    ) -> torch.Tensor:
        """
        Compute counterfactual: "What would have happened if...?"
        Tính phản thực tế: "Điều gì sẽ xảy ra nếu...?"
        
        Args:
            factual: Observed factual sample / Mẫu thực tế quan sát được
            interventions: Counterfactual interventions / Can thiệp phản thực tế
            
        Returns:
            Counterfactual sample / Mẫu phản thực tế
        """
        if factual.dim() == 1:
            factual = factual.unsqueeze(0)
        factual = factual.to(self.device)
        
        adjacency = self._get_adjacency()
        
        # Abduction / Suy diễn
        with torch.no_grad():
            mean, _ = self.model.mlp(factual, adjacency)
            noise = factual - mean
        
        return self._counterfactual_sample(factual, noise, interventions, adjacency)
    
    def _counterfactual_sample(
        self,
        factual: torch.Tensor,
        noise: torch.Tensor,
        interventions: Dict[int, float],
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Internal: Generate counterfactual sample. / Nội bộ: Tạo mẫu phản thực tế."""
        cf = factual.clone()
        
        # Apply interventions / Áp dụng can thiệp
        for node, value in interventions.items():
            cf[:, node] = value
        
        # Recompute downstream variables / Tính toán lại các biến hạ nguồn
        order = self._topological_order(adjacency)
        
        for node in order:
            if node in interventions:
                continue
            
            # Check if affected by intervention / Kiểm tra xem có bị ảnh hưởng bởi can thiệp không
            parents = torch.where(adjacency[:, node] > 0.5)[0]
            if len(parents) > 0:
                with torch.no_grad():
                    mean, _ = self.model.mlp(cf, adjacency)
                    cf[:, node] = mean[:, node] + noise[:, node]
        
        return cf
    
    def causal_effect_matrix(
        self,
        n_samples: int = 2000,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute all pairwise causal effects.
        Tính tất cả các hiệu quả nhân quả từng cặp.
        
        Returns:
            Matrix where [i, j] = effect of do(X_i) on X_j
            Ma trận trong đó [i, j] = hiệu quả của do(X_i) lên X_j
        """
        effects = torch.zeros(self.num_nodes, self.num_nodes)
        
        for i in range(self.num_nodes):
            result_treat = self.do({i: treatment_value}, n_samples)
            result_control = self.do({i: control_value}, n_samples)
            
            for j in range(self.num_nodes):
                if i != j:
                    effect = result_treat[:, j].mean() - result_control[:, j].mean()
                    effects[i, j] = effect.item()
        
        return effects


def add_causal_methods(model):
    """Add causal inference methods to model instance. / Thêm các phương pháp suy luận nhân quả vào thể hiện mô hình."""
    
    def do(self, interventions, n_samples=1000):
        ci = CausalInference(self)
        return ci.do(interventions, n_samples)
    
    def ate(self, treatment, outcome, **kwargs):
        ci = CausalInference(self)
        return ci.ate(treatment, outcome, **kwargs)
    
    def counterfactual(self, factual, interventions):
        ci = CausalInference(self)
        return ci.counterfactual(factual, interventions)
    
    def get_causal_inference(self):
        return CausalInference(self)
    
    model.do = lambda *args, **kwargs: do(model, *args, **kwargs)
    model.ate = lambda *args, **kwargs: ate(model, *args, **kwargs)
    model.counterfactual = lambda *args, **kwargs: counterfactual(model, *args, **kwargs)
    model.get_causal_inference = lambda: CausalInference(model)
    
    return model
