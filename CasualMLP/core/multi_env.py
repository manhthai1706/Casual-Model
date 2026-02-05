"""
Multi-Environment Learning for CausalMLP / Học đa môi trường cho CausalMLP

Learn causal structure from multiple datasets/environments:
Học cấu trúc nhân quả từ nhiều bộ dữ liệu/môi trường:
- Observational + interventional data / Dữ liệu quan sát + can thiệp
- Multi-site studies / Các nghiên cứu đa địa điểm
- Pooled data with environment indicators / Dữ liệu gộp với chỉ số môi trường

Based on concepts from: / Dựa trên các khái niệm từ:
- ICP (Invariant Causal Prediction) / Dự đoán nhân quả bất biến
- DECI's multi-environment support / Hỗ trợ đa môi trường của DECI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class Environment:
    """
    Represents a single environment/dataset.
    Đại diện cho một môi trường/bộ dữ liệu đơn lẻ.
    
    Attributes / Các thuộc tính:
        data: Observations (n_samples, n_nodes) / Các quan sát
        interventions: Dict of {node: value} for hard interventions / Can thiệp cứng
        soft_interventions: Dict of {node: (shift, scale)} for soft interventions / Can thiệp mềm
        name: Environment identifier / Định danh môi trường
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        interventions: Optional[Dict[int, float]] = None,
        soft_interventions: Optional[Dict[int, Tuple[float, float]]] = None,
        name: str = 'env',
    ):
        self.data = data
        self.interventions = interventions or {}
        self.soft_interventions = soft_interventions or {}
        self.name = name
        self.n_samples = data.shape[0]
        self.n_nodes = data.shape[1]
    
    @property
    def intervention_mask(self) -> torch.Tensor:
        """
        Binary mask indicating intervened nodes.
        Mặt nạ nhị phân chỉ định các nút bị can thiệp.
        """
        mask = torch.zeros(self.n_nodes)
        for node in self.interventions:
            mask[node] = 1
        for node in self.soft_interventions:
            mask[node] = 1
        return mask
    
    def __len__(self):
        return self.n_samples


class MultiEnvironmentDataset:
    """
    Collection of environments for multi-environment learning.
    Tập hợp các môi trường cho việc học đa môi trường.
    """
    
    def __init__(self, environments: List[Environment] = None):
        self.environments = environments or []
        self._validate()
    
    def _validate(self):
        """Ensure all environments have same number of nodes. / Đảm bảo tất cả môi trường có cùng số lượng nút."""
        if len(self.environments) > 1:
            n_nodes = self.environments[0].n_nodes
            for env in self.environments[1:]:
                assert env.n_nodes == n_nodes, "All environments must have same number of nodes"
    
    def add_environment(self, env: Environment):
        """Add an environment. / Thêm một môi trường."""
        self.environments.append(env)
        self._validate()
    
    @property
    def n_environments(self) -> int:
        return len(self.environments)
    
    @property
    def n_nodes(self) -> int:
        if self.environments:
            return self.environments[0].n_nodes
        return 0
    
    @property
    def total_samples(self) -> int:
        return sum(len(env) for env in self.environments)
    
    def get_pooled_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all data pooled together with environment indicators.
        Lấy tất cả dữ liệu gộp lại cùng với chỉ số môi trường.
        
        Returns:
            (data, env_indices) where env_indices[i] = environment of sample i
            (dữ liệu, chỉ số môi trường) nơi env_indices[i] = môi trường của mẫu i
        """
        all_data = []
        all_indices = []
        
        for env_idx, env in enumerate(self.environments):
            all_data.append(env.data)
            all_indices.extend([env_idx] * len(env))
        
        return torch.cat(all_data, dim=0), torch.tensor(all_indices)
    
    def get_intervention_targets(self) -> torch.Tensor:
        """
        Get intervention target matrix.
        Lấy ma trận mục tiêu can thiệp.
        
        Returns:
            Matrix (n_environments, n_nodes) where [e, i] = 1 if node i is intervened in env e
            Ma trận trong đó [e, i] = 1 nếu nút i bị can thiệp trong môi trường e
        """
        targets = torch.zeros(self.n_environments, self.n_nodes)
        
        for e, env in enumerate(self.environments):
            targets[e] = env.intervention_mask
        
        return targets
    
    @classmethod
    def from_interventional_data(
        cls,
        observational: torch.Tensor,
        interventional: List[Tuple[torch.Tensor, Dict[int, float]]],
    ) -> 'MultiEnvironmentDataset':
        """
        Create dataset from observational + interventional data.
        Tạo bộ dữ liệu từ dữ liệu quan sát + can thiệp.
        
        Args:
            observational: Observational data / Dữ liệu quan sát
            interventional: List of (data, interventions_dict) tuples / Danh sách các bộ (dữ liệu, từ điển can thiệp)
        """
        envs = [Environment(observational, name='observational')]
        
        for i, (data, interventions) in enumerate(interventional):
            envs.append(Environment(data, interventions, name=f'intervention_{i}'))
        
        return cls(envs)


class MultiEnvironmentModel(nn.Module):
    """
    Causal model that learns from multiple environments.
    Mô hình nhân quả học từ nhiều môi trường.
    
    Key ideas / Ý tưởng chính:
    1. Shared causal structure across environments / Chia sẻ cấu trúc nhân quả giữa các môi trường
    2. Environment-specific noise models (optional) / Mô hình nhiễu đặc thù cho từng môi trường (tùy chọn)
    3. Intervention likelihoods that account for broken mechanisms / Likelihood can thiệp tính đến các cơ chế bị phá vỡ
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        share_noise: bool = True,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.share_noise = share_noise
        
        # Shared adjacency / Ma trận kề chia sẻ
        self.adjacency_logits = nn.Parameter(
            -2.0 * torch.ones(num_nodes, num_nodes)
        )
        self.register_buffer('diag_mask', 1 - torch.eye(num_nodes))
        
        # Shared MLP / MLP chia sẻ
        self._build_mlp(num_nodes, hidden_dim, num_layers)
        
        # Noise parameters (shared or per-environment)
        # Tham số nhiễu (chia sẻ hoặc theo môi trường)
        self.log_std = nn.Parameter(torch.zeros(num_nodes))
        
        # Environment-specific parameters (added dynamically)
        # Tham số đặc thù môi trường (thêm động)
        self.env_log_stds: Dict[str, nn.Parameter] = {}
    
    def _build_mlp(self, num_nodes, hidden_dim, num_layers):
        """Build shared MLP. / Xây dựng MLP chia sẻ."""
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        dims = [num_nodes] + [hidden_dim] * num_layers + [1]
        
        for layer in range(len(dims) - 1):
            weight = torch.empty(num_nodes, dims[layer + 1], dims[layer])
            nn.init.xavier_uniform_(weight.view(-1, dims[layer]))
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(num_nodes, dims[layer + 1])))
    
    @property 
    def adjacency(self) -> torch.Tensor:
        """Get soft adjacency matrix. / Lấy ma trận kề mềm."""
        return torch.sigmoid(self.adjacency_logits) * self.diag_mask
    
    def predict(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict node values given inputs.
        Dự đoán giá trị nút dựa vào đầu vào.
        
        Args:
            x: Input (batch, num_nodes) / Đầu vào
            adjacency: Optional custom adjacency / Ma trận kề tùy chỉnh (tùy chọn)
            
        Returns:
            Predicted means (batch, num_nodes) / Trung bình dự đoán
        """
        if adjacency is None:
            adjacency = self.adjacency
        
        # Mask input by adjacency / Che đầu vào bằng ma trận kề
        masked = adjacency.unsqueeze(0) * x.unsqueeze(1)  # (B, N, N)
        
        h = torch.einsum('nhj,bnj->bnh', self.weights[0], masked)
        h = h + self.biases[0].unsqueeze(0)
        h = F.leaky_relu(h, 0.2)
        
        for layer in range(1, len(self.weights)):
            h = torch.einsum('noh,bnh->bno', self.weights[layer], h)
            h = h + self.biases[layer].unsqueeze(0)
            if layer < len(self.weights) - 1:
                h = F.leaky_relu(h, 0.2)
        
        return h.squeeze(-1)
    
    def log_likelihood_single_env(
        self,
        x: torch.Tensor,
        intervention_mask: torch.Tensor,
        env_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute log likelihood for a single environment.
        Tính log likelihood cho một môi trường đơn lẻ.
        
        Intervened nodes have their mechanisms "broken" - we don't model them causally, only their marginal distribution.
        Các nút bị can thiệp có cơ chế bị "phá vỡ" - chúng ta không mô hình hóa chúng theo nhân quả, chỉ phân phối biên của chúng.
        
        Args:
            x: Data / Dữ liệu
            intervention_mask: Binary mask of intervened nodes / Mặt nạ nhị phân của các nút bị can thiệp
            env_name: Optional environment name / Tên môi trường tùy chọn
        """
        adjacency = self.adjacency
        
        # For intervened nodes: set incoming edges to 0
        # Đối với nút bị can thiệp: đặt cạnh đến bằng 0
        intervened_adj = adjacency.clone()
        for i in range(self.num_nodes):
            if intervention_mask[i] > 0.5:
                intervened_adj[:, i] = 0
        
        # Predict means / Dự đoán trung bình
        means = self.predict(x, intervened_adj)
        
        # Get noise std / Lấy độ lệch chuẩn nhiễu
        log_std = self.log_std
        std = torch.exp(log_std).clamp(0.01, 2.0)
        
        # Gaussian log likelihood
        log_lik = -0.5 * ((x - means) ** 2 / (std ** 2) + 2 * log_std + np.log(2 * np.pi))
        
        # For intervened nodes: use marginal likelihood (just Gaussian with observed mean/std)
        # Đối với nút bị can thiệp: sử dụng log likelihood biên (chỉ là Gaussian với trung bình/độ lệch chuẩn quan sát)
        # This avoids penalizing model for not matching the intervention mechanism
        # Điều này tránh trừng phạt mô hình vì không khớp với cơ chế can thiệp
        for i in range(self.num_nodes):
            if intervention_mask[i] > 0.5:
                # Use empirical mean/std of intervened node
                node_mean = x[:, i].mean()
                node_std = x[:, i].std() + 1e-8
                log_lik[:, i] = -0.5 * ((x[:, i] - node_mean) ** 2 / (node_std ** 2) 
                                        + 2 * torch.log(node_std) + np.log(2 * np.pi))
        
        return log_lik.sum(dim=1)
    
    def loss_multi_env(
        self,
        dataset: MultiEnvironmentDataset,
        batch_size: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss across all environments.
        Tính toán mất mát trên tất cả các môi trường.
        
        Returns:
            Dict with total loss and per-environment losses
            Từ điển chứa tổng mất mát và mất mát theo từng môi trường
        """
        total_nll = 0.0
        env_losses = {}
        
        for env in dataset.environments:
            # Sample batch / Lấy mẫu lô
            n = len(env)
            idx = torch.randint(0, n, (min(batch_size, n),))
            batch = env.data[idx]
            
            # Compute log likelihood / Tính log likelihood
            log_lik = self.log_likelihood_single_env(
                batch,
                env.intervention_mask,
                env.name
            )
            
            nll = -log_lik.mean()
            total_nll = total_nll + nll
            env_losses[env.name] = nll
        
        # DAG constraint / Ràng buộc DAG
        A = self.adjacency
        h = torch.trace(torch.linalg.matrix_exp(A * A)) - self.num_nodes
        
        # Sparsity / Tính thưa thớt
        sparsity = A.abs().sum()
        
        loss = total_nll + 10 * h + 0.001 * sparsity
        
        return {
            'loss': loss,
            'nll': total_nll,
            'h': h,
            'env_losses': env_losses,
        }


class InvariantCausalPrediction:
    """
    Invariant Causal Prediction (ICP) style inference.
    Suy luận kiểu Dự đoán Nhân quả Bất biến (ICP).
    
    Uses multiple environments to identify causal parents:
    Sử dụng nhiều môi trường để xác định cha mẹ nhân quả:
    - True causal parents should give invariant predictions across environments
      Cha mẹ nhân quả thực sự nên đưa ra dự đoán bất biến qua các môi trường
    - Spurious correlations will vary across environments
      Các tương quan giả mạo sẽ thay đổi qua các môi trường
    """
    
    def __init__(
        self,
        dataset: MultiEnvironmentDataset,
        target_node: int,
        alpha: float = 0.05,
    ):
        self.dataset = dataset
        self.target_node = target_node
        self.alpha = alpha
    
    def test_parent_set(
        self,
        parent_set: List[int],
    ) -> Dict[str, float]:
        """
        Test if a parent set gives invariant predictions.
        Kiểm tra xem một tập cha mẹ có đưa ra dự đoán bất biến không.
        
        Uses residual variance test across environments.
        Sử dụng kiểm định phương sai phần dư qua các môi trường.
        """
        from scipy import stats
        
        residual_variances = []
        
        for env in self.dataset.environments:
            data = env.data.numpy()
            
            y = data[:, self.target_node]
            
            if len(parent_set) == 0:
                residuals = y - y.mean()
            else:
                X = data[:, parent_set]
                # Linear regression / Hồi quy tuyến tính
                X_with_bias = np.column_stack([np.ones(len(X)), X])
                try:
                    beta = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
                    residuals = y - X_with_bias @ beta
                except:
                    residuals = y - y.mean()
            
            residual_variances.append(np.var(residuals))
        
        # Levene's test for equality of variances
        # Kiểm định Levene cho sự bằng nhau của phương sai
        if len(residual_variances) > 1:
            # Create groups of residuals / Tạo các nhóm phần dư
            residual_groups = []
            for env in self.dataset.environments:
                data = env.data.numpy()
                y = data[:, self.target_node]
                if len(parent_set) == 0:
                    residuals = y - y.mean()
                else:
                    X = data[:, parent_set]
                    X_with_bias = np.column_stack([np.ones(len(X)), X])
                    try:
                        beta = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
                        residuals = y - X_with_bias @ beta
                    except:
                        residuals = y - y.mean()
                residual_groups.append(residuals)
            
            stat, p_value = stats.levene(*residual_groups)
        else:
            p_value = 1.0
            stat = 0.0
        
        return {
            'parent_set': parent_set,
            'p_value': p_value,
            'invariant': p_value >= self.alpha,
            'variances': residual_variances,
        }
    
    def find_causal_parents(
        self,
        max_parents: int = 5,
    ) -> Dict[str, any]:
        """
        Find causal parent set using ICP.
        Tìm tập cha mẹ nhân quả sử dụng ICP.
        
        Tests all subsets up to max_parents size.
        Kiểm tra tất cả các tập con có kích thước đến max_parents.
        """
        from itertools import combinations
        
        other_nodes = [i for i in range(self.dataset.n_nodes) if i != self.target_node]
        
        # Test empty set / Kiểm tra tập rỗng
        results = [self.test_parent_set([])]
        
        # Test all subsets / Kiểm tra tất cả tập con
        for size in range(1, min(max_parents + 1, len(other_nodes) + 1)):
            for subset in combinations(other_nodes, size):
                result = self.test_parent_set(list(subset))
                results.append(result)
        
        # Find all invariant sets / Tìm tất cả các tập bất biến
        invariant_sets = [r for r in results if r['invariant']]
        
        # Intersection of all invariant sets = causal parents
        # Giao của tất cả các tập bất biến = cha mẹ nhân quả
        if invariant_sets:
            causal_parents = set(range(self.dataset.n_nodes))
            for r in invariant_sets:
                if len(r['parent_set']) > 0:
                    causal_parents &= set(r['parent_set'])
            causal_parents = list(causal_parents)
        else:
            causal_parents = []
        
        return {
            'causal_parents': causal_parents,
            'invariant_sets': invariant_sets,
            'all_results': results,
        }


class MultiEnvTrainer:
    """
    Trainer for multi-environment causal discovery.
    Bộ huấn luyện cho khám phá nhân quả đa môi trường.
    """
    
    def __init__(
        self,
        model: MultiEnvironmentModel,
        lr: float = 0.003,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def fit(
        self,
        dataset: MultiEnvironmentDataset,
        n_epochs: int = 2000,
        batch_size: int = 256,
        verbose: bool = True,
    ):
        """Train on multi-environment data. / Huấn luyện trên dữ liệu đa môi trường."""
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            result = self.model.loss_multi_env(dataset, batch_size)
            result['loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            if verbose and (epoch + 1) % 200 == 0:
                adj = self.model.adjacency
                n_edges = (adj > 0.5).sum().item()
                
                env_str = ", ".join([f"{k}={v.item():.2f}" for k, v in result['env_losses'].items()])
                print(f"Epoch {epoch + 1}: loss={result['loss'].item():.3f}, "
                      f"h={result['h'].item():.3f}, edges={n_edges}")
                print(f"  Env losses: {env_str}")
