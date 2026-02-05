"""
Variational Inference for CausalMLP / Suy luận Biến thiên cho CausalMLP

Implements Bayesian posterior inference over causal graphs:
Triển khai suy luận hậu nghiệm Bayes trên đồ thị nhân quả:
- Variational approximation to posterior P(G|D) / Xấp xỉ biến thiên cho hậu nghiệm P(G|D)
- Stochastic Variational Inference (SVI) / Suy luận biến thiên ngẫu nhiên (SVI)
- Evidence Lower Bound (ELBO) optimization / Tối ưu hóa giới hạn dưới của bằng chứng (ELBO)

Reference: 
- Lorch et al., "DiBS: Differentiable Bayesian Structure Learning" (NeurIPS 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
import math


class GraphPrior(nn.Module):
    """
    Prior distribution over DAGs.
    Phân phối tiên nghiệm trên các DAG.
    
    Implements / Triển khai:
    - Erdos-Renyi prior (uniform edge probability) / Tiên nghiệm Erdos-Renyi (xác suất cạnh đồng đều)
    - Scale-free prior (preferential attachment) / Tiên nghiệm phi tỷ lệ (gắn kết ưu tiên)
    - Sparse prior (low density) / Tiên nghiệm thưa (mật độ thấp)
    """
    
    def __init__(
        self,
        num_nodes: int,
        prior_type: str = 'erdos_renyi',
        sparsity: float = 0.2,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.prior_type = prior_type
        self.sparsity = sparsity
        
        # Prior edge probability / Xác suất cạnh tiên nghiệm
        if prior_type == 'erdos_renyi':
            self.edge_prior = sparsity
        elif prior_type == 'sparse':
            self.edge_prior = 2.0 / num_nodes  # Expected 2 edges per node / Kỳ vọng 2 cạnh mỗi nút
        else:
            self.edge_prior = sparsity
    
    def log_prob(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Compute log prior probability of adjacency.
        Tính log xác suất tiên nghiệm của ma trận kề.
        
        Args:
            adjacency: Soft adjacency matrix (num_nodes, num_nodes) / Ma trận kề mềm
            
        Returns:
            Log prior probability / Log xác suất tiên nghiệm
        """
        n = self.num_nodes
        p = self.edge_prior
        
        # Bernoulli prior on each edge / Tiên nghiệm Bernoulli trên mỗi cạnh
        # log P(A) = sum_ij [ A_ij * log(p) + (1-A_ij) * log(1-p) ]
        log_prior = (
            adjacency * math.log(p + 1e-10) 
            + (1 - adjacency) * math.log(1 - p + 1e-10)
        )
        
        # Remove diagonal / Loại bỏ đường chéo
        mask = 1 - torch.eye(n, device=adjacency.device)
        log_prior = log_prior * mask
        
        return log_prior.sum()


class VariationalPosterior(nn.Module):
    """
    Variational posterior over DAGs.
    Hậu nghiệm biến thiên trên các DAG.
    
    Q(G) is parameterized as a product of independent Bernoulli:
    Q(G) được tham số hóa dưới dạng tích của các Bernoulli độc lập:
    Q(G) = prod_ij Bernoulli(σ(θ_ij))
    """
    
    def __init__(
        self,
        num_nodes: int,
        init_logit: float = -2.0,  # Sparse initialization / Khởi tạo thưa
        temperature: float = 0.5,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.temperature = temperature
        
        # Variational parameters (logits) / Các tham số biến thiên (logits)
        self.logits = nn.Parameter(
            init_logit * torch.ones(num_nodes, num_nodes)
        )
        
        self.register_buffer('diag_mask', 1 - torch.eye(num_nodes))
    
    @property
    def edge_probs(self) -> torch.Tensor:
        """Get edge probabilities. / Lấy xác suất cạnh."""
        return torch.sigmoid(self.logits) * self.diag_mask
    
    def sample(self, n_samples: int = 1, hard: bool = False) -> torch.Tensor:
        """
        Sample from variational posterior using Gumbel-Softmax.
        Lấy mẫu từ hậu nghiệm biến thiên sử dụng Gumbel-Softmax.
        
        Args:
            n_samples: Number of samples / Số lượng mẫu
            hard: Use hard (discrete) samples / Sử dụng mẫu cứng (rời rạc)
            
        Returns:
            Samples of shape (n_samples, num_nodes, num_nodes)
        """
        samples = []
        
        for _ in range(n_samples):
            u = torch.rand_like(self.logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            
            soft = torch.sigmoid((self.logits + gumbel) / self.temperature)
            soft = soft * self.diag_mask
            
            if hard:
                hard_sample = (soft > 0.5).float()
                sample = (hard_sample - soft).detach() + soft
            else:
                sample = soft
            
            samples.append(sample)
        
        return torch.stack(samples)
    
    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of variational posterior.
        Tính entropy của hậu nghiệm biến thiên.
        
        H(Q) = -E_Q[log Q(G)]
        For Bernoulli: H = -p*log(p) - (1-p)*log(1-p)
        """
        p = self.edge_probs
        
        entropy = -(
            p * torch.log(p + 1e-10) 
            + (1 - p) * torch.log(1 - p + 1e-10)
        )
        
        return entropy.sum()
    
    def kl_divergence(self, prior: GraphPrior) -> torch.Tensor:
        """
        Compute KL divergence KL(Q||P).
        Tính phân kỳ KL KL(Q||P).
        
        KL = E_Q[log Q(G)] - E_Q[log P(G)]
        """
        p = self.edge_probs
        p_prior = prior.edge_prior
        
        # KL for Bernoulli / KL cho Bernoulli
        kl = (
            p * torch.log((p + 1e-10) / (p_prior + 1e-10))
            + (1 - p) * torch.log((1 - p + 1e-10) / (1 - p_prior + 1e-10))
        )
        
        kl = kl * self.diag_mask
        
        return kl.sum()


class BayesianCausalMLP(nn.Module):
    """
    Bayesian Causal MLP with variational inference.
    Causal MLP Bayes với suy luận biến thiên.
    
    Optimizes ELBO / Tối ưu hóa ELBO:
    ELBO = E_Q[log P(D|G)] - KL(Q(G)||P(G))
    
    Where / Trong đó:
    - Q(G) is variational posterior over graphs / Q(G) là hậu nghiệm biến thiên trên các đồ thị
    - P(G) is prior (e.g., Erdos-Renyi) / P(G) là tiên nghiệm (ví dụ: Erdos-Renyi)
    - P(D|G) is likelihood given graph / P(D|G) là likelihood khi biết đồ thị
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        prior_sparsity: float = 0.2,
        temperature: float = 0.5,
        n_mc_samples: int = 5,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.n_mc_samples = n_mc_samples
        
        # Prior / Tiên nghiệm
        self.prior = GraphPrior(num_nodes, sparsity=prior_sparsity)
        
        # Variational posterior / Hậu nghiệm biến thiên
        self.posterior = VariationalPosterior(num_nodes, temperature=temperature)
        
        # MLP (shared across graphs) / MLP (chia sẻ qua các đồ thị)
        self._build_mlp(num_nodes, hidden_dim, num_layers)
    
    def _build_mlp(self, num_nodes, hidden_dim, num_layers):
        """Build the causal MLP. / Xây dựng causal MLP."""
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        dims = [num_nodes] + [hidden_dim] * num_layers + [2]
        
        for layer in range(len(dims) - 1):
            weight = torch.empty(num_nodes, dims[layer + 1], dims[layer])
            nn.init.xavier_uniform_(weight.view(-1, dims[layer]))
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(num_nodes, dims[layer + 1])))
    
    def log_likelihood(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log likelihood P(x|G).
        Tính log likelihood P(x|G).
        """
        batch_size = x.shape[0]
        
        # Forward through MLP / Lan truyền qua MLP
        masked = adjacency.unsqueeze(0) * x.unsqueeze(1)
        h = torch.einsum('nhj,bnj->bnh', self.weights[0], masked)
        h = h + self.biases[0].unsqueeze(0)
        h = F.leaky_relu(h, 0.2)
        
        for layer in range(1, len(self.weights) - 1):
            h = torch.einsum('noh,bnh->bno', self.weights[layer], h)
            h = h + self.biases[layer].unsqueeze(0)
            h = F.leaky_relu(h, 0.2)
        
        h = torch.einsum('noh,bnh->bno', self.weights[-1], h)
        h = h + self.biases[-1].unsqueeze(0)
        
        means = h[:, :, 0]
        log_stds = h[:, :, 1]
        
        # Gaussian log likelihood
        var = torch.exp(2 * log_stds).clamp(0.01, 4.0)
        log_lik = -0.5 * ((x - means) ** 2 / var + torch.log(var) + np.log(2 * np.pi))
        
        return log_lik.sum(dim=1)
    
    def dag_constraint(self, adjacency: torch.Tensor) -> torch.Tensor:
        """DAG constraint. / Ràng buộc DAG."""
        n = self.num_nodes
        A = adjacency * adjacency
        E = torch.linalg.matrix_exp(A)
        return torch.trace(E) - n
    
    def elbo(
        self,
        x: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Evidence Lower Bound.
        Tính Giới hạn dưới của Bằng chứng (ELBO).
        
        ELBO = E_Q[log P(D|G)] - beta * KL(Q||P)
        
        Args:
            x: Data / Dữ liệu
            beta: KL annealing weight / Trọng số KL annealing
        """
        # Sample graphs from posterior / Lấy mẫu đồ thị từ hậu nghiệm
        graphs = self.posterior.sample(self.n_mc_samples)
        
        # Monte Carlo estimate of expected log likelihood
        # Ước lượng Monte Carlo của log likelihood kỳ vọng
        log_liks = []
        dag_penalties = []
        
        for g in graphs:
            log_lik = self.log_likelihood(x, g)
            log_liks.append(log_lik.mean())
            dag_penalties.append(self.dag_constraint(g))
        
        expected_log_lik = torch.stack(log_liks).mean()
        expected_dag = torch.stack(dag_penalties).mean()
        
        # KL divergence / Phân kỳ KL
        kl = self.posterior.kl_divergence(self.prior)
        
        # ELBO
        elbo = expected_log_lik - beta * kl
        
        return {
            'elbo': elbo,
            'log_likelihood': expected_log_lik,
            'kl': kl,
            'dag_constraint': expected_dag,
            '-elbo': -elbo + 10 * expected_dag,  # Loss to minimize / Mất mát để tối thiểu hóa
        }
    
    def get_posterior_mean(self) -> torch.Tensor:
        """Get posterior mean graph. / Lấy đồ thị trung bình hậu nghiệm."""
        return self.posterior.edge_probs
    
    def sample_graphs(self, n_samples: int = 100) -> torch.Tensor:
        """Sample graphs from posterior. / Lấy mẫu đồ thị từ hậu nghiệm."""
        return self.posterior.sample(n_samples, hard=True)
    
    def credible_edges(
        self,
        threshold: float = 0.9,
    ) -> List[Tuple[int, int, float]]:
        """
        Get edges with high posterior probability.
        Lấy các cạnh có xác suất hậu nghiệm cao.
        
        Args:
            threshold: Minimum posterior probability / Xác suất hậu nghiệm tối thiểu
            
        Returns:
            List of (source, target, probability) / Danh sách (nguồn, đích, xác suất)
        """
        probs = self.posterior.edge_probs.detach().cpu().numpy()
        
        edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and probs[i, j] >= threshold:
                    edges.append((i, j, probs[i, j]))
        
        edges.sort(key=lambda x: -x[2])
        return edges


class VariationalTrainer:
    """
    Trainer for Bayesian Causal MLP.
    Bộ huấn luyện cho Causal MLP Bayes.
    """
    
    def __init__(
        self,
        model: BayesianCausalMLP,
        lr: float = 0.003,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def fit(
        self,
        data: torch.Tensor,
        n_epochs: int = 2000,
        batch_size: int = 256,
        beta_schedule: str = 'linear',  # KL annealing
        verbose: bool = True,
    ):
        """
        Train with variational inference.
        Huấn luyện với suy luận biến thiên.
        """
        n_samples = data.shape[0]
        
        for epoch in range(n_epochs):
            # KL annealing
            if beta_schedule == 'linear':
                beta = min(1.0, epoch / (n_epochs * 0.5))
            else:
                beta = 1.0
            
            self.optimizer.zero_grad()
            
            idx = torch.randint(0, n_samples, (batch_size,))
            batch = data[idx]
            
            result = self.model.elbo(batch, beta=beta)
            loss = result['-elbo']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            if verbose and (epoch + 1) % 200 == 0:
                with torch.no_grad():
                    probs = self.model.get_posterior_mean()
                    n_edges = (probs > 0.5).sum().item()
                
                print(f"Epoch {epoch + 1}: ELBO={result['elbo'].item():.2f}, "
                      f"KL={result['kl'].item():.2f}, h={result['dag_constraint'].item():.2f}, "
                      f"edges={n_edges}")
