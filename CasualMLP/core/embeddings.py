"""
Node Embeddings for CausalMLP / Node Embeddings cho CausalMLP

Learnable embeddings for each node that:
Các embedding có thể học được cho mỗi nút giúp:
- Capture node-specific characteristics / Nắm bắt các đặc tính cụ thể của nút
- Enable transfer learning across datasets / Cho phép học chuyển giao giữa các tập dữ liệu
- Improve performance on large graphs / Cải thiện hiệu suất trên đồ thị lớn

Based on DECI's embedding mechanism.
Dựa trên cơ chế embedding của DECI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import numpy as np


class NodeEmbeddings(nn.Module):
    """
    Learnable embeddings for graph nodes.
    Các embedding có thể học được cho các nút đồ thị.
    
    Each node gets a vector representation that captures:
    Mỗi nút nhận một biểu diễn vector nắm bắt:
    - Node type/category / Loại/danh mục nút
    - Statistical properties / Các thuộc tính thống kê
    - Role in the causal structure / Vai trò trong cấu trúc nhân quả
    """
    
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 32,
        init_std: float = 0.01,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # Main embeddings for each node
        # Embeddings chính cho mỗi nút
        self.embeddings = nn.Parameter(
            init_std * torch.randn(num_nodes, embedding_dim)
        )
        
        # Optional: learnable positional encoding
        # Tùy chọn: mã hóa vị trí có thể học được
        self.position_encoding = nn.Parameter(
            init_std * torch.randn(num_nodes, embedding_dim // 4)
        )
    
    def forward(self, node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings.
        Lấy các embedding.
        
        Args:
            node_indices: Optional indices (default: all nodes) / Các chỉ số tùy chọn (mặc định: tất cả các nút)
            
        Returns:
            Embeddings (num_nodes, embedding_dim) or (len(indices), embedding_dim)
        """
        if node_indices is None:
            return self.embeddings
        return self.embeddings[node_indices]
    
    def get_similarity_matrix(self) -> torch.Tensor:
        """
        Compute pairwise similarity between node embeddings.
        Tính độ tương đồng từng cặp giữa các embedding nút.
        
        Returns:
            Similarity matrix (num_nodes, num_nodes) / Ma trận tương đồng
        """
        # Normalize embeddings / Chuẩn hóa embedding
        normed = F.normalize(self.embeddings, dim=1)
        
        # Cosine similarity / Tương đồng Cosine
        similarity = normed @ normed.T
        
        return similarity
    
    def cluster_nodes(self, n_clusters: int = 3) -> torch.Tensor:
        """
        Cluster nodes based on embeddings (soft clustering).
        Phân cụm các nút dựa trên embedding (phân cụm mềm).
        
        Returns:
            Cluster assignments (num_nodes, n_clusters)
            Gán cụm
        """
        # Simple k-means style clustering / Phân cụm kiểu k-means đơn giản
        # Initialize cluster centers / Khởi tạo tâm cụm
        indices = torch.randperm(self.num_nodes)[:n_clusters]
        centers = self.embeddings[indices].detach()
        
        # Compute distances to centers / Tính khoảng cách đến các tâm
        emb = self.embeddings.unsqueeze(1)  # (N, 1, D)
        centers = centers.unsqueeze(0)       # (1, K, D)
        
        distances = ((emb - centers) ** 2).sum(dim=2)  # (N, K)
        
        # Soft assignments / Gán mềm
        assignments = F.softmax(-distances, dim=1)
        
        return assignments


class EmbeddedMLP(nn.Module):
    """
    MLP that uses node embeddings.
    MLP sử dụng embedding nút.
    
    For each node i, the prediction depends on:
    Đối với mỗi nút i, dự đoán phụ thuộc vào:
    - Masked input from parents / Đầu vào được che từ cha mẹ
    - Node embedding of i / Embedding nút của i
    - Optionally: embeddings of parents / Tùy chọn: embedding của cha mẹ
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 2,
        use_parent_embeddings: bool = True,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.use_parent_embeddings = use_parent_embeddings
        
        # Node embeddings / Embedding nút
        self.node_embeddings = NodeEmbeddings(num_nodes, embedding_dim)
        
        # Input dimension / Kích thước đầu vào
        if use_parent_embeddings:
            input_dim = num_nodes + embedding_dim + embedding_dim  # x + self_emb + parent_emb
        else:
            input_dim = num_nodes + embedding_dim  # x + self_emb
        
        # MLP layers / Các lớp MLP
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * num_layers + [2]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(nn.LayerNorm(dims[i + 1]))
                self.layers.append(nn.LeakyReLU(0.2))
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with embeddings.
        Lan truyền tiến với embedding.
        
        Args:
            x: Input (batch, num_nodes) / Đầu vào
            adjacency: Adjacency matrix (num_nodes, num_nodes) / Ma trận kề
            
        Returns:
            (means, log_stds)
            (trung bình, log độ lệch chuẩn)
        """
        batch_size = x.shape[0]
        embeddings = self.node_embeddings()  # (num_nodes, emb_dim)
        
        means_list = []
        log_stds_list = []
        
        for i in range(self.num_nodes):
            # Masked input (from parents)
            # Đầu vào được che (từ cha mẹ)
            masked_x = adjacency[:, i] * x  # (batch, num_nodes)
            
            # Self embedding / Embedding bản thân
            self_emb = embeddings[i:i+1].expand(batch_size, -1)
            
            if self.use_parent_embeddings:
                # Weighted average of parent embeddings
                # Trung bình có trọng số của embedding cha mẹ
                parent_weights = adjacency[:, i]  # (num_nodes,)
                parent_weights = parent_weights / (parent_weights.sum() + 1e-8)
                parent_emb = (parent_weights.unsqueeze(0) @ embeddings).expand(batch_size, -1)
                
                # Concatenate / Nối
                mlp_input = torch.cat([masked_x, self_emb, parent_emb], dim=1)
            else:
                mlp_input = torch.cat([masked_x, self_emb], dim=1)
            
            # Forward through MLP / Lan truyền qua MLP
            h = mlp_input
            for layer in self.layers:
                h = layer(h)
            
            means_list.append(h[:, 0])
            log_stds_list.append(h[:, 1])
        
        return torch.stack(means_list, dim=1), torch.stack(log_stds_list, dim=1)
    
    def get_node_representations(self) -> torch.Tensor:
        """Get learned node embeddings. / Lấy các embedding nút đã học."""
        return self.node_embeddings()
    
    def count_parameters(self) -> int:
        """Count parameters. / Đếm số lượng tham số."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EmbeddingTransfer:
    """
    Transfer embeddings between models/datasets.
    Chuyển giao embedding giữa các mô hình/bộ dữ liệu.
    
    Enables transfer learning:
    Cho phép học chuyển giao:
    1. Learn embeddings on source dataset / Học embedding trên tập dữ liệu nguồn
    2. Transfer to target dataset with similar nodes / Chuyển sang tập dữ liệu đích với các nút tương tự
    """
    
    def __init__(
        self,
        source_model,
        target_model,
    ):
        self.source_model = source_model
        self.target_model = target_model
    
    def transfer_by_similarity(
        self,
        source_names: List[str],
        target_names: List[str],
        similarity_threshold: float = 0.5,
    ):
        """
        Transfer embeddings based on node name similarity.
        Chuyển giao embedding dựa trên độ tương đồng tên nút.
        
        Uses simple string matching.
        Sử dụng so khớp chuỗi đơn giản.
        """
        source_emb = self.source_model.node_embeddings.embeddings.detach()
        
        for t_idx, t_name in enumerate(target_names):
            best_match = None
            best_score = 0
            
            for s_idx, s_name in enumerate(source_names):
                # Simple similarity: common prefix length
                # Độ tương đồng đơn giản: độ dài tiền tố chung
                score = self._string_similarity(s_name.lower(), t_name.lower())
                if score > best_score:
                    best_score = score
                    best_match = s_idx
            
            if best_match is not None and best_score >= similarity_threshold:
                with torch.no_grad():
                    self.target_model.node_embeddings.embeddings[t_idx] = source_emb[best_match]
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity (Jaccard on characters). / Tính độ tương đồng chuỗi (Jaccard trên các ký tự)."""
        set1 = set(s1)
        set2 = set(s2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0


class GraphEmbedding(nn.Module):
    """
    Compute a single embedding for the entire graph.
    Tính toán một embedding đơn cho toàn bộ đồ thị.
    
    Useful for: / Hữu ích cho:
    - Comparing different causal structures / So sánh các cấu trúc nhân quả khác nhau
    - Graph classification/retrieval / Phân loại/truy vấn đồ thị
    """
    
    def __init__(
        self,
        node_embeddings: NodeEmbeddings,
        aggregation: str = 'mean',
    ):
        super().__init__()
        
        self.node_embeddings = node_embeddings
        self.aggregation = aggregation
        
        # Learnable aggregation / Tổng hợp có thể học được
        self.attention = nn.Linear(node_embeddings.embedding_dim, 1)
    
    def forward(self, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute graph embedding.
        Tính embedding đồ thị.
        
        Args:
            adjacency: Optional adjacency to weight nodes by centrality.
                       Ma trận kề tùy chọn để trọng số hóa các nút theo độ trung tâm.
            
        Returns:
            Graph embedding (embedding_dim,)
        """
        node_emb = self.node_embeddings()  # (N, D)
        
        if self.aggregation == 'mean':
            return node_emb.mean(dim=0)
        
        elif self.aggregation == 'attention':
            weights = F.softmax(self.attention(node_emb), dim=0)
            return (weights * node_emb).sum(dim=0)
        
        elif self.aggregation == 'max':
            return node_emb.max(dim=0)[0]
        
        elif self.aggregation == 'sum':
            return node_emb.sum(dim=0)
        
        else:
            return node_emb.mean(dim=0)
