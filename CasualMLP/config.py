"""
CausalMLP Configuration / Cấu hình CausalMLP

Centralized configuration with sensible defaults based on analysis of DECI and GraN-DAG.
Cấu hình tập trung với các giá trị mặc định hợp lý dựa trên phân tích của DECI và GraN-DAG.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import torch


@dataclass
class CausalMLPConfig:
    """
    Configuration for CausalMLP model.
    Cấu hình cho mô hình CausalMLP.
    
    Contains all hyperparameters with sensible defaults derived from empirical analysis of both Causica/DECI and GraN-DAG.
    Chứa tất cả các siêu tham số với các giá trị mặc định hợp lý rút ra từ phân tích thực nghiệm của cả Causica/DECI và GraN-DAG.
    """
    
    # ========================
    # DATA / DỮ LIỆU
    # ========================
    num_nodes: int = 10
    
    # ========================
    # ARCHITECTURE / KIẾN TRÚC
    # ========================
    # MLP architecture (GraN-DAG style for interpretability) / Kiến trúc MLP (kiểu GraN-DAG để dễ diễn giải)
    hidden_dim: int = 64
    num_layers: int = 2
    activation: Literal['leaky_relu', 'relu', 'gelu'] = 'leaky_relu'
    use_layer_norm: bool = True  # From DECI - critical for stability / Từ DECI - quan trọng cho sự ổn định
    use_residual: bool = True    # From DECI - helps deep networks / Từ DECI - giúp ích cho mạng sâu
    dropout: float = 0.0         # For uncertainty via MC dropout / Cho độ không chắc chắn qua MC dropout
    
    # Embedding (from DECI - optional, enables transfer learning) / Embedding (từ DECI - tùy chọn, cho phép học chuyển đổi)
    use_embeddings: bool = False  # Default off for simplicity / Mặc định tắt cho đơn giản
    embedding_dim: int = 16
    
    # ========================
    # ADJACENCY / MA TRẬN KỀ
    # ========================
    # Dual-head paradigm: ENCO for learning + path weights for interpretation
    # Mô hình hai đầu: ENCO để học + trọng số đường dẫn để diễn giải
    adjacency_type: Literal['soft', 'enco', 'dual'] = 'soft'
    init_edge_prob: float = 0.3  # Initial probability of edges / Xác suất ban đầu của các cạnh
    
    # ========================
    # NOISE MODEL / MÔ HÌNH NHIỄU
    # ========================
    noise_type: Literal['gaussian', 'heteroscedastic', 'adaptive'] = 'gaussian'
    min_std: float = 0.01
    max_std: float = 2.0
    
    # ========================
    # DAG CONSTRAINT / RÀNG BUỘC DAG
    # ========================
    dag_type: Literal['exact', 'polynomial'] = 'exact'  # GPU matrix_exp
    h_tol: float = 1e-8  # Convergence threshold / Ngưỡng hội tụ
    
    # ========================
    # TRAINING - CURRICULUM / HUẤN LUYỆN - CHƯƠNG TRÌNH
    # ========================
    # Phase 1: Warm-up (NLL only, no DAG constraint) / Giai đoạn 1: Khởi động (chỉ NLL, không ràng buộc DAG)
    warmup_steps: int = 2000
    warmup_lr: float = 0.003
    
    # Phase 2-3: Main training with AugLag / Giai đoạn 2-3: Huấn luyện chính với AugLag
    main_lr: float = 0.002
    lr_decay: float = 0.95
    
    # Augmented Lagrangian
    init_alpha: float = 0.0
    init_rho: float = 0.01
    rho_mult: float = 2.0  # Slower than DECI's 10x / Chậm hơn 10x của DECI
    rho_max: float = 1e4   # Cap to prevent over-pruning / Giới hạn để tránh cắt tỉa quá mức
    alpha_max: float = 1e10
    
    inner_steps: int = 500
    max_outer_iter: int = 30
    
    # Regularization / Điều chuẩn
    sparsity_lambda: float = 0.001
    l2_reg: float = 0.0
    
    # Gumbel-Softmax
    gumbel_temp_init: float = 1.0
    gumbel_temp_min: float = 0.1
    anneal_temp: bool = True
    
    # ========================
    # OPTIMIZATION / TỐI ƯU HÓA
    # ========================
    batch_size: int = 256
    grad_clip: float = 5.0
    optimizer: Literal['adam', 'adamw'] = 'adam'
    weight_decay: float = 0.0
    
    # Early stopping / Dừng sớm
    patience: int = 200
    min_delta: float = 1e-4
    
    # ========================
    # INFERENCE / SUY LUẬN
    # ========================
    # For interventions and uncertainty / Cho can thiệp và độ không chắc chắn
    n_mc_samples: int = 100
    
    # ========================
    # DEVICE / THIẾT BỊ
    # ========================
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========================
    # PRIOR KNOWLEDGE / KIẾN THỨC TIÊN NGHIỆM
    # ========================
    use_prior: bool = False
    prior_weight: float = 10.0
    
    def __post_init__(self):
        """Validate configuration. / Kiểm tra cấu hình."""
        assert self.num_nodes > 0, "num_nodes must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert 0 < self.init_edge_prob < 1, "init_edge_prob must be in (0, 1)"
        assert self.rho_mult > 1, "rho_mult must be > 1"
    
    def to_dict(self):
        """Convert to dictionary. / Chuyển đổi sang từ điển."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary. / Tạo từ từ điển."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def for_sachs(cls):
        """Preset for Sachs dataset (11 nodes). / Cài đặt sẵn cho tập dữ liệu Sachs (11 nút)."""
        return cls(
            num_nodes=11,
            hidden_dim=48,
            num_layers=2,
            warmup_steps=2000,
            max_outer_iter=25,
            sparsity_lambda=0.0005,
        )
    
    @classmethod
    def for_small(cls, num_nodes: int = 10):
        """Preset for small graphs (<20 nodes). / Cài đặt sẵn cho đồ thị nhỏ (<20 nút)."""
        return cls(
            num_nodes=num_nodes,
            hidden_dim=64,
            num_layers=2,
            warmup_steps=1500,
        )
    
    @classmethod
    def for_medium(cls, num_nodes: int = 50):
        """Preset for medium graphs (20-100 nodes). / Cài đặt sẵn cho đồ thị trung bình (20-100 nút)."""
        return cls(
            num_nodes=num_nodes,
            hidden_dim=128,
            num_layers=3,
            warmup_steps=3000,
            batch_size=512,
            use_embeddings=True,
        )
    
    @classmethod
    def for_large(cls, num_nodes: int = 200):
        """Preset for large graphs (100+ nodes). / Cài đặt sẵn cho đồ thị lớn (100+ nút)."""
        return cls(
            num_nodes=num_nodes,
            hidden_dim=256,
            num_layers=3,
            warmup_steps=5000,
            batch_size=1024,
            use_embeddings=True,
            dag_type='polynomial',  # Faster for large graphs / Nhanh hơn cho đồ thị lớn
        )
