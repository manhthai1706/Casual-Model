"""
Curriculum Trainer for CausalMLP / Trình huấn luyện chương trình cho CausalMLP

Implements multi-phase curriculum training:
Triển khai huấn luyện chương trình đa giai đoạn:
- Phase 1: Warm-up (NLL only, learn correlations) / Giai đoạn 1: Khởi động (chỉ NLL, học tương quan)
- Phase 2: Soft DAG (gradually increase constraint) / Giai đoạn 2: Soft DAG (tăng dần ràng buộc)
- Phase 3: Hard DAG (full augmented Lagrangian) / Giai đoạn 3: Hard DAG (Lagrangian tăng cường đầy đủ)
- Phase 4: Refinement (fine-tune with low LR) / Giai đoạn 4: Tinh chỉnh (tinh chỉnh với LR thấp)

Novel contribution: Addresses both DECI and GraN-DAG's weakness of unstable early training.
Đóng góp mới lạ: Giải quyết điểm yếu của cả DECI và GraN-DAG về việc huấn luyện sớm không ổn định.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, List, Callable
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CausalMLPConfig
from core.model import CausalMLPModel
from utils.dag_utils import compute_metrics, to_dag


class CurriculumTrainer:
    """
    Multi-phase curriculum trainer for CausalMLP.
    Trình huấn luyện chương trình đa giai đoạn cho CausalMLP.
    
    Training phases / Các giai đoạn huấn luyện:
    1. Warm-up: Train NLL only (no DAG constraint) / Khởi động: Chỉ huấn luyện NLL (không ràng buộc DAG)
       - Allows model to learn data distribution / Cho phép mô hình học phân phối dữ liệu
       - Prevents early collapse of adjacency / Ngăn chặn sự sụp đổ sớm của ma trận kề
    
    2. Soft constraint: Gradually introduce DAG penalty / Ràng buộc mềm: Dần dần đưa vào phạt DAG
       - Start with small rho / Bắt đầu với rho nhỏ
       - Anneal Gumbel temperature / Ủ nhiệt độ Gumbel
    
    3. Hard constraint: Full augmented Lagrangian / Ràng buộc cứng: Lagrangian tăng cường đầy đủ
       - Increase rho until convergence / Tăng rho cho đến khi hội tụ
       - Update alpha after each outer iteration / Cập nhật alpha sau mỗi vòng lặp ngoài
    
    4. Refinement: Fine-tune with pruning / Tinh chỉnh: Tinh chỉnh với cắt tỉa
       - Lower learning rate / Giảm tốc độ học
       - Apply CAM-style pruning / Áp dụng cắt tỉa kiểu CAM
    """
    
    def __init__(
        self,
        model: CausalMLPModel,
        config: Optional[CausalMLPConfig] = None,
    ):
        self.model = model
        self.config = config or model.config
        self.device = self.config.device
        
        self.model.to(self.device)
        
        # Optimizer / Bộ tối ưu hóa
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.warmup_lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.warmup_lr,
                weight_decay=self.config.weight_decay,
            )
        
        # History / Lịch sử
        self.history: List[Dict] = []
        self.best_f1 = 0.0
        self.best_state = None
    
    def fit(
        self,
        data: torch.Tensor,
        true_adjacency: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model.
        Huấn luyện mô hình.
        
        Args:
            data: Training data (n_samples, num_nodes) / Dữ liệu huấn luyện
            true_adjacency: Optional ground truth for evaluation / Ground truth tùy chọn để đánh giá
            verbose: Print progress / In tiến trình
            
        Returns:
            Dictionary with training results
            Từ điển kết quả huấn luyện
        """
        data = data.to(self.device)
        if true_adjacency is not None:
            true_adjacency = true_adjacency.to(self.device)
        
        if verbose:
            print("=" * 60)
            print("CausalMLP Training")
            print("=" * 60)
            print(f"Samples: {data.shape[0]}, Nodes: {data.shape[1]}")
            print(f"Parameters: {self.model.count_parameters():,}")
        
        # Phase 1: Warm-up / Giai đoạn 1: Khởi động
        if self.config.warmup_steps > 0:
            self._phase_warmup(data, true_adjacency, verbose)
        
        # Phase 2-3: Main training / Giai đoạn 2-3: Huấn luyện chính
        self._phase_main(data, true_adjacency, verbose)
        
        # Restore best / Khôi phục tốt nhất
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return {
            'history': self.history,
            'best_f1': self.best_f1,
        }
    
    def _phase_warmup(
        self,
        data: torch.Tensor,
        true_adj: Optional[torch.Tensor],
        verbose: bool
    ):
        """Phase 1: Warm-up (NLL only). / Giai đoạn 1: Khởi động (chỉ NLL)."""
        if verbose:
            print("\n[PHASE 1] Warm-up (NLL only)...")
        
        # Disable DAG constraint / Vô hiệu hóa ràng buộc DAG
        self.model.auglag.alpha.zero_()
        self.model.auglag.rho.zero_()
        
        # Set learning rate / Đặt tốc độ học
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.warmup_lr
        
        best_nll = float('inf')
        patience_count = 0
        
        for step in range(self.config.warmup_steps):
            self.optimizer.zero_grad()
            
            # Mini-batch
            idx = torch.randint(0, len(data), (self.config.batch_size,))
            batch = data[idx]
            
            # Forward (NLL + sparsity only) / Lan truyền tiến (chỉ NLL + tính thưa thớt)
            adj = self.model.get_adjacency()
            nll = -self.model.compute_log_likelihood(batch, adj).mean()
            sparsity = self.model.compute_sparsity(adj)
            
            loss = nll + 0.0001 * sparsity  # Light sparsity / Tính thưa thớt nhẹ
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            # Update adaptive noise / Cập nhật nhiễu thích ứng
            if hasattr(self.model.noise, 'update_step'):
                self.model.noise.update_step()
            
            # Early stopping / Dừng sớm
            if nll.item() < best_nll - self.config.min_delta:
                best_nll = nll.item()
                patience_count = 0
            else:
                patience_count += 1
            
            if patience_count > self.config.patience:
                if verbose:
                    print(f"  Early stop at step {step}")
                break
            
            # Logging / Ghi nhật ký
            if verbose and step % max(1, self.config.warmup_steps // 5) == 0:
                with torch.no_grad():
                    h = self.model.compute_dag_constraint(adj).item()
                    edges = (adj > 0.5).sum().item()
                    
                    msg = f"  Step {step}: NLL={nll.item():.3f}, h={h:.1f}, edges={edges}"
                    
                    if true_adj is not None:
                        metrics = self.model.evaluate(true_adj, threshold=0.3)
                        msg += f", F1={metrics['f1']:.3f}"
                        
                        if metrics['f1'] > self.best_f1:
                            self.best_f1 = metrics['f1']
                            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    
                    print(msg)
        
        if verbose:
            print(f"  Warm-up complete. Best NLL: {best_nll:.3f}")
    
    def _phase_main(
        self,
        data: torch.Tensor,
        true_adj: Optional[torch.Tensor],
        verbose: bool
    ):
        """Phase 2-3: Main training with augmented Lagrangian. / Giai đoạn 2-3: Huấn luyện chính với Lagrangian tăng cường."""
        if verbose:
            print("\n[PHASE 2-3] Main Training (Augmented Lagrangian)...")
        
        # Initialize augmented Lagrangian / Khởi tạo Lagrangian tăng cường
        self.model.auglag.reset(alpha=0.0, rho=self.config.init_rho)
        
        # Set learning rate / Đặt tốc độ học
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.main_lr
        
        prev_h = float('inf')
        init_temp = self.config.gumbel_temp_init
        
        for outer in range(self.config.max_outer_iter):
            # Anneal temperature / Ủ nhiệt độ
            if self.config.anneal_temp:
                progress = outer / self.config.max_outer_iter
                temp = max(
                    self.config.gumbel_temp_min,
                    init_temp * (1 - 0.8 * progress)
                )
                self.model.set_temperature(temp)
            
            # Decay learning rate / Giảm tốc độ học
            if self.config.lr_decay < 1.0:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.main_lr * (self.config.lr_decay ** outer)
            
            # Inner optimization loop / Vòng lặp tối ưu hóa bên trong
            inner_losses = []
            
            for inner in range(self.config.inner_steps):
                self.optimizer.zero_grad()
                
                idx = torch.randint(0, len(data), (self.config.batch_size,))
                batch = data[idx]
                
                result = self.model(batch, return_components=True)
                loss = result['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                
                inner_losses.append(loss.item())
            
            # Evaluate / Đánh giá
            with torch.no_grad():
                adj = self.model.adjacency.probs
                h = self.model.compute_dag_constraint(adj).item()
                nll = -self.model.compute_log_likelihood(data[:500], adj).mean().item()
                
                record = {
                    'outer': outer,
                    'h': h,
                    'nll': nll,
                    'alpha': self.model.auglag.alpha.item(),
                    'rho': self.model.auglag.rho.item(),
                    'mean_loss': np.mean(inner_losses),
                }
                
                if true_adj is not None:
                    for thresh in [0.3, 0.5]:
                        metrics = self.model.evaluate(true_adj, threshold=thresh)
                        record[f'f1_{thresh}'] = metrics['f1']
                        record[f'shd_{thresh}'] = metrics['shd']
                    
                    # Track best / Theo dõi tốt nhất
                    f1 = metrics['f1']
                    if f1 > self.best_f1:
                        self.best_f1 = f1
                        self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                self.history.append(record)
                
                if verbose:
                    msg = f"Outer {outer+1}/{self.config.max_outer_iter}: "
                    msg += f"h={h:.2e}, alpha={record['alpha']:.1e}, rho={record['rho']:.1e}"
                    
                    if true_adj is not None:
                        msg += f", F1@0.3={record.get('f1_0.3', 0):.3f}"
                        msg += f", F1@0.5={record.get('f1_0.5', 0):.3f}"
                    
                    print(msg)
                
                # Check convergence / Kiểm tra hội tụ
                if h < self.config.h_tol:
                    if verbose:
                        print(f"Converged! h < {self.config.h_tol}")
                    break
                
                # Update augmented Lagrangian / Cập nhật Lagrangian tăng cường
                self.model.auglag.update_alpha(torch.tensor(h, device=self.device))
                
                if h > prev_h * 0.8:  # Not improving enough / Không cải thiện đủ
                    self.model.auglag.update_rho()
                
                prev_h = h
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint. / Lưu checkpoint huấn luyện."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'best_f1': self.best_f1,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint. / Tải checkpoint huấn luyện."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.history = ckpt['history']
        self.best_f1 = ckpt.get('best_f1', 0.0)


class FastTrainer:
    """
    Simplified trainer for quick experiments.
    Trình huấn luyện đơn giản hóa cho các thí nghiệm nhanh.
    
    Single-phase training with sensible defaults.
    Huấn luyện một giai đoạn với các mặc định hợp lý.
    """
    
    def __init__(self, model: CausalMLPModel, lr: float = 0.003):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def fit(
        self,
        data: torch.Tensor,
        n_steps: int = 5000,
        verbose: bool = True
    ) -> Dict:
        """Quick training. / Huấn luyện nhanh."""
        data = data.to(self.device)
        
        for step in range(n_steps):
            self.optimizer.zero_grad()
            
            idx = torch.randint(0, len(data), (256,))
            result = self.model(data[idx], return_components=True)
            result['loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            if verbose and step % 500 == 0:
                with torch.no_grad():
                    adj = self.model.adjacency.probs
                    h = self.model.compute_dag_constraint(adj).item()
                    print(f"Step {step}: loss={result['loss'].item():.3f}, h={h:.2e}")
        
        return {}
