"""
Checkpointing and Ensemble for CausalMLP / Checkpointing và Ensemble cho CausalMLP

Provides / Cung cấp:
- Save/load model checkpoints / Lưu/tải checkpoint mô hình
- Best model tracking / Theo dõi mô hình tốt nhất
- Ensemble training with multiple seeds / Huấn luyện ensemble với nhiều seed
- Model averaging / Trung bình hóa mô hình
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import numpy as np


class Checkpointer:
    """
    Save and load model checkpoints.
    Lưu và tải checkpoint mô hình.
    
    Features / Tính năng:
    - Save model state, optimizer, and training history / Lưu trạng thái mô hình, bộ tối ưu hóa và lịch sử huấn luyện
    - Track best model by metric / Theo dõi mô hình tốt nhất theo chỉ số
    - List and manage checkpoints / Liệt kê và quản lý checkpoint
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict] = None,
        filename: str = 'checkpoint.pt',
    ) -> Path:
        """
        Save checkpoint.
        Lưu checkpoint.
        
        Args:
            model: Model to save / Mô hình cần lưu
            optimizer: Optional optimizer / Bộ tối ưu hóa tùy chọn
            epoch: Current epoch/iteration / Kỷ nguyên/vòng lặp hiện tại
            metrics: Performance metrics / Các chỉ số hiệu năng
            filename: Checkpoint filename / Tên file checkpoint
            
        Returns:
            Path to saved checkpoint / Đường dẫn đến checkpoint đã lưu
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': model.config.to_dict() if hasattr(model, 'config') else {},
            'epoch': epoch,
            'metrics': metrics or {},
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save adjacency for easy access
        # Lưu ma trận kề để dễ truy cập
        if hasattr(model, 'adjacency'):
            with torch.no_grad():
                checkpoint['adjacency'] = model.adjacency.probs.cpu().numpy().tolist()
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        
        # Also save metrics as JSON / Cũng lưu các chỉ số dưới dạng JSON
        if metrics:
            json_path = path.with_suffix('.json')
            with open(json_path, 'w') as f:
                # Convert numpy/tensor values / Chuyển đổi các giá trị numpy/tensor
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        v = v.tolist() if hasattr(v, 'tolist') else float(v)
                    json_metrics[k] = v
                json.dump(json_metrics, f, indent=2)
        
        return path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = 'checkpoint.pt',
    ) -> Dict:
        """
        Load checkpoint.
        Tải checkpoint.
        
        Args:
            model: Model to load into / Mô hình để tải vào
            optimizer: Optional optimizer to load / Bộ tối ưu hóa tùy chọn để tải
            filename: Checkpoint filename / Tên file checkpoint
            
        Returns:
            Checkpoint data / Dữ liệu checkpoint
        """
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def save_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict,
        metric_name: str = 'f1',
        higher_is_better: bool = True,
    ) -> bool:
        """
        Save if current is best.
        Lưu nếu hiện tại là tốt nhất.
        
        Args:
            model: Model to save / Mô hình cần lưu
            optimizer: Optimizer / Bộ tối ưu hóa
            epoch: Current epoch / Kỷ nguyên hiện tại
            metrics: Current metrics / Các chỉ số hiện tại
            metric_name: Metric to compare / Chỉ số để so sánh
            higher_is_better: Direction of metric / Hướng của chỉ số (cao hơn là tốt hơn hay không)
            
        Returns:
            True if saved as new best / True nếu đã lưu là tốt nhất mới
        """
        best_path = self.save_dir / 'best_metrics.json'
        
        current_value = metrics.get(metric_name, 0)
        
        if best_path.exists():
            with open(best_path, 'r') as f:
                best = json.load(f)
            best_value = best.get(metric_name, 
                                 float('-inf') if higher_is_better else float('inf'))
        else:
            best_value = float('-inf') if higher_is_better else float('inf')
        
        is_best = (current_value > best_value) if higher_is_better else (current_value < best_value)
        
        if is_best:
            self.save(model, optimizer, epoch, metrics, 'best_model.pt')
            return True
        
        return False
    
    def list_checkpoints(self) -> List[Path]:
        """List all checkpoints. / Liệt kê tất cả các checkpoint."""
        return list(self.save_dir.glob('*.pt'))
    
    def load_best(self, model: nn.Module) -> Dict:
        """Load best model checkpoint. / Tải checkpoint mô hình tốt nhất."""
        return self.load(model, filename='best_model.pt')


class EnsembleTrainer:
    """
    Train ensemble of models with different seeds.
    Huấn luyện tập hợp các mô hình với các seed khác nhau.
    
    Provides / Cung cấp:
    - Multiple runs with different initializations / Nhiều lần chạy với các khởi tạo khác nhau
    - Result averaging / Trung bình hóa kết quả
    - Voting for edge selection / Bỏ phiếu để chọn cạnh
    """
    
    def __init__(
        self,
        model_class,
        config,
        trainer_class,
        device: str = 'cpu',
    ):
        """
        Args:
            model_class: Model class to instantiate / Lớp mô hình để khởi tạo
            config: Model configuration / Cấu hình mô hình
            trainer_class: Trainer class / Lớp huấn luyện
            device: Computation device / Thiết bị tính toán
        """
        self.model_class = model_class
        self.config = config
        self.trainer_class = trainer_class
        self.device = device
        
        self.models: List[nn.Module] = []
        self.histories: List[List[Dict]] = []
    
    def fit(
        self,
        data: torch.Tensor,
        true_adjacency: Optional[torch.Tensor] = None,
        n_runs: int = 5,
        seeds: Optional[List[int]] = None,
        verbose: bool = True,
    ):
        """
        Train ensemble.
        Huấn luyện ensemble.
        
        Args:
            data: Training data / Dữ liệu huấn luyện
            true_adjacency: Ground truth / Ground truth
            n_runs: Number of runs / Số lần chạy
            seeds: Random seeds (default: 0, 1, 2, ...) / Seed ngẫu nhiên (mặc định: 0, 1, 2, ...)
            verbose: Print progress / In tiến trình
        """
        if seeds is None:
            seeds = list(range(n_runs))
        
        self.models = []
        self.histories = []
        
        for i, seed in enumerate(seeds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Run {i+1}/{n_runs} (seed={seed})")
                print('='*60)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create fresh model / Tạo mô hình mới
            model = self.model_class(self.config)
            model.to(self.device)
            
            # Train / Huấn luyện
            trainer = self.trainer_class(model, self.config)
            result = trainer.fit(data, true_adjacency, verbose=verbose)
            
            self.models.append(model)
            self.histories.append(result.get('history', []))
    
    def ensemble_adjacency(
        self,
        method: str = 'mean',
    ) -> torch.Tensor:
        """
        Get ensemble adjacency.
        Lấy ma trận kề ensemble.
        
        Args:
            method: 'mean' or 'vote'
            
        Returns:
            Aggregated adjacency matrix / Ma trận kề tổng hợp
        """
        adjacencies = []
        
        for model in self.models:
            with torch.no_grad():
                adj = model.adjacency.probs.cpu()
                adjacencies.append(adj)
        
        adjacencies = torch.stack(adjacencies)
        
        if method == 'mean':
            return adjacencies.mean(dim=0)
        elif method == 'vote':
            votes = (adjacencies > 0.5).float()
            return votes.mean(dim=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def evaluate_ensemble(
        self,
        true_adjacency: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate ensemble performance.
        Đánh giá hiệu năng ensemble.
        
        Returns metrics for mean and vote aggregation.
        Trả về các chỉ số cho tổng hợp trung bình và bỏ phiếu.
        """
        from utils.dag_utils import compute_metrics
        
        results = {}
        
        # Mean aggregation / Tổng hợp trung bình
        adj_mean = self.ensemble_adjacency('mean')
        results['mean'] = compute_metrics(adj_mean, true_adjacency, threshold)
        
        # Vote aggregation / Tổng hợp bỏ phiếu
        adj_vote = self.ensemble_adjacency('vote')
        results['vote'] = compute_metrics(adj_vote, true_adjacency, threshold)
        
        # Individual models / Các mô hình cá nhân
        individual_f1s = []
        for model in self.models:
            with torch.no_grad():
                adj = model.adjacency.probs.cpu()
                metrics = compute_metrics(adj, true_adjacency, threshold)
                individual_f1s.append(metrics['f1'])
        
        results['individual_f1_mean'] = np.mean(individual_f1s)
        results['individual_f1_std'] = np.std(individual_f1s)
        
        return results
    
    def get_confident_edges(
        self,
        min_agreement: float = 0.8,
        threshold: float = 0.5,
    ) -> List[tuple]:
        """
        Get edges that most models agree on.
        Lấy các cạnh mà hầu hết các mô hình đều đồng ý.
        
        Args:
            min_agreement: Minimum fraction of models / Tỷ lệ tối thiểu các mô hình
            threshold: Binarization threshold / Ngưỡng nhị phân hóa
            
        Returns:
            List of (source, target, agreement) tuples / Danh sách các bộ (nguồn, đích, thỏa thuận)
        """
        n = self.config.num_nodes
        
        votes = torch.zeros(n, n)
        
        for model in self.models:
            with torch.no_grad():
                adj = model.adjacency.probs.cpu()
                votes += (adj > threshold).float()
        
        agreement = votes / len(self.models)
        
        confident = []
        for i in range(n):
            for j in range(n):
                if i != j and agreement[i, j] >= min_agreement:
                    confident.append((i, j, agreement[i, j].item()))
        
        confident.sort(key=lambda x: -x[2])
        
        return confident


def train_with_bootstrap(
    model_class,
    config,
    trainer_class,
    data: torch.Tensor,
    n_bootstrap: int = 10,
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Train on bootstrap samples for uncertainty.
    Huấn luyện trên các mẫu bootstrap để lấy độ không chắc chắn.
    
    Returns:
        List of learned adjacency matrices / Danh sách các ma trận kề đã học
    """
    n_samples = data.shape[0]
    adjacencies = []
    
    for i in range(n_bootstrap):
        if verbose:
            print(f"Bootstrap {i+1}/{n_bootstrap}")
        
        # Bootstrap sample / Mẫu bootstrap
        indices = torch.randint(0, n_samples, (n_samples,))
        boot_data = data[indices]
        
        # Train / Huấn luyện
        torch.manual_seed(i)
        model = model_class(config)
        trainer = trainer_class(model, config)
        trainer.fit(boot_data, verbose=False)
        
        # Get adjacency / Lấy ma trận kề
        with torch.no_grad():
            adj = model.adjacency.probs.cpu().numpy()
            adjacencies.append(adj)
    
    return adjacencies
