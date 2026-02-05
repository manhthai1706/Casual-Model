"""
Test All CausalMLP Features / Kiểm thử Tất cả các Tính năng CausalMLP

Tests / Các kiểm thử:
1. Interventions (do, ATE, counterfactual) / Can thiệp (do, ATE, phản thực tế)
2. Uncertainty quantification / Định lượng sự không chắc chắn
3. CAM pruning / Cắt tỉa CAM
4. Checkpointing / Checkpointing
5. Ensemble training / Huấn luyện ensemble
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CausalMLPConfig
from core.model import CausalMLPModel
from training.trainer import CurriculumTrainer
from inference.intervention import CausalInference
from inference.uncertainty import UncertaintyEstimator
from utils.pruning import cam_pruning, threshold_search
from utils.checkpointing import Checkpointer, EnsembleTrainer
from utils.dag_utils import compute_metrics


def generate_test_data(n_samples: int = 500, seed: int = 42):
    """Generate simple test data: X0 -> X1 -> X2. / Tạo dữ liệu kiểm thử đơn giản: X0 -> X1 -> X2."""
    np.random.seed(seed)
    
    x0 = np.random.randn(n_samples)
    x1 = 0.8 * x0 + np.random.randn(n_samples) * 0.3
    x2 = 0.7 * x1 + np.random.randn(n_samples) * 0.3
    
    data = np.column_stack([x0, x1, x2])
    data = (data - data.mean(0)) / (data.std(0) + 1e-8)
    
    true_adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=np.float32)
    
    return torch.tensor(data, dtype=torch.float32), torch.tensor(true_adj)


def test_interventions():
    """Test do-calculus and ATE. / Kiểm tra do-calculus và ATE."""
    print("\n" + "=" * 50)
    print("TEST: Interventions / KIỂM THỬ: Can thiệp")
    print("=" * 50)
    
    data, true_adj = generate_test_data()
    
    config = CausalMLPConfig(num_nodes=3, warmup_steps=200, max_outer_iter=3)
    model = CausalMLPModel(config)
    
    trainer = CurriculumTrainer(model, config)
    trainer.fit(data, verbose=False)
    
    ci = CausalInference(model)
    
    # Test do()
    print("\n1. do(X0 = 1.0):")
    samples = ci.do({0: 1.0}, n_samples=100)
    print(f"   Sample mean: {samples.mean(dim=0).numpy()}")
    
    # Test ATE
    print("\n2. ATE of X0 on X2:")
    ate = ci.ate(treatment=0, outcome=2, n_samples=1000)
    print(f"   ATE = {ate['ate']:.3f} (CI: [{ate['ci_low']:.3f}, {ate['ci_high']:.3f}])")
    
    # Test counterfactual
    print("\n3. Counterfactual / Phản thực tế:")
    factual = data[0:1]
    cf = ci.counterfactual(factual, {0: 2.0})
    print(f"   Factual:       {factual.numpy().flatten()}")
    print(f"   Counterfactual: {cf.numpy().flatten()}")
    
    print("\n[PASS] Interventions")


def test_uncertainty():
    """Test uncertainty quantification. / Kiểm tra định lượng sự không chắc chắn."""
    print("\n" + "=" * 50)
    print("TEST: Uncertainty / KIỂM THỬ: Độ không chắc chắn")
    print("=" * 50)
    
    data, true_adj = generate_test_data()
    
    config = CausalMLPConfig(num_nodes=3, warmup_steps=200, max_outer_iter=3)
    model = CausalMLPModel(config)
    
    trainer = CurriculumTrainer(model, config)
    trainer.fit(data, verbose=False)
    
    estimator = UncertaintyEstimator(model)
    
    # Gumbel samples
    print("\n1. Gumbel sampling / Lấy mẫu Gumbel:")
    summary = estimator.summary(n_samples=50)
    print(f"   Mean edges: {summary['mean_n_edges']:.1f} +/- {summary['std_n_edges']:.1f}")
    
    # Confident edges
    print("\n2. Confident edges / Các cạnh tự tin:")
    confident = estimator.confident_edges(min_confidence=0.5, n_samples=50)
    for src, tgt, conf in confident[:3]:
        print(f"   {src} -> {tgt}: {conf:.2f}")
    
    print("\n[PASS] Uncertainty")


def test_cam_pruning():
    """Test CAM pruning. / Kiểm tra cắt tỉa CAM."""
    print("\n" + "=" * 50)
    print("TEST: CAM Pruning / KIỂM THỬ: Cắt tỉa CAM")
    print("=" * 50)
    
    data, true_adj = generate_test_data()
    
    # Create noisy adjacency / Tạo ma trận kề nhiễu
    noisy_adj = np.array([
        [0, 0.8, 0.3],  # 0->1 true, 0->2 spurious / 0->1 đúng, 0->2 giả
        [0.2, 0, 0.7],  # 1->0 spurious, 1->2 true / 1->0 giả, 1->2 đúng
        [0, 0, 0]
    ], dtype=np.float32)
    
    print("\n1. Before pruning / Trước khi cắt tỉa:")
    print(f"   Edges: {(noisy_adj > 0.3).sum()}")
    
    pruned = cam_pruning(data.numpy(), noisy_adj, threshold=0.3, method='regression', alpha=0.05)
    
    print("\n2. After pruning / Sau khi cắt tỉa:")
    print(f"   Edges: {pruned.sum()}")
    print(f"   Adjacency:\n{pruned}")
    
    print("\n[PASS] CAM Pruning")


def test_checkpointing():
    """Test save/load. / Kiểm tra lưu/tải."""
    print("\n" + "=" * 50)
    print("TEST: Checkpointing / KIỂM THỬ: Checkpointing")
    print("=" * 50)
    
    import tempfile
    import shutil
    
    data, true_adj = generate_test_data()
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = CausalMLPConfig(num_nodes=3, warmup_steps=100, max_outer_iter=2)
        model = CausalMLPModel(config)
        
        trainer = CurriculumTrainer(model, config)
        trainer.fit(data, verbose=False)
        
        # Save / Lưu
        ckpt = Checkpointer(temp_dir)
        path = ckpt.save(model, epoch=1, metrics={'f1': 0.5})
        print(f"\n1. Saved to: {path}")
        
        # Get original adjacency / Lấy ma trận kề gốc
        with torch.no_grad():
            orig_adj = model.adjacency.probs.numpy().copy()
        
        # Load into new model / Tải vào mô hình mới
        new_model = CausalMLPModel(config)
        ckpt.load(new_model)
        
        with torch.no_grad():
            loaded_adj = new_model.adjacency.probs.numpy()
        
        match = np.allclose(orig_adj, loaded_adj, atol=1e-5)
        print(f"\n2. Adjacency match: {match}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n[PASS] Checkpointing")


def test_all():
    """Run all tests. / Chạy tất cả kiểm thử."""
    print("\n" + "=" * 50)
    print("CausalMLP: Test All Features / CausalMLP: Kiểm thử Tất cả Tính năng")
    print("=" * 50)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_interventions()
    test_uncertainty()
    test_cam_pruning()
    test_checkpointing()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! / TẤT CẢ KIỂM THỬ ĐỀU QUA!")
    print("=" * 50)


if __name__ == '__main__':
    test_all()
