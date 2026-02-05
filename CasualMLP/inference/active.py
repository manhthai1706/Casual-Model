"""
Active Interventions for CausalMLP / Can thiệp Chủ động cho CausalMLP

Optimal experimental design for causal discovery:
Thiết kế thí nghiệm tối ưu cho khám phá nhân quả:
- Information-theoretic intervention selection / Lựa chọn can thiệp dựa trên lý thuyết thông tin
- Expected information gain / Lợi ích thông tin kỳ vọng
- Batch experimental design / Thiết kế thí nghiệm theo lô

Reference / Tham khảo:
- Tong & Koller, "Active Learning for Structure in Bayesian Networks" (IJCAI 2001)
- Murphy, "Active Learning of Causal Bayes Net Structure" (2001)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Set
import numpy as np
from itertools import combinations


class ActiveInterventionDesigner:
    """
    Design optimal interventions for causal discovery.
    Thiết kế can thiệp tối ưu cho khám phá nhân quả.
    
    Selects which variables to intervene on to maximally reduce uncertainty about the causal graph.
    Chọn các biến để can thiệp nhằm giảm thiểu tối đa độ không chắc chắn về đồ thị nhân quả.
    """
    
    def __init__(
        self,
        model,  # CausalMLP model with uncertainty
        data: torch.Tensor,
        device: str = 'cpu',
    ):
        self.model = model
        self.data = data
        self.device = device
        self.num_nodes = model.config.num_nodes
        
        # Track completed interventions / Theo dõi các can thiệp đã hoàn thành
        self.completed_interventions: List[Dict] = []
    
    def expected_information_gain(
        self,
        intervention_set: Set[int],
        n_samples: int = 100,
    ) -> float:
        """
        Compute expected information gain from intervening on a set of variables.
        Tính lợi ích thông tin kỳ vọng từ việc can thiệp vào một tập hợp các biến.
        
        EIG = H(G) - E[H(G|D_int)]
        
        Where H(G) is current entropy and H(G|D_int) is posterior entropy after observing intervention data.
        Trong đó H(G) là entropy hiện tại và H(G|D_int) là entropy hậu nghiệm sau khi quan sát dữ liệu can thiệp.
        
        Args:
            intervention_set: Set of variable indices to intervene on / Tập hợp các chỉ số biến để can thiệp
            n_samples: Monte Carlo samples / Số lượng mẫu Monte Carlo
            
        Returns:
            Expected information gain / Lợi ích thông tin kỳ vọng
        """
        # Current entropy (approximate from edge probabilities)
        # Entropy hiện tại (xấp xỉ từ xác suất cạnh)
        with torch.no_grad():
            probs = self.model.adjacency.probs.cpu()
        
        current_entropy = self._edge_entropy(probs)
        
        # Estimate expected posterior entropy
        # Ước lượng entropy hậu nghiệm kỳ vọng
        expected_posterior_entropy = self._estimate_posterior_entropy(
            intervention_set, n_samples
        )
        
        return current_entropy - expected_posterior_entropy
    
    def _edge_entropy(self, probs: torch.Tensor) -> float:
        """Compute entropy of edge distribution. / Tính entropy của phân phối cạnh."""
        # Binary entropy for each edge / Entropy nhị phân cho mỗi cạnh
        eps = 1e-10
        h = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        
        # Remove diagonal / Loại bỏ đường chéo
        n = probs.shape[0]
        mask = 1 - torch.eye(n)
        h = h * mask
        
        return h.sum().item()
    
    def _estimate_posterior_entropy(
        self,
        intervention_set: Set[int],
        n_samples: int,
    ) -> float:
        """
        Estimate expected posterior entropy after intervention.
        Ước lượng entropy hậu nghiệm kỳ vọng sau khi can thiệp.
        
        This is a simplified approximation based on which edges become identifiable after intervention.
        Đây là một xấp xỉ đơn giản dựa trên các cạnh nào trở nên xác định được sau khi can thiệp.
        """
        # Edges involving intervened variables become more identifiable
        # Các cạnh liên quan đến biến được can thiệp trở nên dễ xác định hơn
        probs = self.model.adjacency.probs.cpu().clone()
        
        for node in intervention_set:
            # Edges TO intervened node become certain (0) because we break the causal mechanism
            # Các cạnh ĐẾN nút được can thiệp trở nên chắc chắn (0) vì chúng ta phá vỡ cơ chế nhân quả
            probs[:, node] = 0.5 * probs[:, node]  # Reduce uncertainty / Giảm độ không chắc chắn
            
            # Edges FROM intervened node become more identifiable
            # Các cạnh TỪ nút được can thiệp trở nên dễ xác định hơn
            probs[node, :] = self._sharpen(probs[node, :])
        
        return self._edge_entropy(probs)
    
    def _sharpen(self, probs: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
        """Sharpen probabilities toward 0 or 1. / Làm sắc nét xác suất về phía 0 hoặc 1."""
        # Move probabilities away from 0.5 / Di chuyển xác suất ra xa 0.5
        sharpened = probs.clone()
        mask = probs > 0.5
        sharpened[mask] = 0.5 + (probs[mask] - 0.5) * factor
        sharpened[~mask] = 0.5 - (0.5 - probs[~mask]) * factor
        return sharpened.clamp(0, 1)
    
    def select_single_intervention(
        self,
        exclude: Optional[Set[int]] = None,
    ) -> Tuple[int, float]:
        """
        Select single best variable to intervene on.
        Chọn một biến tốt nhất để can thiệp.
        
        Args:
            exclude: Variables to exclude from selection / Các biến loại trừ khỏi lựa chọn
            
        Returns:
            (best_variable, expected_gain)
        """
        exclude = exclude or set()
        
        best_node = None
        best_gain = -float('inf')
        
        for node in range(self.num_nodes):
            if node in exclude:
                continue
            
            gain = self.expected_information_gain({node})
            
            if gain > best_gain:
                best_gain = gain
                best_node = node
        
        return best_node, best_gain
    
    def select_intervention_batch(
        self,
        batch_size: int = 3,
        method: str = 'greedy',
    ) -> List[Tuple[int, float]]:
        """
        Select a batch of interventions to perform.
        Chọn một lô các can thiệp để thực hiện.
        
        Args:
            batch_size: Number of interventions / Số lượng can thiệp
            method: 'greedy' or 'exhaustive' / 'tham lam' hoặc 'toàn diện'
            
        Returns:
            List of (variable, gain) tuples / Danh sách các bộ (biến, lợi ích)
        """
        if method == 'greedy':
            return self._greedy_batch(batch_size)
        elif method == 'exhaustive' and batch_size <= 3:
            return self._exhaustive_batch(batch_size)
        else:
            return self._greedy_batch(batch_size)
    
    def _greedy_batch(self, batch_size: int) -> List[Tuple[int, float]]:
        """Greedy intervention selection. / Lựa chọn can thiệp tham lam."""
        selected = []
        excluded = set()
        
        for _ in range(batch_size):
            node, gain = self.select_single_intervention(exclude=excluded)
            if node is not None:
                selected.append((node, gain))
                excluded.add(node)
        
        return selected
    
    def _exhaustive_batch(self, batch_size: int) -> List[Tuple[int, float]]:
        """Exhaustive search for best batch. / Tìm kiếm toàn diện cho lô tốt nhất."""
        best_set = None
        best_gain = -float('inf')
        
        for subset in combinations(range(self.num_nodes), batch_size):
            gain = self.expected_information_gain(set(subset))
            if gain > best_gain:
                best_gain = gain
                best_set = subset
        
        return [(node, best_gain / len(best_set)) for node in best_set]
    
    def record_intervention(
        self,
        node: int,
        value: float,
        observed_data: torch.Tensor,
    ):
        """Record a completed intervention for future reference. / Ghi lại can thiệp đã hoàn thành để tham khảo sau này."""
        self.completed_interventions.append({
            'node': node,
            'value': value,
            'data': observed_data,
        })
    
    def intervention_priority_ranking(self) -> List[Tuple[int, float]]:
        """
        Rank all variables by intervention priority.
        Xếp hạng tất cả các biến theo ưu tiên can thiệp.
        
        Returns list of (variable, expected_gain) sorted by gain.
        Trả về danh sách (biến, lợi ích kỳ vọng) được sắp xếp theo lợi ích.
        """
        rankings = []
        
        for node in range(self.num_nodes):
            gain = self.expected_information_gain({node})
            rankings.append((node, gain))
        
        rankings.sort(key=lambda x: -x[1])
        return rankings
    
    def edge_identification_analysis(self) -> Dict[Tuple[int, int], Dict]:
        """
        Analyze which interventions would help identify each edge.
        Phân tích can thiệp nào sẽ giúp xác định từng cạnh.
        
        For each potential edge (i, j), determine:
        Với mỗi cạnh tiềm năng (i, j), xác định:
        - Current probability / Xác suất hiện tại
        - Which intervention would be most informative / Can thiệp nào sẽ mang lại nhiều thông tin nhất
        """
        analysis = {}
        
        with torch.no_grad():
            probs = self.model.adjacency.probs.cpu()
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                
                current_prob = probs[i, j].item()
                uncertainty = -current_prob * np.log(current_prob + 1e-10) - \
                             (1 - current_prob) * np.log(1 - current_prob + 1e-10)
                
                # Best intervention for this edge / Can thiệp tốt nhất cho cạnh này
                # Intervening on source (i) helps identify outgoing edges / Can thiệp vào nguồn (i) giúp xác định các cạnh đi ra
                # Intervening on target (j) helps rule out incoming edges / Can thiệp vào đích (j) giúp loại trừ các cạnh đi vào
                best_intervention = i if current_prob > 0.5 else j
                
                analysis[(i, j)] = {
                    'probability': current_prob,
                    'uncertainty': uncertainty,
                    'best_intervention': best_intervention,
                    'reason': 'confirm' if current_prob > 0.5 else 'refute',
                }
        
        return analysis


class SequentialExperimentDesigner:
    """
    Sequential experimental design for causal discovery.
    Thiết kế thí nghiệm tuần tự cho khám phá nhân quả.
    
    Iteratively / Lặp lại:
    1. Analyze current uncertainty / Phân tích độ không chắc chắn hiện tại
    2. Select best intervention / Chọn can thiệp tốt nhất
    3. Collect interventional data / Thu thập dữ liệu can thiệp
    4. Update model / Cập nhật mô hình
    5. Repeat until convergence / Lặp lại cho đến khi hội tụ
    """
    
    def __init__(
        self,
        model_class,
        config,
        trainer_class,
    ):
        self.model_class = model_class
        self.config = config
        self.trainer_class = trainer_class
        
        self.models: List = []
        self.interventions: List[Dict] = []
    
    def run_experiment_cycle(
        self,
        observational_data: torch.Tensor,
        intervention_simulator,  # Function that simulates interventional data
        n_rounds: int = 5,
        n_samples_per_intervention: int = 100,
        verbose: bool = True,
    ):
        """
        Run sequential experiment design.
        Chạy thiết kế thí nghiệm tuần tự.
        
        Args:
            observational_data: Initial observational data / Dữ liệu quan sát ban đầu
            intervention_simulator: Function (node, value, n_samples) -> data / Hàm mô phỏng dữ liệu can thiệp
            n_rounds: Number of intervention rounds / Số vòng can thiệp
            n_samples_per_intervention: Samples to collect per intervention / Số mẫu thu thập mỗi can thiệp
            verbose: Print progress / In tiến trình
        """
        current_data = observational_data
        
        for round_idx in range(n_rounds):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Round {round_idx + 1}/{n_rounds}")
            
            # Train model on current data / Huấn luyện mô hình trên dữ liệu hiện tại
            model = self.model_class(self.config)
            trainer = self.trainer_class(model, self.config)
            trainer.fit(current_data, verbose=False)
            self.models.append(model)
            
            # Design next intervention / Thiết kế can thiệp tiếp theo
            designer = ActiveInterventionDesigner(model, current_data)
            node, gain = designer.select_single_intervention(
                exclude={i['node'] for i in self.interventions}
            )
            
            if node is None:
                if verbose:
                    print("No more interventions to perform")
                    print("Không còn can thiệp nào để thực hiện")
                break
            
            if verbose:
                print(f"Selected intervention: X{node} (expected gain: {gain:.3f})")
                print(f"Can thiệp được chọn: X{node} (lợi ích kỳ vọng: {gain:.3f})")
            
            # Simulate intervention / Mô phỏng can thiệp
            intervention_value = 0.0  # Could be optimized
            new_data = intervention_simulator(
                node, intervention_value, n_samples_per_intervention
            )
            
            # Record and update / Ghi lại và cập nhật
            self.interventions.append({
                'round': round_idx,
                'node': node,
                'value': intervention_value,
                'gain': gain,
            })
            
            # Augment dataset / Tăng cường bộ dữ liệu
            current_data = torch.cat([current_data, new_data], dim=0)
            
            if verbose:
                with torch.no_grad():
                    probs = model.adjacency.probs
                    n_edges = (probs > 0.5).sum().item()
                print(f"Current edges: {n_edges}")
                print(f"Số cạnh hiện tại: {n_edges}")
        
        return {
            'models': self.models,
            'interventions': self.interventions,
            'final_data': current_data,
        }
