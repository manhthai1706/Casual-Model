"""
CausalMLP - Complete Causal Discovery Framework / CausalMLP - Framework Khám phá Nhân quả Hoàn chỉnh

Combining the best of Causica/DECI and GraN-DAG with novel improvements.
Kết hợp những điểm tốt nhất của Causica/DECI và GraN-DAG với những cải tiến mới.

Features / Tính năng:
    Core / Cốt lõi:
    - DAG learning with NOTEARS constraint / Học DAG với ràng buộc NOTEARS
    - Per-node MLPs with path weight interpretation / MLP mỗi nút với diễn giải trọng số đường dẫn
    - ENCO adjacency parameterization / Tham số hóa ma trận kề ENCO
    - Spline flows for non-Gaussian noise / Luồng Spline cho nhiễu phi Gaussian
    - ADMG for latent confounders / ADMG cho các biến ẩn
    - Temporal causal discovery / Khám phá nhân quả theo thời gian
    
    Inference / Suy luận:
    - do-calculus interventions / Can thiệp do-calculus
    - ATE/CATE/ITE estimation / Ước lượng ATE/CATE/ITE
    - Counterfactual reasoning / Lý luận phản thực tế
    - Uncertainty quantification / Định lượng độ không chắc chắn
    - Variational inference / Suy luận biến phân
    - Active intervention design / Thiết kế can thiệp chủ động
    
    Training / Huấn luyện:
    - Curriculum learning / Học theo chương trình
    - Ensemble methods / Phương pháp Ensemble
    - Checkpointing / Checkpointing
    - CAM pruning / Cắt tỉa CAM

Version: 2.0.0
"""

from config import CausalMLPConfig

# Core
from core import (
    # Model
    CausalMLPModel,
    # Adjacency
    SoftAdjacency,
    ENCOAdjacency,
    DualHeadAdjacency,
    # MLP
    CausalMLP,
    EfficientCausalMLP,
    PiecewiseLinearMLP,
    PathWeightExtractor,
    # Noise
    GaussianNoise,
    HeteroscedasticNoise,
    AdaptiveNoise,
    SplineFlow,
    SplineNoiseModel,
    # ADMG
    ADMGModel,
    ADMGAdjacency,
    # Temporal
    TemporalCausalModel,
    create_temporal_windows,
    GrangerCausalityTest,
)

# Training
from training import CurriculumTrainer, FastTrainer

# Inference
from inference import (
    # Interventions
    CausalInference,
    add_causal_methods,
    # Uncertainty
    UncertaintyEstimator,
    EdgeUncertainty,
    # CATE
    TARNet,
    DragonNet,
    CATETrainer,
    DoublyRobustEstimator,
    # Variational
    BayesianCausalMLP,
    VariationalTrainer,
    # Active
    ActiveInterventionDesigner,
)

# Utils
from utils import (
    # DAG
    calculate_dag_constraint,
    to_dag,
    compute_metrics,
    compute_shd,
    AugmentedLagrangian,
    # Pruning
    cam_pruning,
    iterative_pruning,
    threshold_search,
    pns_selection,
    # Variable types
    VariableType,
    VariableSpec,
    infer_variable_types,
    # Visualization
    plot_graph,
    plot_training_curves,
    compare_adjacencies,
    # Checkpointing
    Checkpointer,
    EnsembleTrainer,
)

__version__ = '2.0.0'

__all__ = [
    # Config
    'CausalMLPConfig',
    
    # Core Model
    'CausalMLPModel',
    
    # Adjacency
    'SoftAdjacency',
    'ENCOAdjacency',
    'DualHeadAdjacency',
    
    # MLPs
    'CausalMLP',
    'EfficientCausalMLP',
    'PiecewiseLinearMLP',
    'PathWeightExtractor',
    
    # Noise
    'GaussianNoise',
    'HeteroscedasticNoise',
    'AdaptiveNoise',
    'SplineFlow',
    'SplineNoiseModel',
    
    # ADMG
    'ADMGModel',
    'ADMGAdjacency',
    
    # Temporal
    'TemporalCausalModel',
    'create_temporal_windows',
    'GrangerCausalityTest',
    
    # Training
    'CurriculumTrainer',
    'FastTrainer',
    
    # Inference - Interventions
    'CausalInference',
    'add_causal_methods',
    
    # Inference - Uncertainty
    'UncertaintyEstimator',
    'EdgeUncertainty',
    
    # Inference - CATE
    'TARNet',
    'DragonNet',
    'CATETrainer',
    'DoublyRobustEstimator',
    
    # Inference - Variational
    'BayesianCausalMLP',
    'VariationalTrainer',
    
    # Inference - Active
    'ActiveInterventionDesigner',
    
    # Utils
    'calculate_dag_constraint',
    'to_dag',
    'compute_metrics',
    'compute_shd',
    'AugmentedLagrangian',
    'cam_pruning',
    'iterative_pruning',
    'threshold_search',
    'pns_selection',
    'VariableType',
    'VariableSpec',
    'infer_variable_types',
    'plot_graph',
    'plot_training_curves',
    'compare_adjacencies',
    'Checkpointer',
    'EnsembleTrainer',
]
