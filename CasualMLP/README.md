# CausalMLP v2.1

A complete causal discovery and inference framework combining **Causica/DECI** and **GraN-DAG** features with novel improvements.

Một framework khám phá và suy luận nhân quả hoàn chỉnh kết hợp các tính năng của **Causica/DECI** và **GraN-DAG** với những cải tiến mới.

## 🌟 Key Features / Tính năng chính

### Core Discovery / Khám phá cốt lõi
*   **Adjacency Learning**: ENCO, Soft Adjacency, and novel Dual-Head parameterization.
    *   Học ma trận kề: ENCO, Cạnh mềm, và tham số hóa hai đầu mới.
*   **Non-linear Relationships**: Efficient per-node MLPs with residuals and LayerNorm.
    *   Mối quan hệ phi tuyến: MLP mỗi nút hiệu quả với phần dư và chuẩn hóa lớp.
*   **Graph Constraints**: GPU-accelerated NOTEARS constraint with Augmented Lagrangian.
    *   Ràng buộc đồ thị: Ràng buộc NOTEARS tăng tốc GPU với Lagrangian tăng cường.
*   **Noise Models**: Gaussian, Heteroscedastic, Adaptive, and Spline Flows.
    *   Mô hình nhiễu: Gaussian, Dị phương sai, Thích ứng và Luồng Spline.

### Inference & Interventions / Suy luận & Can thiệp
*   **Causal Inference**: `do()` calculus, ATE, CATE, ITE, and Counterfactuals.
    *   Suy luận nhân quả: Phép tính `do()`, ATE, CATE, ITE và Phản thực tế.
*   **Neural CATE**: TARNet and DragonNet implementations.
    *   Neural CATE: Triển khai TARNet và DragonNet.
*   **Uncertainty**: Gumbel sampling for graph posteriors and bootstrapping.
    *   Độ không chắc chắn: Lấy mẫu Gumbel cho hậu nghiệm đồ thị và bootstrapping.
*   **Active Learning**: Experimental design strategies for optimal interventions.
    *   Học chủ động: Chiến lược thiết kế thử nghiệm cho các can thiệp tối ưu.
*   **Variational Inference**: Bayesian posterior approximation.
    *   Suy luận biến phân: Xấp xỉ hậu nghiệm Bayesian.

### Advanced Capabilities / Khả năng nâng cao
*   **Multi-Environment**: Learn from heterogeneous datasets (observational + interventional).
    *   Đa môi trường: Học từ các tập dữ liệu không đồng nhất (quan sát + can thiệp).
*   **Latent Confounders**: Handling hidden variables via ADMGs.
    *   Biến ẩn: Xử lý các biến ẩn thông qua ADMG.
*   **Temporal Discovery**: Time-series causal modeling.
    *   Khám phá theo thời gian: Mô hình hóa nhân quả chuỗi thời gian.
*   **Missing Data**: Native handling of missing values.
    *   Dữ liệu thiếu: Xử lý tự nhiên các giá trị bị thiếu.
*   **Embeddings**: Learnable node embeddings for transfer learning.
    *   Embeddings: Các embedding nút có thể học được cho học chuyển đổi.

## 🚀 Quick Start / Bắt đầu nhanh

### Installation / Cài đặt

```bash
pip install -e .
```

### Basic Training / Huấn luyện cơ bản

```python
from config import CausalMLPConfig
from core import CausalMLPModel
from training import CurriculumTrainer

# Configure and train / Cấu hình và huấn luyện
config = CausalMLPConfig.for_sachs()
model = CausalMLPModel(config)
trainer = CurriculumTrainer(model)
trainer.fit(data)

# Get the learned graph / Lấy đồ thị đã học
adj_matrix = model.get_adj()
```

### Interventions / Can thiệp

```python
from inference import CausalInference

ci = CausalInference(model)
# Estimate ATE / Ước lượng ATE
ate = ci.ate(treatment_idx=0, outcome_idx=1)
print(f"ATE: {ate}")

# Counterfactual: What if node 0 had been 2.0?
# Phản thực tế: Điều gì xảy ra nếu nút 0 là 2.0?
cf = ci.counterfactual(observation, {0: 2.0})
```

## 📁 Project Structure / Cấu trúc dự án

*   `core/`: Main model components (MLP, Adjacency, Noise, Embeddings, Temporal, Multi-env).
    *   Các thành phần mô hình chính.
*   `training/`: Training loops and curriculum strategies.
    *   Vòng lặp huấn luyện và chiến lược chương trình.
*   `inference/`: Tools for interventions, uncertainty, and active learning.
    *   Công cụ cho can thiệp, độ không chắc chắn và học chủ động.
*   `utils/`: Metrics, pruning, visualization, and data handling.
    *   Các chỉ số, cắt tỉa, trực quan hóa và xử lý dữ liệu.

## 📄 License

MIT License
