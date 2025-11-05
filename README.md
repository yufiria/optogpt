# OptoGPT - 光学多层薄膜Transformer基础模型

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 📖 项目简介

本仓库包含三个相关的研究工作，使用Transformer架构解决光学多层薄膜的设计和仿真问题：

### 1. OptoGPT - 多层薄膜逆向设计基础模型
OptoGPT（Opto Generative Pretrained Transformer）是一个仅解码器的Transformer模型，用于解决多层薄膜结构的逆向设计问题。给定目标光谱，模型可以预测能够实现该光谱的薄膜结构。

📄 **论文**: [OptoGPT: A Foundation Model for Inverse Design](https://www.oejournal.org/article/doi/10.29026/oea.2024.240062)

### 2. OL-Transformer - 通用光学仿真器
OL-Transformer（Opto-Layer Transformer）是一个通用的代理仿真器，能够快速准确地预测多层薄膜结构的反射和透射光谱，支持多达10^25种不同的结构组合。

📄 **论文**: [OL-Transformer: A Fast and Universal Surrogate Simulator](https://arxiv.org/abs/2305.11984)

### 3. 自我改进数据增强
通过神经网络的外推能力，提出自我改进数据增强技术，解决光学基础模型的分布外(OOD)挑战，显著提升实际设计任务的性能。

📄 **论文**: [Solving Out-of-Distribution Challenges](https://openreview.net/forum?id=8jqhElTmNP)

![OptoGPT工作流程](optogpt/figures/optogpt.png)

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA-capable GPU（推荐，用于加速训练和推理）
- 至少16GB RAM

### 安装步骤

#### 1. 克隆仓库
```bash
git clone https://github.com/yufiria/optogpt.git
cd optogpt
```

#### 2. 创建Conda环境
```bash
cd optogpt
conda env create -f environment.yml
conda activate optogpt
```

#### 3. 下载数据集和模型

**数据集**: 从 [HuggingFace](https://huggingface.co/datasets/mataigao/optogpt_data) 下载数据集
```bash
# 下载后解压到 optogpt/dataset/ 目录
# 然后运行 data_conversion.ipynb 将 .csv 转换为 .pkl 格式
```

**预训练模型**: 从 [HuggingFace](https://huggingface.co/mataigao/optogpt) 下载预训练模型

---

## 📁 项目结构

```
optogpt/
├── optogpt/                        # OptoGPT和OL-Transformer主目录
│   ├── core/                       # 核心功能模块
│   │   ├── models/                 # 模型架构
│   │   │   ├── transformer.py      # Transformer实现（编码器/解码器）
│   │   │   └── __init__.py
│   │   ├── datasets/               # 数据处理
│   │   │   ├── datasets.py         # 数据加载和预处理
│   │   │   ├── sim.py              # 光学仿真（传输矩阵法）
│   │   │   └── __init__.py
│   │   ├── trains/                 # 训练模块
│   │   │   ├── train.py            # 训练循环和优化器
│   │   │   └── __init__.py
│   │   └── utils/                  # 工具函数
│   │       ├── color_system.py     # 颜色空间转换
│   │       ├── common.py           # 通用工具
│   │       ├── cie-cmf.txt         # CIE颜色匹配函数
│   │       └── __init__.py
│   ├── dataset/                    # 数据集目录（需下载）
│   │   ├── Structure_train.pkl     # 训练集结构
│   │   ├── Structure_dev.pkl       # 验证集结构
│   │   ├── Spectrum_train.pkl      # 训练集光谱
│   │   └── Spectrum_dev.pkl        # 验证集光谱
│   ├── nk/                         # 材料折射率数据（18种材料）
│   │   ├── Al.csv
│   │   ├── TiO2.csv
│   │   └── ...                     # 其他材料的n,k数据
│   ├── saved_models/               # 保存的模型检查点
│   ├── figures/                    # 论文图表
│   ├── run_optogpt.py              # OptoGPT训练脚本（逆向设计）
│   ├── run_ol_transformer.py       # OL-Transformer训练脚本（正向设计）
│   ├── analysis_optogpt.ipynb      # OptoGPT分析和评估
│   ├── analysis_ol_transformer.ipynb # OL-Transformer分析
│   ├── data_conversion.ipynb       # 数据格式转换
│   └── environment.yml             # Conda环境配置
│
└── self_improving/                 # 自我改进数据增强模块
    ├── run_self_improving.py       # 主流程脚本
    ├── prepare_aug_data.py         # 数据增强准备
    ├── data_perturb.py             # 结构扰动（GA/PSO）
    ├── generate_dev_data.py        # OOD测试数据生成
    ├── combine_data.py             # 数据集合并
    ├── model_retrain.py            # 模型重训练
    ├── requirements.txt            # Python依赖
    └── README.md                   # 模块说明文档
```

---

## 💻 使用方法

### 1. OptoGPT逆向设计（光谱→结构）

训练OptoGPT模型预测给定光谱对应的薄膜结构：

```bash
cd optogpt
CUDA_VISIBLE_DEVICES=0 python run_optogpt.py \
    --epochs 1000 \
    --batch_size 1000 \
    --spec_type R_T \
    --layers 1 \
    --d_model 1024
```

**主要参数说明**:
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--spec_type`: 光谱类型（R: 反射，T: 透射，R_T: 两者）
- `--layers`: Transformer层数
- `--d_model`: 模型维度
- `--smoothing`: 标签平滑系数（KL散度）

### 2. OL-Transformer正向仿真（结构→光谱）

训练OL-Transformer模型预测薄膜结构的光谱响应：

```bash
cd optogpt
CUDA_VISIBLE_DEVICES=0 python run_ol_transformer.py \
    --epochs 1000 \
    --batch_size 1000 \
    --spec_type R_T \
    --layers 12 \
    --d_model 1024
```

### 3. 自我改进数据增强

使用自我改进数据增强提升模型对OOD数据的泛化能力：

```bash
cd self_improving
python run_self_improving.py \
    --model_path ../optogpt/saved_models/optogpt/model_best.pt \
    --train_struct_path ../optogpt/dataset/Structure_train.pkl \
    --train_spec_path ../optogpt/dataset/Spectrum_train.pkl \
    --dev_struct_path ../optogpt/dataset/Structure_dev.pkl \
    --dev_spec_path ../optogpt/dataset/Spectrum_dev.pkl \
    --output_dir ./output \
    --decoding_method TOP-KP_Decode_v2 \
    --perturbation_method GA_PSO \
    --epochs 10
```

**自我改进流程**:
1. 生成多样化的开发数据（高斯光谱、DBR结构等）
2. 使用不同解码策略（Greedy/Top-K/Beam Search）生成候选结构
3. 应用智能扰动（GA/PSO）优化候选结构
4. 筛选性能提升的结构作为增强数据
5. 使用增强数据重新训练模型

### 4. 模型评估和分析

使用Jupyter Notebook进行详细分析：

```bash
jupyter notebook
# 打开 analysis_optogpt.ipynb 或 analysis_ol_transformer.ipynb
```

---

## 🔬 核心技术

### Transformer架构
- **编码器（Encoder）**: 用于OL-Transformer，处理结构序列
- **解码器（Decoder）**: 用于OptoGPT，自回归生成结构
- **多头注意力**: 捕捉层间光-物质相互作用
- **位置编码**: 保留序列位置信息

### 数据表示
- **结构序列化**: 将多层薄膜表示为token序列（材料_厚度）
- **光谱向量化**: 反射率和透射率在71个波长点的采样
- **词汇表**: 包含18种材料 × 50种厚度 ≈ 900个token

### 训练策略
- **Noam学习率调度**: 带预热的学习率衰减
- **标签平滑**: 防止过拟合，提升泛化
- **早停机制**: 基于验证集损失自动停止

---

## 📦 依赖关系

### 核心依赖
- **PyTorch 2.0.1**: 深度学习框架
- **NumPy 1.23.4**: 数值计算
- **Pandas 1.5.1**: 数据处理
- **SciPy 1.10.1**: 科学计算

### 光学仿真
- **tmm 0.1.8**: 传输矩阵法光学仿真
- **colour-science 0.4.1**: 颜色空间转换

### 可视化
- **Matplotlib 3.7.5**: 绘图
- **Seaborn 0.13.2**: 统计可视化

### 优化（自我改进模块）
- **pyswarms**: 粒子群优化
- **scikit-learn**: 机器学习工具

完整依赖列表请查看 `optogpt/environment.yml` 和 `self_improving/requirements.txt`

---

## 🎨 颜色到光谱转换工具

项目还提供了颜色到光谱的优化转换算法，对结构色研究很有帮助：

```python
import numpy as np
from scipy.optimize import minimize
from colour.difference import delta_E_CIE2000

# 定义目标颜色（RGB或LAB）
TARGET_LAB = [50, 80, 0]  # 绿色

# 优化得到对应的光谱
def fitness(spec):
    lab = get_color(spec)
    mse = delta_E_CIE2000(lab, TARGET_LAB)
    smoothness = np.square(np.gradient(np.gradient(spec))).mean()
    return mse + 50*smoothness

res = minimize(fitness, x0, method='SLSQP', bounds=bounds)
optimized_spectrum = res.x
```

详细代码请参考 `optogpt/README.md` 中的示例。

---

## 📊 数据集说明

### 训练数据
- **结构**: 5-15层薄膜，18种材料，厚度5-250nm
- **光谱**: 400-1100nm波长范围，10nm间隔，71个采样点
- **数量**: ~100,000个结构-光谱对

### 材料分类
- **高折射率**: TiO2, ZnS, ZnSe, Ta2O5, HfO2
- **中折射率**: SiO2, Al2O3, MgF2, Si3N4  
- **低折射率**: MgO, ITO
- **金属**: Al, Ag
- **半导体**: Ge, Si

---

## 🔧 常见问题

### Q: 如何处理CUDA内存不足？
A: 减小 `--batch_size` 或 `--d_model` 参数，或使用梯度累积。

### Q: 训练需要多长时间？
A: 在单个GPU（如RTX 3090）上，OptoGPT约需12-24小时，OL-Transformer约需24-48小时。

### Q: 如何使用自己的材料数据？
A: 在 `nk/` 目录下添加新的CSV文件（包含wavelength, n, k三列），然后在代码中更新材料列表。

### Q: 模型可以处理多少层的结构？
A: 训练时使用5-15层，但模型可以外推到更多层（需要适当的位置编码）。

---

## 📝 引用

如果本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@article{ma2024optogpt,
  title={OptoGPT: a foundation model for inverse design in optical multilayer thin film structures},
  author={Ma, Taigao and Wang, Haozhu and Guo, L Jay},
  journal={Opto-Electronic Advances},
  volume={7},
  number={7},
  year={2024},
  publisher={Opto-Electronic Advance}
}

@article{ma2023ol,
  title={OL-Transformer: A Fast and Universal Surrogate Simulator for Optical Multilayer Thin Film Structures},
  author={Ma, Taigao and Wang, Haozhu and Guo, L Jay},
  journal={arXiv preprint arXiv:2305.11984},
  year={2023}
}

@inproceedings{ma2024solving,
  title={Solving Out-of-Distribution Challenges in Optical Foundation Models using Self-Improving Data Augmentation},
  author={Ma, Mingqian and Ma, Taigao and Guo, L Jay},
  booktitle={Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges}
}
```

---

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 [GitHub Issue](https://github.com/yufiria/optogpt/issues)
- 查看原始论文获取作者联系方式

---

## 📄 许可证

本项目遵循原作者的许可证协议。

---

<a name="english"></a>
## English Version

# OptoGPT - Transformer Foundation Model for Optical Multilayer Thin Films

## Overview

This repository contains three related research works on Transformer-based optical multilayer thin film design and simulation.

For detailed English documentation, please refer to:
- `/optogpt/README.md` - OptoGPT and OL-Transformer documentation
- `/self_improving/README.md` - Self-improving data augmentation documentation

### Quick Start

```bash
# Clone and setup
git clone https://github.com/yufiria/optogpt.git
cd optogpt/optogpt
conda env create -f environment.yml
conda activate optogpt

# Train OptoGPT (inverse design: spectrum → structure)
CUDA_VISIBLE_DEVICES=0 python run_optogpt.py

# Train OL-Transformer (forward simulation: structure → spectrum)
CUDA_VISIBLE_DEVICES=0 python run_ol_transformer.py
```

For more details, see the README files in respective directories.