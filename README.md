```markdown
# Wafer-to-Wafer Bonding CSAM 预测项目

本项目展示了一个端到端的工作流程，用于从 wafer-to-wafer bonding 中采集的 bond wave 数据预测 CSAM 图像（即 non bond 区域）。项目分为三个主要模块：
- **模拟数据生成**
- **模型训练**
- **模型推断（预测）**

## 目录结构

```
.
├── simulate_data.py       # 生成模拟 bond wave 数据和 CSAM 图（CSV 和 PNG 格式）
├── train_model.py         # 加载 CSV/PNG 数据，训练模型，并保存训练好的模型（trained_model.h5）
├── infer_model.py         # 从输入的 bond wave CSV 文件中加载数据，利用训练好的模型预测 CSAM 图
├── bond_wave_data/        # 模拟数据生成中保存的 CSV 文件，每个文件包含 52×1000 的 bond wave 数据
├── csam_images/           # 模拟数据生成中保存的 PNG 文件，每个文件为 256×256 的 CSAM 图像
└── input_bond_wave/       # infer_model.py 使用的输入文件夹，存放真实或模拟的 bond wave CSV 文件
```

## 依赖与安装

项目基于 Python 3.x，主要依赖以下库：
- numpy
- pandas
- matplotlib
- tensorflow (>=2.x)
- Pillow

使用 pip 安装依赖：
```bash
pip install numpy pandas matplotlib tensorflow pillow
```

## 使用方法

### 1. 生成模拟数据

运行 `simulate_data.py` 脚本，生成模拟的 bond wave 数据和对应的 CSAM 图。生成的数据将保存为：
- **bond_wave_data/**：每个样本保存为一个 CSV 文件，格式为 52×1000 的矩阵（每行代表一个传感器）。
- **csam_images/**：每个样本保存为一个 PNG 图片，尺寸为 256×256（灰度图，0 表示正常区域，1 表示 non bond 缺陷区域）。

运行命令：
```bash
python simulate_data.py
```

### 2. 训练模型

运行 `train_model.py` 脚本，加载模拟数据（CSV/PNG 格式），构建并训练一个简单的全连接网络模型。训练完成后，将保存训练好的模型为 `trained_model.h5`。

运行命令：
```bash
python train_model.py
```

你可以根据需要修改训练参数（如 epoch 数、batch size）和模型结构。

### 3. 模型推断

将待预测的 bond wave CSV 文件放入文件夹 **input_bond_wave/**（格式要求：52×1000 的矩阵）。运行 `infer_model.py` 脚本，脚本将：
- 从 **input_bond_wave/** 文件夹中加载第一个 CSV 文件作为输入；
- 利用训练好的模型预测出对应的 CSAM 图；
- 显示输入的 bond wave 数据和预测的 CSAM 图。

运行命令：
```bash
python infer_model.py
```

## 自定义说明

- **数据替换**：  
  当你拥有真实数据时，只需将真实数据以相同格式（bond wave 数据为 CSV 文件，CSAM 图为 PNG 文件）保存，或修改数据加载部分代码即可。

- **模型调整**：  
  项目提供的模型结构为基础示例，可根据真实数据的特性修改模型结构、调整超参数，或尝试更复杂的网络（如卷积网络、编码器-解码器结构等）。

## 模型架构说明

在 `train_model.py` 中构建的模型流程为：
1. 将 52×1000 的输入数据展平。
2. 经过若干全连接层（含 Dropout 防止过拟合）。
3. 输出一个 256×256 的向量，再 reshape 成图像形式，并使用 Sigmoid 激活确保输出在 [0,1] 范围内。

该模型用于初步验证映射关系，后续可以根据效果进行优化。