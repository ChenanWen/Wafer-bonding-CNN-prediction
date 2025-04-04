import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# 固定随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# 1. 加载训练好的模型
# -------------------------
model_path = "trained_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("未找到训练好的模型文件，请确保 'trained_model.h5' 存在。")
model = load_model(model_path)
print("训练好的模型已加载。")

# -------------------------
# 2. 加载待预测的 bond wave 数据
# -------------------------
# 新的文件夹名称为 "input_bond_wave"
csv_folder = "input_bond_wave"
csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
if len(csv_files) == 0:
    raise ValueError("未找到 CSV 文件，请检查文件夹：{}".format(csv_folder))
# 选取第一个 CSV 文件作为输入示例
csv_file = csv_files[0]
print("加载样本：", csv_file)

# 读取 CSV 文件中的 bond wave 数据，期望形状为 (52, 1000)
df = pd.read_csv(csv_file)
bond_wave = df.values.astype(np.float32)

# 对单个样本进行归一化（按该样本自身的最值归一化）
min_val = bond_wave.min()
max_val = bond_wave.max()
if max_val > min_val:
    bond_wave = (bond_wave - min_val) / (max_val - min_val)
else:
    bond_wave = bond_wave - min_val

# 增加 channel 维度，形状变为 (52, 1000, 1)
bond_wave = np.expand_dims(bond_wave, axis=-1)
# 增加 batch 维度，形状变为 (1, 52, 1000, 1)
bond_wave_input = np.expand_dims(bond_wave, axis=0)

# -------------------------
# 3. 使用模型进行预测
# -------------------------
prediction = model.predict(bond_wave_input)
predicted_csam = prediction[0]  # 预测结果形状为 (256, 256, 1)

# -------------------------
# 4. 可视化输入与预测结果
# -------------------------
plt.figure(figsize=(12, 4))

# 显示输入的 bond wave 数据（热图展示 52×1000 数据）
plt.subplot(1, 2, 1)
plt.imshow(bond_wave.squeeze(), aspect='auto', cmap='viridis')
plt.title('Bond Wave Data (CSV)')
plt.colorbar()

# 显示模型预测的 CSAM 图（灰度图）
plt.subplot(1, 2, 2)
plt.imshow(predicted_csam.squeeze(), cmap='gray', origin='lower')
plt.title('Predicted CSAM')
plt.colorbar()

plt.tight_layout()
plt.show()
