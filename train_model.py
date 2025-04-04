import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 固定随机种子，便于复现
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# 1. 加载数据
# -------------------------
# 定义数据文件夹路径
csv_folder = "bond_wave_data"
png_folder = "csam_images"

# 获取所有 CSV 文件（bond wave 数据）和 PNG 文件（CSAM 图），并按文件名排序
csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
png_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))

# 检查数量是否一致
if len(csv_files) != len(png_files):
    raise ValueError("CSV 文件和 PNG 文件数量不一致，请检查数据！")

X_list = []
Y_list = []

# 循环加载每个样本数据
for csv_file, png_file in zip(csv_files, png_files):
    # 加载 bond wave 数据
    df = pd.read_csv(csv_file)
    # 得到 numpy 数组，shape: (52, 1000)
    bond_wave = df.values.astype(np.float32)
    # 对单个样本做归一化（每个样本按其自身最值归一化）
    min_val = bond_wave.min()
    max_val = bond_wave.max()
    if max_val > min_val:
        bond_wave = (bond_wave - min_val) / (max_val - min_val)
    else:
        bond_wave = bond_wave - min_val
    # 增加一个 channel 维度，变成 (52, 1000, 1)
    bond_wave = np.expand_dims(bond_wave, axis=-1)
    X_list.append(bond_wave)

    # 加载 CSAM 图像
    img = Image.open(png_file).convert("L")  # 转为灰度图
    csam = np.array(img).astype(np.float32)
    # 如果图像是 8 位，通常取值范围 0-255，归一化到 [0,1]
    csam = csam / 255.0
    # 确保尺寸为 (256, 256)（如果有差异，可进行resize）
    if csam.shape != (256, 256):
        csam = np.array(img.resize((256, 256))).astype(np.float32) / 255.0
    # 增加一个 channel 维度，变成 (256, 256, 1)
    csam = np.expand_dims(csam, axis=-1)
    Y_list.append(csam)

# 转换为 numpy 数组
X_data = np.array(X_list)
Y_data = np.array(Y_list)

print("Loaded bond wave data shape:", X_data.shape)  # (样本数, 52, 1000, 1)
print("Loaded CSAM images shape:", Y_data.shape)  # (样本数, 256, 256, 1)

# -------------------------
# 2. 数据划分：训练集、验证集、测试集
# -------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

print("Train set:", X_train.shape, Y_train.shape)
print("Validation set:", X_val.shape, Y_val.shape)
print("Test set:", X_test.shape, Y_test.shape)

# -------------------------
# 3. 构建模型
# -------------------------
# 此模型将 52×1000 的输入数据展平后通过全连接层映射到 256×256 图像
input_shape = (52, 1000, 1)
output_size = 256 * 256  # 输出图像总像素数

inputs = Input(shape=input_shape)
x = Flatten()(inputs)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(8192, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(output_size, activation='sigmoid')(x)  # 输出范围 [0,1]
outputs = Reshape((256, 256, 1))(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# 4. 模型训练
# -------------------------
epochs = 50
batch_size = 16

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=epochs,
                    batch_size=batch_size)

# -------------------------
# 5. 模型评估与保存
# -------------------------
loss, acc = model.evaluate(X_test, Y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

# 保存训练好的模型
model.save("trained_model.h5")
print("模型已保存为 trained_model.h5")
