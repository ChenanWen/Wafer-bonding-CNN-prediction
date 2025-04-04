import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def simulate_bond_wave_data(time_steps=1000, num_sensors=52):
    """
    模拟单个样本的 bond wave 数据
    输出 shape: (num_sensors, time_steps)
    每个传感器生成一个正弦波加噪声
    """
    time = np.linspace(0, 10, time_steps)
    sensor_data = []
    for _ in range(num_sensors):
        frequency = np.random.uniform(0.5, 2.0)  # 随机频率
        amplitude = np.random.uniform(0.1, 1.0)  # 随机振幅
        phase = np.random.uniform(0, 2 * np.pi)  # 随机相位
        noise = np.random.normal(0, 0.05, size=time.shape)  # 添加噪声
        wave = amplitude * np.sin(2 * np.pi * frequency * time + phase) + noise
        sensor_data.append(wave)
    sensor_data = np.array(sensor_data)  # (num_sensors, time_steps)
    return sensor_data


def simulate_csam_image(image_size=256, wafer_radius=100, boundary_thickness=2):
    """
    模拟单个 CSAM 图
    输出 shape: (image_size, image_size)
    0 表示正常区域，1 表示 non bond 缺陷区域
    在晶圆边缘绘制白色边界，并在内部随机添加几个白色斑点表示缺陷
    """
    csam = np.zeros((image_size, image_size), dtype=float)
    center = image_size // 2  # 晶圆中心

    # 绘制晶圆边缘（白色环）
    for i in range(image_size):
        for j in range(image_size):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if wafer_radius - boundary_thickness <= dist <= wafer_radius:
                csam[i, j] = 1.0

    # 在晶圆内部随机添加若干白色斑点（缺陷区域）
    num_spots = np.random.randint(3, 10)
    for _ in range(num_spots):
        spot_radius = np.random.randint(3, 8)
        # 确保斑点在晶圆内部（避开边缘）
        spot_center_x = np.random.randint(center - wafer_radius + spot_radius, center + wafer_radius - spot_radius)
        spot_center_y = np.random.randint(center - wafer_radius + spot_radius, center + wafer_radius - spot_radius)
        Y, X = np.ogrid[:image_size, :image_size]
        mask = (X - spot_center_x) ** 2 + (Y - spot_center_y) ** 2 <= spot_radius ** 2
        csam[mask] = 1.0

    return csam


if __name__ == '__main__':
    num_samples = 10  # 生成 10 个样本供查看
    # 创建用于保存数据的文件夹
    os.makedirs("bond_wave_data", exist_ok=True)
    os.makedirs("csam_images", exist_ok=True)

    for i in range(num_samples):
        # 生成模拟数据
        bond_wave = simulate_bond_wave_data()
        csam = simulate_csam_image()

        # 保存 bond wave 数据到 CSV 文件，每行代表一个传感器
        df = pd.DataFrame(bond_wave)
        csv_filename = os.path.join("bond_wave_data", f"bond_wave_sample_{i + 1}.csv")
        df.to_csv(csv_filename, index=False)

        # 保存 CSAM 图为 PNG 图片
        plt.figure(figsize=(6, 6))
        plt.imshow(csam, cmap='gray', origin='lower')
        plt.title(f'Simulated CSAM Sample {i + 1}')
        plt.axis('off')
        png_filename = os.path.join("csam_images", f"csam_sample_{i + 1}.png")
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"样本 {i + 1}: Bond wave 数据已保存到 {csv_filename}，CSAM 图已保存到 {png_filename}")

    print("所有模拟数据已生成，请查看 'bond_wave_data' 和 'csam_images' 文件夹。")
