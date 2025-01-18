import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件，假设文件中有 'x', 'y', 'z', 'intensity' 四个字段
df = pd.read_csv('/media/viewer/Image_Lab/embod_data/MarsMind_data/sample/episode_31/1_move_forward/pc/2025_01_07-15_47_33.747903.csv')

# 随机采样4096个点
sampled_df = df.sample(n=4096, random_state=42)

# 提取采样后的点云数据
x = sampled_df['x']
y = sampled_df['y']
z = sampled_df['z']

# 创建一个 3D 图形
fig = plt.figure(figsize=(10, 7))

# 定义视角列表
view_angles = [
    (30, 30),  # 第一个视角 (azim, elev)
    (90, 30),  # 第二个视角
    (180, 30), # 第三个视角
    (270, 30), # 第四个视角
    (0, 90)    # 第五个视角
]

# 绘制每个视角并保存
for i, (azim, elev) in enumerate(view_angles):
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, z, c='b', s=1)

    # 设置标题和标签
    ax.set_title(f'LiDAR Point Cloud - View {i+1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    # 保存图像
    plt.savefig(f'/home/zhangyusi/Improved-3D-Diffusion-Policy/scripts/vis_test_view_{i+1}.png', dpi=300)

    # 清除当前图像，以便绘制下一个视角
    ax.cla()