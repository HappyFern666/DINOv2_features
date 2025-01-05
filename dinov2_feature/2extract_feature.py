import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

# 加载图像，调整大小到 448x448，转换为 fp32 并归一化到 (0,1)
transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Lambda(lambda x: x[[2, 1, 0]]),  # BGR to RGB
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))  # 使用新的归一化参数
])

# 创建一个形状为 [4, 3, 448, 448] 的输入张量
imgs_tensor = torch.zeros(4, 3, 448, 448, dtype=torch.float32)
for i in range(4):
    list_frame = [2049, 3565, 3596, 3609]
    img_path = f"/home/xuchuan/test_image/frame_{list_frame[i]}_rgb.png"
    img = Image.open(img_path).convert('RGB')
    imgs_tensor[i] = transform(img)[:3]

# 将图像张量移动到设备
device = "cuda" if torch.cuda.is_available() else "cpu"
imgs_tensor = imgs_tensor.to(device)

# 加载模型并运行前向传播
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
with torch.no_grad():
    result = dinov2_vitg14.forward_features(imgs_tensor)

# 从模型输出中获取 patch tokens
patch_features = result['x_prenorm'][:, 1:, :]

# 将 patch_features 移动到 CPU
patch_features_cpu = patch_features.cpu()

# 计算所有图像的第一个 PCA 分量
pca = PCA(n_components=1)
projected_features = pca.fit_transform(patch_features_cpu.reshape(-1, patch_features_cpu.shape[-1]))
norm_features = minmax_scale(projected_features)

# 使用阈值获取前景补丁掩码
foreground_mask = norm_features > 0.98  # 这里的阈值可以根据需要调整

# 确保 foreground_mask 是一维的
foreground_mask = foreground_mask.squeeze()

# 使用前景掩码索引 patch_features
foreground_features = patch_features.reshape(-1, patch_features.shape[-1])[foreground_mask]

# 将 foreground_features 移动到 CPU
foreground_features_cpu = foreground_features.cpu()

# 计算前景补丁的前三个 PCA 分量
# foreground_features = patch_features[foreground_mask]
pca = PCA(n_components=3)
pca_features_rem = pca.fit_transform(foreground_features_cpu)

# 归一化 PCA 输出
pca_features_rem = minmax_scale(pca_features_rem)

# 打印 patch_features 的形状以进行调试
print("patch_features shape:", patch_features.shape)

# 计算 patch 的数量
num_patches = patch_features.shape[1]

# 使用 PCA 结果作为 RGB 值
# pca_features_rgb = np.zeros_like(patch_features.cpu().numpy())
pca_features_rgb = np.zeros((patch_features.shape[0] * num_patches, 3))
pca_features_rgb[foreground_mask] = pca_features_rem

# 根据实际的 patch 数量调整重塑参数
patch_h = int(np.sqrt(num_patches))
patch_w = patch_h

# 确保重塑参数与实际数据大小匹配
pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)

# 可视化
# pca_features_rgb = pca_features_rgb.reshape(4, 40, 40, 3)  # 假设 patch_h 和 patch_w 为 40
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features_rgb[i][..., ::-1])
plt.savefig('features2.png')
plt.show()
plt.close()