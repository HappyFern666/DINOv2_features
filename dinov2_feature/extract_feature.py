import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# PCA for feature inferred
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

patch_h = 40 # patch_num
patch_w = 40 # patch_num
# feat_dim = 384 # vits14
# feat_dim = 768 # vitb14
feat_dim = 1024 # vitl14
# feat_dim = 1536 # vitg14

'''
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
'''
transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)), # (560, 560)
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Lambda(lambda x: x[[2, 1, 0]]), # RGB to BGR
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
# add device
dinov2_vitl14 = dinov2_vitl14.half()  # 添加这行
device = "cuda" if torch.cuda.is_available() else "cpu"
dinov2_vitl14 = dinov2_vitl14.to(device)


print(dinov2_vitl14)

# extract features
# imgs_tensor = torch.zeros(4, 3, 448, 448, dtype=torch.float32)
features = torch.zeros(4, patch_h * patch_w, feat_dim, device=device, dtype=torch.uint8)
# 4 images, (4, 40*40, 1024)
original_images = []
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14)

for i in range(4):
    list_frame = [2049, 3565, 3596, 3609]
    # img_path = f"/home/xuchuan/DINOv2_features/dinov2_feature/test_image/frame_{list_frame[i]}_rgb.png"
    img_path = f"/home/xuchuan/DINOv2_features/dinov2_feature/self_test/test{i}.jpg"
    img = Image.open(img_path).convert('RGB')
    # img_rgb = np.array(img)[..., ::-1]
    original_images.append(img)
    print(f"图片格式信息：")
    print(f"PIL格式: {img.mode}")
    print(f"数据类型: {np.array(img).dtype}")
    print(f"值范围: [{np.array(img).min()}, {np.array(img).max()}]")
    imgs_tensor[i] = transform(img)[:3]

# to device
imgs_tensor = imgs_tensor.to(device).half()

with torch.no_grad():
    features_dict = dinov2_vitl14.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']



features = features.cpu().float()

features = features.reshape(4 * patch_h * patch_w, feat_dim) # (6400, 1024)

pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features) # (6400, 3)


# uncomment below to plot the first pca component
# pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())

# segment using the first component 用第一主成分分割前景后景
pca_features_bg = pca_features[:, 0] > 4 # 背景
pca_features_fg = ~pca_features_bg # 前景

# PCA for only foreground patches
pca_features_rem = pca.transform(features[pca_features_fg])
for i in range(3):
    # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
    # transform using mean and std, I personally found this transformation gives a better visualization
    pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

pca_features_rgb = pca_features.copy()
pca_features_rgb[pca_features_bg] = 0
pca_features_rgb[pca_features_fg] = pca_features_rem

pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)

# 对特征可视化结果进行归一化
def normalize_for_display(img):
    img = img - img.min()  # 移到0为最小值
    img = img / img.max()  # 缩放到[0,1]范围
    return img

# 创建一个大图，2行4列布局
plt.figure(figsize=(20, 10))

# 第一行显示原始图片
for i in range(4):
    plt.subplot(2, 4, i+1)
    img_rgb = np.array(original_images[i])
    # 假设原始图片存储在变量 original_images 中
    plt.imshow(img_rgb)  # 如果是PIL图像对象，直接使用
    # 或者使用 plt.imshow(np.array(original_images[i])) 
    plt.title(f'Original Image {i+1}')
    plt.axis('off')

# 第二行显示特征可视化结果
for i in range(4):
    plt.subplot(2, 4, i+5)  # 使用5,6,7,8位置
    plt.imshow(normalize_for_display(pca_features_rgb[i][..., ::-1]))
    plt.title(f'Feature Visualization {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('photo_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()