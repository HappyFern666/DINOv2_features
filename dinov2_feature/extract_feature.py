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
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14)
for i in range(4):
    list_frame = [2049, 3565, 3596, 3609]
    img_path = f"/home/xuchuan/test_image/frame_{list_frame[i]}_rgb.png"
    # img_path = f"/home/xuchuan/test_dino/image{i}.png"
    img = Image.open(img_path).convert('RGB')
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
pca_features_bg = pca_features[:, 0] > 10 # 背景
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

# ... 前面的代码保持不变 ...

# 创建一个大图，包含所有可视化内容
plt.figure(figsize=(20, 10))

# 1. PCA组件的直方图 (第一行左侧)
plt.subplot(2, 3, 1)
plt.hist(pca_features[:, 0])
plt.title('PCA Component 1')
plt.subplot(2, 3, 2)
plt.hist(pca_features[:, 1])
plt.title('PCA Component 2')
plt.subplot(2, 3, 3)
plt.hist(pca_features[:, 2])
plt.title('PCA Component 3')

# 2. 前景/背景分割结果 (第二行左侧)
plt.subplot(2, 3, 4)
for i in range(4):
    plt.subplot(2, 6, 7+i)  # 在第二行的左半部分
    plt.imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))
    plt.title(f'Seg Img {i+1}')
    plt.axis('off')

# 3. 最终的特征可视化 (第二行右侧)
for i in range(4):
    plt.subplot(2, 6, 10+i)  # 在第二行的右半部分
    plt.imshow(pca_features_rgb[i][..., ::-1])
    plt.title(f'Feature Img {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('all_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
