import os
import requests
import math
import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# 从官方默认值中抽取
INCEPTION_IMAGE_SIZE = 1008  # 14 * 72 = 1008, 输出特征图大小为 72x72
'''
如果输入图像大小是 (H, W)
Patch size 是 (patch_h, patch_w)
那么最终的特征图大小将是 (H/patch_h, W/patch_w)
'''
CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 添加输出目录常量
OUTPUT_DIR = "outputs"

def get_inception_transform(crop_size=INCEPTION_IMAGE_SIZE):
    """
    参考 dinov2/data/transforms.py 中官方的默认预处理流程:
    1. Resize (保持长宽比)
    2. CenterCrop 到指定大小 crop_size
    3. ToTensor (将 [0,255] 转成 [0,1])
    4. Normalize (减均值除方差)
    
    注意: 由于 ViT 模型的 patch_size=14,输入1008x1008会得到72x72的特征图
    """
    resize_size = int(math.floor(crop_size / CROP_PCT))
    return T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

# 如果想让这个函数和官方 "DEFAULT_TRANSFORMS" 等价，可以直接用:
DEFAULT_TRANSFORMS = get_inception_transform(INCEPTION_IMAGE_SIZE)

def load_image(image_path_or_url):
    """
    自动判断输入是本地路径还是 URL，返回 RGB 格式的 PIL.Image 对象。
    """
    if image_path_or_url.startswith("http"):
        # 如果是 URL
        resp = requests.get(image_path_or_url, stream=True)
        resp.raise_for_status()
        img = Image.open(resp.raw).convert('RGB')
    else:
        # 如果是本地路径
        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"File not found: {image_path_or_url}")
        img = Image.open(image_path_or_url).convert('RGB')
    return img

def extract_features(model, img_tensor):
    """
    从 DinoV2 模型中获取每个像素点的特征。
    由于 patch_size=14,输入1008x1008的图像会得到72x72的特征图。
    返回大小 [72, 72, feature_dim] 的张量。
    """
    try:
        with torch.no_grad():
            print(f"Input tensor shape: {img_tensor.shape}") # [1, 3, 1008, 1008]
            
            # 使用 forward_features 获取特征
            features = model.forward_features(img_tensor)
            features = features['x_prenorm']  # 获取预归一化特征
            
            
            # 重塑特征以匹配图像尺寸
            B, N, D = features.shape  # [batch_size, num_tokens, feature_dim]
            H = W = int(np.sqrt(N-1))  # 减1是因为第一个token是[CLS], 得到72x72
            
            # 移除[CLS] token并重塑
            features = features[:, 1:, :].reshape(B, H, W, D)
            features = features[0]  # 移除batch维度
            
            print(f"Final features shape: {features.shape} (72x72 feature map)")
            return features
            
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        raise

def visualize_features(features, output_size=(1008, 1008), n_clusters=5):
    """
    对每个像素点的特征进行聚类分析和可视化。
    注意:输入特征图为72x72,会被上采样到output_size大小。
    n_clusters: 聚类的数量,用于区分不同的物体部分(如瓶盖、瓶身等)
    """
    print(f"\nFeature Visualization:")
    print(f"Input features shape: {features.shape} (72x72 low-resolution feature map)")
    
    # 转换到CPU并转为numpy数组
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    
    H, W, D = features.shape
    features_flat = features.reshape(-1, D)
    
    # 使用KMeans进行特征聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_flat)
    
    # 对每个聚类进行单独的PCA分析和可视化
    cluster_colors = np.zeros((H*W, 3))
    
    for cluster_id in range(n_clusters):
        # 获取当前聚类的特征
        cluster_mask = cluster_labels == cluster_id
        cluster_features = features_flat[cluster_mask]
        
        if len(cluster_features) > 0:
            # 对当前聚类进行PCA
            pca = PCA(n_components=3)
            cluster_pca = pca.fit_transform(cluster_features)
            
            # 归一化到[0,1]
            cluster_min = cluster_pca.min(axis=0, keepdims=True)
            cluster_max = cluster_pca.max(axis=0, keepdims=True)
            cluster_norm = (cluster_pca - cluster_min) / (cluster_max - cluster_min + 1e-9)
            
            # 将颜色分配给对应的像素
            cluster_colors[cluster_mask] = cluster_norm
    
    # 重塑为图像格式 (72x72x3)
    rgb_img = cluster_colors.reshape(H, W, 3)
    
    # 上采样到目标大小 (默认1008x1008)
    rgb_img_pil = Image.fromarray((rgb_img * 255).astype(np.uint8))
    rgb_img_pil = rgb_img_pil.resize(output_size, Image.Resampling.LANCZOS)
    rgb_img = np.array(rgb_img_pil)
    
    # 可视化原始特征图
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Feature-based Part Segmentation", fontsize=14)
    plt.axis("off")
    
    # 可视化聚类标签图
    plt.subplot(1, 2, 2)
    cluster_map = cluster_labels.reshape(H, W)
    plt.imshow(cluster_map, cmap='tab10')
    plt.title("Cluster Labels", fontsize=14)
    plt.axis("off")
    
    plt.tight_layout()
    
    # 创建输出目录(如果不存在)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "semantic_features.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"保存可视化结果到: {output_path}")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载 DinoV2 模型 (patch_size=14)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
    model.eval()
    
    # 2. 读取图像并做预处理
    image_path = "test_image/test2.jpg"
    image = load_image(image_path)
    print(f"Original image size: {image.size}")
    
    img_tensor = DEFAULT_TRANSFORMS(image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {img_tensor.shape}")
    
    # 3. 提取特征 (得到72x72特征图)
    features = extract_features(model, img_tensor)
    print(f"Features shape: {features.shape} (72x72 feature map)")
    
    # 4. 可视化特征 (上采样到1008x1008)
    visualize_features(features, output_size=(1008, 1008), n_clusters=10)

if __name__ == "__main__":
    main()