import os
import requests
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# from dinov2.models import dinov2_vits14      # 以 ViT-S/14 为例
from dinov2.data.transforms import DEFAULT_TRANSFORMS


def load_image(image_path_or_url):
    """
    根据输入字符串是本地路径还是 URL 来加载图像（PIL 格式）。
    如果加载失败，抛出异常。
    """
    try:
        if image_path_or_url.startswith("http"):
            # 如果是 URL
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()  # 如果下载出错会抛异常
            image = Image.open(response.raw).convert('RGB')
        else:
            # 如果是本地路径
            if not os.path.exists(image_path_or_url):
                raise FileNotFoundError(f"File not found: {image_path_or_url}")
            image = Image.open(image_path_or_url).convert('RGB')
        return image
    except Exception as e:
        print(f"[Error] Failed to load image: {e}")
        raise


def visualize_dinov2_attention(
    image_path_or_url,
    model=None,
    device=None,
    layer_idx=-1,
    head_idx=None,
    alpha=0.5,
    show_plot=True
):
    """
    加载图像并对其进行 DinoV2 注意力图可视化。

    参数：
    -------
    image_path_or_url : str
        图像的本地路径或 URL。
    model : torch.nn.Module (可选)
        若已实例化好的 DinoV2 模型，可传入；若为 None，将自动加载预训练的 dinov2_vits14。
    device : torch.device (可选)
        指定运行设备；若为 None，则自动检测可用的 GPU，否则使用 CPU。
    layer_idx : int
        指定可视化第几层的注意力（默认 -1 表示最后一层）。
    head_idx : int or None
        指定可视化哪一个 head；若为 None，则对该层的所有 head 求平均。
    alpha : float
        叠加热力图时的透明度，取值 [0, 1]。
    show_plot : bool
        若为 True，则直接调用 plt.show() 显示图像；若为 False，仅返回 figure 对象。

    返回：
    -------
    fig : matplotlib.figure.Figure
        若 show_plot=False，返回绘图的 Figure 对象；若 show_plot=True，返回 None。
    """
    # 0. 设备与模型准备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
        model.eval()

    # 1. 加载并预处理图像
    try:
        image = load_image(image_path_or_url)
    except Exception as e:
        print("[Error] Cannot proceed without a valid image.")
        return None

    img_tensor = DEFAULT_TRANSFORMS(image).unsqueeze(0).to(device)

    # 2. 前向传播并获取注意力权重
    #    使用 return_all_layers=True 来获得额外的注意力权重
    with torch.no_grad():
        try:
            out_dict = model.forward_flexibly(img_tensor, return_all_layers=True)
            attn = out_dict["attn_head_weights"]  # [B, L, H, S, S]
        except Exception as e:
            print(f"[Error] Failed to run inference on the model: {e}")
            return None

    # 3. 根据指定层与 head 选择注意力权重
    #    - batch_size = 1, 所以取 attn[0]
    #    - num_layers = attn[0].shape[0]
    #    - num_heads = attn[0].shape[1]
    #    - seq_len = attn[0].shape[2]
    attn_specific_layer = attn[0, layer_idx]  # [num_heads, seq_len, seq_len]

    if head_idx is not None:
        # 如果指定了具体的 head，就只可视化这一头
        if head_idx < 0 or head_idx >= attn_specific_layer.shape[0]:
            print(f"[Error] Invalid head_idx: {head_idx}. "
                  f"Must be in [0, {attn_specific_layer.shape[0]-1}].")
            return None
        attn_map = attn_specific_layer[head_idx]  # [seq_len, seq_len]
    else:
        # 否则对所有头求平均
        attn_map = attn_specific_layer.mean(dim=0)  # [seq_len, seq_len]

    # 4. 去掉 [CLS] 的行/列，只保留对图像 patch 的注意力
    #    对于 ViT-S/14, 输入为 224×224，patch_size=14×16；seq_len = 1 + H*W
    attn_map_no_cls = attn_map[1:, 1:]  # [H*W, H*W]
    num_patches = attn_map_no_cls.shape[0]
    h_feat = w_feat = int(num_patches ** 0.5)
    attn_map_no_cls = attn_map_no_cls.reshape(h_feat, w_feat).cpu().numpy()

    # 5. 将注意力图 resize 到与原图相同尺寸
    attn_resized = cv2.resize(
        attn_map_no_cls, (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC
    )

    # 6. 绘图展示
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image", fontsize=14)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    title_str = (
        f"Layer {layer_idx}, Head {head_idx}" if head_idx is not None
        else f"Layer {layer_idx}, All Heads Mean"
    )
    plt.title(f"Attention Map - {title_str}", fontsize=14)
    plt.imshow(image)
    plt.imshow(attn_resized, cmap="jet", alpha=alpha)
    plt.axis("off")

    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    else:
        return fig


if __name__ == "__main__":
    """
    使用示例：
    1. 可视化最后一层、所有 heads 的平均注意力：
       visualize_dinov2_attention("http://xxx/your_image.jpg")
    2. 可视化第2层、第0号 head 的注意力（alpha设为0.6）：
       visualize_dinov2_attention(
           "http://xxx/your_image.jpg",
           layer_idx=1, head_idx=0, alpha=0.6
       )
    """
    # 如果你想直接测试，可以用下面这一行：
    visualize_dinov2_attention(
        image_path_or_url="/home/xuchuan/DINOv2_features/dinov2_feature/test_image/frame_2049_rgb.png",
        layer_idx=-1,   # 最后一层
        head_idx=None,  # 对所有 heads 求平均
        alpha=0.5,      # 注意力热力图透明度
        show_plot=True
    )