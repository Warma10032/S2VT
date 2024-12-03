import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import imageio
import skimage.transform


# 读取并处理输入
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(VGG16FeatureExtractor, self).__init__()
        self.features = model.features  # VGG16卷积部分
        self.classifier = model.classifier[:6]  # 保留前6层全连接层

    def forward(self, x):
        # 获取卷积层输出
        x = self.features(x)

        # 展平卷积层的输出为(batch_size, 25088)
        x = x.view(x.size(0), -1)  # batch_size, 512 * 7 * 7 -> batch_size, 25088

        # 将展平后的特征传递给全连接层
        x = self.classifier(x)
        return x


def extract_features(namelist, batch_size, device, save_dir):
    """Extract features for a single video."""
    # 加载预训练的VGG-16模型
    model = VGG16FeatureExtractor(models.vgg16(pretrained=True))
    model.eval()  # 设置为评估模式
    model.to(device)  # 移动模型到GPU或CPU

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ]
    )

    for file in namelist:
        # 读取视频
        vid = imageio.get_reader(f"dataset/video/{file}", "ffmpeg")
        curr_frames = []

        for frame in vid:
            # 调整帧大小
            frame = skimage.transform.resize(frame, [224, 224])
            if len(frame.shape) < 3:
                frame = np.repeat(frame, 3).reshape([224, 224, 3])
            frame = Image.fromarray((frame * 255).astype(np.uint8))
            curr_frames.append(preprocess(frame))

        curr_frames = torch.stack(curr_frames).to(device)  # 将图像移动到 GPU/CPU

        idx = np.linspace(0, len(curr_frames) - 1, 80).astype(int)  # 获取80帧
        curr_frames = curr_frames[idx]

        curr_feats = []
        for i in range(0, 80, batch_size):
            curr_batch = curr_frames[i : i + batch_size]
            with torch.no_grad():
                features = model(curr_batch)
                curr_feats.append(features.cpu().numpy())

        curr_feats = np.concatenate(curr_feats, axis=0)
        save_path = os.path.join(save_dir, f"{file[:-4]}.npy")
        np.save(save_path, curr_feats)
        print(f"Saved features for {file} to {save_path}")


def remove_featured_video(save_dir, namelist):
    """根据save_dir中最后一个保存的文件名，删除namelist中该文件名前的内容"""
    # 获取save_dir目录下所有.npy文件
    npy_files = [f for f in os.listdir(save_dir) if f.endswith(".npy")]

    if not npy_files:
        return namelist  # 如果没有保存的文件，直接返回原始namelist

    # 排序并选择最新的文件
    latest_file = max(
        npy_files, key=lambda f: os.path.getmtime(os.path.join(save_dir, f))
    )

    # 提取视频名称，不包括扩展名
    latest_video_name = latest_file.replace(".npy", ".avi")  # 去除 ".npy" 后缀

    # 从namelist中删除latest_video_name之前的所有文件
    if latest_video_name in namelist:
        video_index = namelist.index(latest_video_name)
        filtered_namelist = namelist[video_index:]  # 保留从video_name开始的文件
    else:
        filtered_namelist = namelist  # 如果找不到，直接返回原始namelist

    return filtered_namelist


if __name__ == "__main__":
    # 获取视频文件列表
    video_dir = "dataset/video"
    save_dir = "dataset/features"
    namelist = os.listdir(video_dir)

    # 检查GPU是否可用，如果可用则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 过滤namelist，删除最后保存文件前的视频名称
    namelist = remove_featured_video(save_dir, namelist)
    print(f"Filtered namelist: {namelist[0]}")

    # 创建保存特征的目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取特征并保存
    extract_features(namelist, 10, device, save_dir)
