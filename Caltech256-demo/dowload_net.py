import os

# 设置预训练模型保存路径
custom_model_path = './model'

# 设置 TORCH_HOME 环境变量
os.environ['TORCH_HOME'] = custom_model_path

# 下载并加载预训练模型
import torch
import torchvision.models as models

# 加载预训练的 VGG16 模型
vgg16 = models.vgg16(pretrained=True)
