import torch.nn as nn
import torch
import torchvision.models as models
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np
from dataset import Caltech256Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
#net
EfficientNet = models.efficientnet_b0()
# 从.pth文件中加载预训练参数
EfficientNet.load_state_dict(torch.load(r'model/EfficientNetb0.pth',map_location='cpu'))
# 改写网络结构
EfficientNet.classifier._modules['1'] = nn.Linear(1280,257)

#改写网络结构
# EfficientNet.classifier._modules['1'] = nn.Linear(1280,257)

#数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_dataset = Caltech256Dataset(root_dir='Caltech256',train=True,transforms=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = Caltech256Dataset(root_dir='Caltech256',train=False,transforms=data_transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#损失函数
criterion = nn.CrossEntropyLoss()
#搬运数据
EfficientNet.to(device)
#优化器
optimizer = optim.SGD(EfficientNet.parameters(), lr=0.001, momentum=0.9)
#学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#train
log_folder = 'log_efficientNetb0.txt' # 指定 log 文件夹路径

# 定义一个函数来测试模型的准确率
def test_accuracy(model, test_dataloader, device=torch.device('cuda')):
    correct = 0
    total = 0
    with torch.no_grad():  # 停用梯度计算
        for inputs, labels in tqdm.tqdm(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

# 记录每个batch的loss到log中
def log_loss(log_folder, epoch, batch_idx, loss):
    with open(log_folder, 'a') as f:
        f.write(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss:.4f}\n")

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(train_dataloader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = EfficientNet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 记录每个 batch 的 loss
        log_loss(log_folder, epoch, batch_idx, loss)
        
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # 测试模型准确率
    accuracy = test_accuracy(EfficientNet, test_dataloader, device)
    print(f"Epoch {epoch+1} accuracy on train set: {accuracy:.2f}%")

    scheduler.step()

print("Training finished.")
save_path = 'model/EfficientNetb0_out2.pth'
torch.save(EfficientNet.state_dict(), save_path)