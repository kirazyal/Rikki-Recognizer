import copy
import time
import os

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import TakiClassifier
import torch.nn as nn
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_dataloader(data_subfolder, batch_size=32, shuffle=True):
    """根据子文件夹名获取DataLoader，并显示类别信息"""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_path = f'D:/AI_Projects/taki_recognizer/data/raw/{data_subfolder}'

    full_dataset = ImageFolder(
        root=data_path,
        transform=transform
    )

    # 显示类别信息（关键！）
    print(f"📂 文件夹 '{data_subfolder}' 包含类别: {full_dataset.classes}")
    print(f"📂 类别到索引的映射: {full_dataset.class_to_idx}")

    # 划分训练集(80%)和验证集(20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = Data.random_split(full_dataset, [train_size, val_size])

    print(f"📊 {data_subfolder} - 总图片: {len(full_dataset)}张 | 训练: {train_size} | 验证: {val_size}")

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader


def train_phase(model, train_loader, val_loader, num_epochs, phase_name, lr=0.0001):
    """单个阶段的训练函数"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    if device.type == 'cuda':
        print(f"🎮 显卡: {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    print(f"\n{'=' * 50}")
    print(f"🚀 开始阶段 {phase_name}")
    print(f"{'=' * 50}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} - {phase_name}")
        print("-" * 30)

        # 训练
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

            if step % 10 == 0:
                print(f"  Step {step:3d}/{len(train_loader):3d} | Loss: {loss.item():.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        # 计算指标
        train_loss_epoch = train_loss / train_num
        train_acc_epoch = train_corrects.double() / train_num
        val_loss_epoch = val_loss / val_num
        val_acc_epoch = val_corrects.double() / val_num

        train_loss_all.append(train_loss_epoch)
        train_acc_all.append(train_acc_epoch)
        val_loss_all.append(val_loss_epoch)
        val_acc_all.append(val_acc_epoch)

        print(f"\n✅ {phase_name} Epoch {epoch + 1} 结果:")
        print(f"  Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.4f}")
        print(f"  Val Loss: {val_loss_epoch:.4f} | Val Acc: {val_acc_epoch:.4f}")

        # 保存最佳模型
        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  ⭐ 保存最佳模型 (Acc: {best_acc:.4f})")

    # 加载本阶段最佳模型
    model.load_state_dict(best_model_wts)
    return model, train_loss_all, val_loss_all, train_acc_all, val_acc_all


def plot_phase_curves(phase1_history, phase2_history):
    """绘制两个阶段的训练曲线"""

    plt.figure(figsize=(14, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    epochs1 = range(1, len(phase1_history['train_loss']) + 1)
    epochs2 = range(len(phase1_history['train_loss']) + 1,
                    len(phase1_history['train_loss']) + len(phase2_history['train_loss']) + 1)

    plt.plot(epochs1, phase1_history['train_loss'], 'b-', label="Phase1 Train Loss", linewidth=2)
    plt.plot(epochs1, phase1_history['val_loss'], 'b--', label="Phase1 Val Loss", linewidth=2)
    plt.plot(epochs2, phase2_history['train_loss'], 'r-', label="Phase2 Train Loss", linewidth=2)
    plt.plot(epochs2, phase2_history['val_loss'], 'r--', label="Phase2 Val Loss", linewidth=2)
    plt.axvline(x=len(epochs1), color='gray', linestyle=':', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("两阶段损失曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs1, phase1_history['train_acc'], 'b-', label="Phase1 Train Acc", linewidth=2)
    plt.plot(epochs1, phase1_history['val_acc'], 'b--', label="Phase1 Val Acc", linewidth=2)
    plt.plot(epochs2, phase2_history['train_acc'], 'r-', label="Phase2 Train Acc", linewidth=2)
    plt.plot(epochs2, phase2_history['val_acc'], 'r--', label="Phase2 Val Acc", linewidth=2)
    plt.axvline(x=len(epochs1), color='gray', linestyle=':', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("两阶段准确率曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../models/two_phase_training.png', dpi=150)
    plt.show()
    print("📊 两阶段训练曲线已保存")


if __name__ == "__main__":
    print("🚀 开始椎名立希两阶段训练")
    print("=" * 50)

    # 创建模型
    model = TakiClassifier(num_classes=2)
    print(f"📋 模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # 阶段一：用official文件夹训练（官方立希图）
    print("\n📁 阶段一数据准备：official文件夹")
    train_loader1, val_loader1 = get_dataloader('official', batch_size=32)

    model, loss1, val_loss1, acc1, val_acc1 = train_phase(
        model, train_loader1, val_loader1,
        num_epochs=20, phase_name="官方图训练", lr=0.0001
    )

    # 保存阶段一模型
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/taki_phase1.pth')
    print(f"\n💾 阶段一模型已保存: ../models/taki_phase1.pth")

    # 阶段二：用taki文件夹继续训练（所有立希图，包括官方和同人）
    print("\n📁 阶段二数据准备：taki文件夹（包含所有立希图）")
    train_loader2, val_loader2 = get_dataloader('taki', batch_size=32)

    # 降低学习率进行微调
    model, loss2, val_loss2, acc2, val_acc2 = train_phase(
        model, train_loader2, val_loader2,
        num_epochs=30, phase_name="全部图微调", lr=0.00005
    )

    # 保存最终模型
    torch.save(model.state_dict(), '../models/taki_final.pth')
    print(f"\n💾 最终模型已保存: ../models/taki_final.pth")

    # 绘制曲线
    phase1_history = {
        'train_loss': [l.item() if torch.is_tensor(l) else l for l in loss1],
        'val_loss': [l.item() if torch.is_tensor(l) else l for l in val_loss1],
        'train_acc': [a.item() if torch.is_tensor(a) else a for a in acc1],
        'val_acc': [a.item() if torch.is_tensor(a) else a for a in val_acc1]
    }

    phase2_history = {
        'train_loss': [l.item() if torch.is_tensor(l) else l for l in loss2],
        'val_loss': [l.item() if torch.is_tensor(l) else l for l in val_loss2],
        'train_acc': [a.item() if torch.is_tensor(a) else a for a in acc2],
        'val_acc': [a.item() if torch.is_tensor(a) else a for a in val_acc2]
    }

    plot_phase_curves(phase1_history, phase2_history)

    print("\n" + "=" * 50)
    print("🎉 两阶段训练完成！")
    print("📁 阶段一模型: ../models/taki_phase1.pth")
    print("📁 最终模型: ../models/taki_final.pth")
    print("📊 训练曲线: ../models/two_phase_training.png")