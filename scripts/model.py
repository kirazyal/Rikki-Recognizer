import torch.nn as nn
import torchvision.models as models


class TakiClassifier(nn.Module):
    """椎名立希识别模型 - 二分类"""

    def __init__(self, num_classes=2):  # 二分类：是立希/不是立希
        super(TakiClassifier, self).__init__()
        # 使用预训练的ResNet18（迁移学习）
        self.backbone = models.resnet18(pretrained=True)

        # 替换最后一层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
            # 注意：CrossEntropyLoss会自动处理，不需要Sigmoid
        )

    def forward(self, x):
        return self.backbone(x)