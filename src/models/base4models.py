import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18', pretrained=True):
        super(BaseModel, self).__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name.startswith('resnet'):
            self.backbone = getattr(tv_models, self.backbone_name)(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.head = nn.Linear(self.feature_dim, num_classes)
        
        elif self.backbone_name.startswith('efficientnet'):
            self.backbone = timm.create_model(self.backbone_name, pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.head = nn.Linear(self.feature_dim, num_classes)
        
        # DenseNet 지원 추가!
        elif self.backbone_name.startswith('densenet'):
            self.backbone = getattr(tv_models, self.backbone_name)(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.head = nn.Linear(self.feature_dim, num_classes)
        
        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone_name}")

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out

# 사용 예시
num_classes = 100  # 예시
# ResNet18
model1 = BaseModel(num_classes=num_classes, backbone_name='resnet18')
# EfficientNet-B3
model2 = BaseModel(num_classes=num_classes, backbone_name='efficientnet_b3')

# 여기 모델에서 반복문으로 돌리기.
models = [
    'resnet18',
    'resnet101',         # 추가 추천!
    'efficientnet_b3',
    'efficientnet_b4',
    # 'densenet121',     # 보너스로 추가해도 좋음
    # 'vit_base_patch16_224',  # 최신 트렌드 실험용
]

