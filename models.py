# REF. SWIN-TRANSFORMER: https://python.plainenglish.io/swin-transformer-from-scratch-in-pytorch-31275152bf03

import timm
import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class ResNetModel(nn.Module):
    def __init__(self, num_classes, resnet_version='resnet18', checkpoint_path=None, binary_classification=False):
        super(ResNetModel, self).__init__()
        self.binary_classification = binary_classification
        
        self.model = getattr(models, resnet_version)(pretrained=True if checkpoint_path is None else False) # Load a pretrained ResNet model
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes) # Replace the classifier layer

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path) # Load checkpoint file

    def forward(self, x):
        if self.binary_classification:
            return torch.sigmoid(self.model(x))
        else:
            return self.model(x)
    
    def load_checkpoint(self, checkpoint_path):
        weights = torch.load(checkpoint_path)
        self.model.load_state_dict(weights)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, efficientnet_version='efficientnet-b0', checkpoint_path=None, binary_classification=False):
        super(EfficientNetModel, self).__init__()
        self.binary_classification = binary_classification

        if checkpoint_path is None:
            self.model = EfficientNet.from_pretrained(efficientnet_version) # Load a pretrained EfficientNet model
        else:
            self.model = EfficientNet.from_name(efficientnet_version) # Load without pretrained weights

        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, num_classes) # Replace the classifier layer

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path) # Load checkpoint file

    def forward(self, x):
        if self.binary_classification:
            return torch.sigmoid(self.model(x))
        else:
            return self.model(x)
    
    def load_checkpoint(self, checkpoint_path):
        weights = torch.load(checkpoint_path)
        self.model.load_state_dict(weights)

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='swin_tiny_patch4_window7_224'):
        super(SwinTransformerModel, self).__init__()
        self.swin_transformer = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.swin_transformer(x)

