import torch
from torch.nn import BatchNorm2d, Softmax2d
from torchvision.models.resnet import ResNet
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple
from torch import nn
import torch.nn.functional as F

class Baseline(torch.nn.Module):
    def __init__(self, num_classes: int = 2000) -> None:
        """
        Parameters
        ----------
        num_classes: int
            Number of classes for the classification
        """
        super().__init__()

        # The base model
        self.num_classes = num_classes
        init_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.finalpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc =  nn.Linear(2048, num_classes, bias=False)



    def getFeat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4 = self.layer4(x)
        feat = self.finalpool(x4).squeeze(dim=-1).squeeze(dim=-1)
        return feat
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x: torch.Tensor
            Input image

        Returns
        -------
        scores: torch.Tensor
            Classification scores per landmark
        """
        # Pretrained ResNet part of the model
        feat = self.getFeat(x)
        score = self.fc(feat)

        return score