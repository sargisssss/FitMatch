import torch
import torch.nn as nn
import torchvision.models as models


class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingModel, self).__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)

        embedding = nn.functional.normalize(embedding, p=2, dim=1)

        return embedding