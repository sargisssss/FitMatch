import torch
from models.embedding_model import EmbeddingModel

model = EmbeddingModel(embedding_dim=128)

dummy_input = torch.randn(4, 3, 224, 224)

output = model(dummy_input)

print("Output shape:", output.shape)