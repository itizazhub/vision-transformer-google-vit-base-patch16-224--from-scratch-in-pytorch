import torch
import torch.nn as nn
from vit_model.encoder import Encoder
from vit_model.embeddings import Embeddings

class Vit(nn.Module):
    def __init__(self, classes:int, blocks:int, channels:int, height: int, width:int, patch_size:int, H:int, inner_dim:int, dropout:float) -> None:
        super().__init__()
        self.embeddings = Embeddings(channels, height, width, patch_size)
        self.encoder = nn.Sequential(
            *[Encoder(self.embeddings.D, H, inner_dim, dropout) for _ in range(blocks)]
        )
        self.pooler = nn.Linear(self.embeddings.D, self.embeddings.D)
        self.layernorm = nn.LayerNorm(normalized_shape=self.embeddings.D)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embeddings.D),
            nn.Linear(self.embeddings.D, classes)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(x)
        encoder = self.encoder(embeddings)
        layernorm = self.layernorm(encoder)
        pooler = self.pooler(layernorm)
        cls_token = pooler[: , 0]
        mlp_head = self.mlp_head(cls_token)
        return mlp_head

