import torch
import torch.nn as nn
from vit_model.multi_head_attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, D:int, H:int, inner_dim:int, dropout:float) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(normalized_shape=D)
        self.msa = MultiHeadAttention(D, H, dropout)
        self.layernorm_after = nn.LayerNorm(normalized_shape=D)
        self.intermediate = nn.Linear(D, inner_dim)
        self.gelu = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(inner_dim, D)
        # self.mlp = nn.Sequential(
        #     nn.Linear(D, inner_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(inner_dim, D),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layernorm_before = self.layernorm_before(x)
        msa = self.msa(layernorm_before)
        residual_1 = msa + x
        layernorm_after = self.layernorm_after(residual_1)
        intermediate = self.intermediate(layernorm_after)
        gelu = self.gelu(intermediate)
        output = self.output(gelu)
        residual_2 = output + residual_1

        # msa_out = self.msa(self.layernorm_before(x)) + x
        # norm_out = self.layernorm_after(msa_out)
        # intermediate = self.dropout_layer(self.gelu(self.intermediate(norm_out)))
        # out = self.output(intermediate)
        return residual_2

