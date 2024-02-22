import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, D:int, H:int, dropout:float ) -> None:
        super().__init__()
        self.D = D
        self.H = H
        self.dropout = dropout
        assert self.D % self.H == 0, f"features {D} not divisible by heads {self.H}"
        self.d_k = D // H
        self.query = nn.Linear(self.D,self.D)
        self.key = nn.Linear(self.D,self.D)
        self.value = nn.Linear(self.D,self.D)
        self.output = nn.Linear(self.D, self.D)
        self.dropout_layer = nn.Dropout(self.dropout)
        # self.output = nn.Sequential(
        #     nn.Linear(self.D,self.D, bias=False),
        #     nn.Dropout(self.dropout))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch, N, _ = x.size() #[2, 145, 108]
        q = self.query(x) #[2, 145, 108]
        q = q.view(batch, self.H, N, self.d_k) #[2, 6, 145, 18]
        k = self.key(x)
        k = k.view(batch, self.H, N, self.d_k)
        v = self.value(x)
        v = v.view(batch, self.H, N, self.d_k)
        dots = (q @ k.transpose(2,3)) / (self.d_k ** 0.5) #[2, 6, 145, 145]
        attn = F.softmax(dots, dim=3) #[2, 6, 145, 145]
        out = attn @ v #[2, 6, 145, 18]
        out = out.transpose(1, 2).reshape(batch, N, self.D) #[2, 145, 108]
        out = self.output(out)
        return self.dropout_layer(out) #[2, 145, 108]

