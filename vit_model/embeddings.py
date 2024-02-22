import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, channels:int, height: int, width:int, patch_size:int) -> None:
        super().__init__()
        assert height % patch_size == 0, "height is not divible by patch_size"
        self.N = (height * width) // (patch_size ** 2)
        self.D = (patch_size ** 2) * channels
        self.projection = nn.Conv2d(channels, self.D, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.D))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.N + 1, self.D))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x->[2, 3, 72, 72]
        out = self.projection(x) #[2, 108, 12, 12]
        out = out.flatten(2) #[2, 108, 144]
        out = out.transpose(1,2) #[2, 144, 108]
        #repeat() function is used to add cls_tonken to all images in the batch
        out = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), out], dim=1) #[2, 145, 108]
        out = out + self.position_embeddings #[2, 145, 108]
        return out #[2, 145, 108]

