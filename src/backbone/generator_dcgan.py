import torch
import torch.nn as nn
from .blocks import GBlock
from src.config import latent_dim, channels

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            GBlock(latent_dim, 1024, kernel_size=4, stride=1, padding=0),
            GBlock(1024, 512),
            GBlock(512, 256),
            GBlock(256, 128),
            GBlock(128, channels, final_layer=True)  
        )

    def forward(self, x):
        return self.main(x)
