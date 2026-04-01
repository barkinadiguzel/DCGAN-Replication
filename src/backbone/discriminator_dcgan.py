import torch
import torch.nn as nn
from .blocks import DBlock
from src.config import channels

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            DBlock(channels, 128, first_layer=True),
            DBlock(128, 256),
            DBlock(256, 512),
            DBlock(512, 1024),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)
