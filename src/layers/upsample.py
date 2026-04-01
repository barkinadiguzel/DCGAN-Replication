import torch.nn as nn

def upsample_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
    from src.backbone.blocks import GBlock
    return GBlock(in_channels, out_channels, kernel_size, stride, padding, final_layer)
