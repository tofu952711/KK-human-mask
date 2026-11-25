"""Sapiens2 0.4B segmentation head and unmodified Meta standalone backbone.

Head layout adapted from starsFriday/ComfyUI-Sapiens2 (sapiens2/modeling.py).
Sapiens2 materials are governed by licenses/Sapiens2-LICENSE.md.
"""
from torch import nn

from .backbone import Sapiens2


class SegHead(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1024
        for out_channels in (512, 256, 128, 64):
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ])
            in_channels = out_channels
        self.deconv_layers = nn.Sequential(*layers)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.InstanceNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 1), nn.InstanceNorm2d(64), nn.SiLU(inplace=True),
        )
        self.conv_seg = nn.Conv2d(64, 29, 1)

    def forward(self, features):
        return self.conv_seg(self.conv_layers(self.deconv_layers(features)))


class Sapiens2PersonSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Sapiens2(
            arch='sapiens2_0.4b', img_size=(1024, 768), patch_size=16,
            final_norm=True, use_tokenizer=False, with_cls_token=True, out_type='featmap',
        )
        self.decode_head = SegHead()

    def forward(self, image):
        return self.decode_head(self.backbone(image)[0])
