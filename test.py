import torch
import torch.nn as nn
from functools import partial

from imagebind.models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)

def prune_conv3d_out_channels(conv3d: nn.Conv3d, prune_ratio: float = 0.5):
    out_channels = conv3d.out_channels
    num_prune = int(out_channels * prune_ratio)
    num_keep = out_channels - num_prune

    weight_scores = conv3d.weight.data.abs().mean(dim=(1,2,3,4))  # shape: out_channels
    keep_idx = torch.argsort(weight_scores, descending=True)[:num_keep]

    new_weight = conv3d.weight.data[keep_idx, :, :, :, :].clone()
    if conv3d.bias is not None:
        new_bias = conv3d.bias.data[keep_idx].clone()
    else:
        new_bias = None

    new_conv = nn.Conv3d(
        in_channels=conv3d.in_channels,
        out_channels=num_keep,
        kernel_size=conv3d.kernel_size,
        stride=conv3d.stride,
        padding=conv3d.padding,
        dilation=conv3d.dilation,
        groups=conv3d.groups,
        bias=(new_bias is not None),
        padding_mode=conv3d.padding_mode,
    )
    new_conv.weight.data = new_weight
    if new_bias is not None:
        new_conv.bias.data = new_bias

    return new_conv, keep_idx


def prune_patch_embed_generic_conv3d(patch_embed: PatchEmbedGeneric, prune_ratio: float = 0.5):
    new_proj_stem = []
    for layer in patch_embed.proj:
        if isinstance(layer, nn.Conv3d):
            new_conv, _ = prune_conv3d_out_channels(layer, prune_ratio)
            new_proj_stem.append(new_conv)
        else:
            new_proj_stem.append(layer)
    return PatchEmbedGeneric(proj_stem=new_proj_stem, norm_layer=patch_embed.norm_layer)


kernel_size = (2, 16, 16)
vision_embed_dim = 128
video_frames = 8

rgbt_stem = PatchEmbedGeneric(
    proj_stem=[
        PadIm2Video(pad_type="repeat", ntimes=2),
        nn.Conv3d(
            in_channels=3,
            kernel_size=kernel_size,
            out_channels=vision_embed_dim,
            stride=kernel_size,
            bias=False,
        ),
    ]
)

rgbt_stem_100 = rgbt_stem  
rgbt_stem_50 = prune_patch_embed_generic_conv3d(rgbt_stem, prune_ratio=0.5)  


from functools import partial

rgbt_preprocessor_100 = RGBDTPreprocessor(
    img_size=[3, video_frames, 224, 224],
    num_cls_tokens=1,
    pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
    rgbt_stem=rgbt_stem_100,
    depth_stem=None,
)

rgbt_preprocessor_50 = RGBDTPreprocessor(
    img_size=[3, video_frames, 224, 224],
    num_cls_tokens=1,
    pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
    rgbt_stem=rgbt_stem_50,
    depth_stem=None,
)
dummy_input = torch.randn(2, 3, 224, 224)
print(rgbt_preprocessor_50)
print(rgbt_preprocessor_100)