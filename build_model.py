# File :     build_model.py
# Author :   Chuxuan,Changjin
# Project :  SimDMMCI
import time
import statistics
import torch
import torch.nn as nn
from functools import partial
from functools import reduce
import operator
import torchprofile

from imagebind import ModalityType
from imagebind.models.multimodal_preprocessors import (
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    AudioPreprocessor,
    TextPreprocessor,
    ThermalPreprocessor,
    IMUPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    PadIm2Video
)
device = "cpu"




def prune_conv2d_out_channels(conv2d: nn.Conv2d, prune_ratio=0.5):
    out_channels = conv2d.out_channels
    num_prune = int(out_channels * prune_ratio)
    num_keep = out_channels - num_prune

    weight_scores = conv2d.weight.data.abs().mean(dim=(1,2,3))
    keep_idx = torch.argsort(weight_scores, descending=True)[:num_keep]

    new_weight = conv2d.weight.data[keep_idx, :, :, :].clone()
    if conv2d.bias is not None:
        new_bias = conv2d.bias.data[keep_idx].clone()
    else:
        new_bias = None

    new_conv = nn.Conv2d(
        in_channels=conv2d.in_channels,
        out_channels=num_keep,
        kernel_size=conv2d.kernel_size,
        stride=conv2d.stride,
        padding=conv2d.padding,
        dilation=conv2d.dilation,
        groups=conv2d.groups,
        bias=(new_bias is not None),
        padding_mode=conv2d.padding_mode,
    )
    new_conv.weight.data = new_weight
    if new_bias is not None:
        new_conv.bias.data = new_bias

    return new_conv, keep_idx

def prune_linear_out_features(linear: nn.Linear, prune_ratio=0.5):
    out_features = linear.out_features
    num_prune = int(out_features * prune_ratio)
    num_keep = out_features - num_prune

    weight_scores = linear.weight.data.abs().mean(dim=1)  # 按行求均值
    keep_idx = torch.argsort(weight_scores, descending=True)[:num_keep]

    new_weight = linear.weight.data[keep_idx, :].clone()
    if linear.bias is not None:
        new_bias = linear.bias.data[keep_idx].clone()
    else:
        new_bias = None

    new_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=num_keep,
        bias=(new_bias is not None),
    )
    new_linear.weight.data = new_weight
    if new_bias is not None:
        new_linear.bias.data = new_bias

    return new_linear, keep_idx

def prune_patch_embed_generic_conv2d_linear(patch_embed: PatchEmbedGeneric, prune_ratio=0.5):
    new_proj_stem = []

    if isinstance(patch_embed.proj, nn.Sequential):
        for layer in patch_embed.proj:
            if isinstance(layer, nn.Conv2d):
                new_layer, _ = prune_conv2d_out_channels(layer, prune_ratio)
                new_proj_stem.append(new_layer)
            elif isinstance(layer, nn.Linear):
                new_layer, _ = prune_linear_out_features(layer, prune_ratio)
                new_proj_stem.append(new_layer)
            else:
                new_proj_stem.append(layer)
    else:
        layer = patch_embed.proj
        if isinstance(layer, nn.Conv2d):
            new_layer, _ = prune_conv2d_out_channels(layer, prune_ratio)
            new_proj_stem.append(new_layer)
        elif isinstance(layer, nn.Linear):
            new_layer, _ = prune_linear_out_features(layer, prune_ratio)
            new_proj_stem.append(new_layer)
        else:
            new_proj_stem.append(layer)

    
    if patch_embed.norm_layer is not None and hasattr(patch_embed.norm_layer, 'normalized_shape'):
        old_norm_shape = patch_embed.norm_layer.normalized_shape
        if isinstance(old_norm_shape, tuple):
            new_norm_shape = (int(old_norm_shape[0] * (1 - prune_ratio)),) + old_norm_shape[1:]
        else:
            new_norm_shape = int(old_norm_shape * (1 - prune_ratio))
        new_norm_layer = torch.nn.LayerNorm(normalized_shape=new_norm_shape)
    else:
        new_norm_layer = patch_embed.norm_layer

    return PatchEmbedGeneric(proj_stem=new_proj_stem, norm_layer=new_norm_layer)



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
def build_preprocessors(init_param_style="openclip"):
    video_frames = 2

    # Vision Preprocessor
    rgbt_stem = PatchEmbedGeneric([
        PadIm2Video(pad_type="repeat", ntimes=2),
        torch.nn.Conv3d(
            in_channels=3,
            out_channels=1024,
            kernel_size=(2, 14, 14),
            stride=(2, 14, 14),
            bias=False
        )
    ])
   
    rgbt_stem_100 = rgbt_stem  
    rgbt_stem_50 = prune_patch_embed_generic_conv3d(rgbt_stem, prune_ratio=0.5)  
    rgbt_preprocessor_100 = RGBDTPreprocessor(
        img_size=[3, video_frames, 224, 224],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        rgbt_stem=rgbt_stem_100,
        depth_stem=None,
        init_param_style=init_param_style
    )
    rgbt_preprocessor_50 = RGBDTPreprocessor(
        img_size=[3, video_frames, 224, 224],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        rgbt_stem=rgbt_stem_50,
        depth_stem=None,
        init_param_style=init_param_style
    )

    
    pruned_embed_dim = 384  
    # Text Preprocessor
    text_preprocessor_100 = TextPreprocessor(
        context_length=77,
        vocab_size=49408,
        embed_dim=768,
        causal_masking=True,
        init_param_style=init_param_style
    )

    text_preprocessor_50 = TextPreprocessor(
        context_length=77,
        vocab_size=49408,
        embed_dim=pruned_embed_dim,
        causal_masking=True,
        init_param_style=init_param_style
    )

    # Audio Preprocessor
    audio_stem = PatchEmbedGeneric(
        [
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=768,
                kernel_size=16,
                stride=10,
                bias=False
            )
        ],
        norm_layer=torch.nn.LayerNorm(normalized_shape=768)
    )
   
    audio_stem_100 = audio_stem
    audio_stem_50 = prune_patch_embed_generic_conv2d_linear(audio_stem, prune_ratio=0.5)
    audio_preprocessor_100 = AudioPreprocessor(
        img_size=[1, 128, 204],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        audio_stem=audio_stem_100,
        init_param_style=init_param_style
    )
    audio_preprocessor_50 = AudioPreprocessor(
        img_size=[1, 128, 204],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        audio_stem=audio_stem_50,
        init_param_style=init_param_style
    )

    # Depth Preprocessor
    depth_stem = PatchEmbedGeneric([
        torch.nn.Conv2d(
            in_channels=1,
            out_channels=384,
            kernel_size=16,
            stride=16,
            bias=False
        )
    ], norm_layer=torch.nn.LayerNorm(normalized_shape=384))
    
    depth_stem_100 = depth_stem
    depth_stem_50 = prune_patch_embed_generic_conv2d_linear(depth_stem, prune_ratio=0.5)
    depth_preprocessor_100 = RGBDTPreprocessor(
        img_size=[1, 224, 224],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        rgbt_stem=depth_stem_100,
        depth_stem=depth_stem_100,
        init_param_style=init_param_style
    )
    depth_preprocessor_50 = RGBDTPreprocessor(
        img_size=[1, 224, 224],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        rgbt_stem=depth_stem_50,
        depth_stem=depth_stem_50,
        init_param_style=init_param_style
    )

    # Thermal Preprocessor
    thermal_stem = PatchEmbedGeneric([
        torch.nn.Conv2d(
            in_channels=1,
            out_channels=768,
            kernel_size=16,
            stride=16,
            bias=False
        )
    ], norm_layer=torch.nn.LayerNorm(normalized_shape=768))
    
    thermal_stem_100 = thermal_stem
    thermal_stem_50 = prune_patch_embed_generic_conv2d_linear(thermal_stem, prune_ratio=0.5)
    thermal_preprocessor_100 = ThermalPreprocessor(
        img_size=[1, 224, 224],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        thermal_stem=thermal_stem_100,
        init_param_style=init_param_style
    )
    thermal_preprocessor_50 = ThermalPreprocessor(
        img_size=[1, 224, 224],
        num_cls_tokens=1,
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        thermal_stem=thermal_stem_50,
        init_param_style=init_param_style
    )

    # IMU Preprocessor
    imu_stem = PatchEmbedGeneric([
        torch.nn.Linear(in_features=48, out_features=512, bias=False)
    ], norm_layer=torch.nn.LayerNorm(normalized_shape=512))
    
    imu_stem_100 = imu_stem
    imu_stem_50 = prune_patch_embed_generic_conv2d_linear(imu_stem, prune_ratio=0.5)
    imu_preprocessor_100 = IMUPreprocessor(
        img_size=[6, 2000],
        num_cls_tokens=1,
        kernel_size=8,
        embed_dim=512,  
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        imu_stem=imu_stem_100,
        init_param_style=init_param_style
    )
    imu_preprocessor_50 = IMUPreprocessor(
        img_size=[6, 2000],
        num_cls_tokens=1,
        kernel_size=8,
        embed_dim=256,  
        pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        imu_stem=imu_stem_50,
        init_param_style=init_param_style
    )

    return {
        "vision_100": rgbt_preprocessor_100,
        "vision_50": rgbt_preprocessor_50,
        "text_100": text_preprocessor_100,
        "text_50": text_preprocessor_50,
        "audio_100": audio_preprocessor_100,
        "audio_50": audio_preprocessor_50,
        "depth_100": depth_preprocessor_100,
        "depth_50": depth_preprocessor_50,
        "thermal_100": thermal_preprocessor_100,
        "thermal_50": thermal_preprocessor_50,
        "imu_100": imu_preprocessor_100,
        "imu_50": imu_preprocessor_50,
    }


from fvcore.nn import FlopCountAnalysis
import torch
import argparse

def print_tensor_shapes(d, prefix=""):
    if isinstance(d, dict):
        for key, val in d.items():
            print_tensor_shapes(val, prefix + key + ".")
    elif isinstance(d, torch.Tensor):
        print(f"{prefix[:-1]} shape: {d.shape}, dtype: {d.dtype}")

def test_model(model, dummy_input, num_runs, k):
    all_times = []
    ts = time.time()
    for _ in range(num_runs):
        starttime = time.time()
        output, logs = model(dummy_input, return_logs=True)
        run_times = [t - starttime for t, shape in logs]
        run_shapes = [shape for t, shape in logs]

        all_times.append(run_times)
        all_shapes = run_shapes

    all_times_T = list(zip(*all_times))
    avg_times = [sum(times) / len(times) for times in all_times_T]
    # Effective on the server
    total_time = (time.time() - ts) / num_runs
    print(f"total time: {total_time:.6f}s")

    print(f">>> {k}-avg time (total {num_runs} ):")
    for i, (avg, shape) in enumerate(zip(avg_times, all_shapes)):
        data = reduce(operator.mul, shape, 1)
        total_bits = data * 32
        print(f"  - Stage {i + 1}: {avg:.6f}s  shape: {shape}  total bits: {total_bits}  left time: {(total_time - avg):.6f}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handling the n parameter")

    parser.add_argument("n", type=int, nargs="?", default=1, help="Number of samples to input (default: 1)")
    args = parser.parse_args()
    print(device)
    N = args.n
    num_runs = 10

    openclip_preprocessors = build_preprocessors("openclip")
    for k, model in openclip_preprocessors.items():
        if 'text' in k.lower():
            # print(model)
            dummy_input = torch.randint(0, model.vocab_size, (2, model.context_length))
            dummy_input = dummy_input.repeat(N, 1)
            test_model(model, dummy_input, num_runs, k)

        if 'vision' in k.lower():
            dummy_input = torch.randn(2, 3, 224, 224)
            dummy_input = dummy_input.repeat(N, 1, 1, 1)  # 2 × 8 = 16 样本
            test_model(model, dummy_input, num_runs, k)

        elif 'audio' in k.lower():
            dummy_input = torch.randn(2, 1, 128, 204)
            dummy_input = dummy_input.repeat(N, 1, 1, 1)
            test_model(model, dummy_input, num_runs, k)

        elif 'thermal' in k.lower():
            dummy_input = torch.randn(2, 1, 224, 224)
            dummy_input = dummy_input.repeat(N, 1, 1, 1)
            test_model(model, dummy_input, num_runs, k)
        #
        elif 'depth' in k.lower():
            dummy_input = torch.randn(2, 1, 224, 224)
            dummy_input = dummy_input.repeat(N, 1, 1, 1)
            test_model(model, dummy_input, num_runs, k)

        elif 'imu' in k.lower():
            dummy_input = torch.randn(2, 6, 2000)
            dummy_input = dummy_input.repeat(N, 1, 1)
            test_model(model, dummy_input, num_runs, k)

        # print_tensor_shapes(output)




