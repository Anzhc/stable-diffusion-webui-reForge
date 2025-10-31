from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def normalize_spatially(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each channel of a BCHW tensor to zero mean and unit variance.
    """
    if tensor.ndim < 3:
        return tensor

    dims = tuple(range(2, tensor.ndim))
    mean = tensor.mean(dim=dims, keepdim=True)
    var = tensor.var(dim=dims, keepdim=True, unbiased=False)
    return (tensor - mean) / torch.sqrt(var + eps)


def apply_reso_block_noise(
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: Optional[torch.Tensor],
    target_sizes_hw: Optional[torch.Tensor] = None,
    *,
    enabled: bool = False,
    base_resolution: int = 512,
    num_train_timesteps: Optional[int] = None,
    scale_quant_step: float = 0.5,
) -> torch.Tensor:
    """
    Apply block-structured noise to high-resolution samples so their low-frequency SNR better
    matches the 512 resolution bucket.
    """
    if (
        not enabled
        or noise is None
        or noise.ndim != 4
        or noise.shape[0] == 0
        or latents is None
    ):
        return noise

    device = noise.device
    work_dtype = torch.float32
    batch_size = noise.shape[0]

    if target_sizes_hw is not None:
        sizes_hw = target_sizes_hw.to(device=device, dtype=work_dtype)
    else:
        lat_h = latents.shape[2] * 8
        lat_w = latents.shape[3] * 8
        sizes_hw = torch.tensor(
            [lat_h, lat_w],
            device=device,
            dtype=work_dtype,
        ).repeat(batch_size, 1)

    heights = sizes_hw[:, 0]
    widths = sizes_hw[:, 1]
    effective_res = torch.sqrt(torch.clamp(heights * widths, min=1.0))

    step = max(scale_quant_step, 1e-6)
    raw_scale = effective_res / float(base_resolution)
    quantized_scale = torch.round(raw_scale / step) * step
    quantized_scale = torch.clamp(quantized_scale, min=step)
    extra_scale = torch.clamp(quantized_scale - 1.0, min=0.0)

    contributing = torch.nonzero(extra_scale > 1e-6, as_tuple=False).flatten()
    if contributing.numel() == 0:
        return noise

    if timesteps is None:
        timesteps_f = torch.zeros((batch_size,), device=device, dtype=work_dtype)
    else:
        timesteps_f = timesteps.to(device=device, dtype=work_dtype)

    if num_train_timesteps and num_train_timesteps > 1:
        max_t = float(num_train_timesteps - 1)
    else:
        if timesteps_f.numel() > 0:
            max_t = float(timesteps_f.max().item())
            if max_t <= 0.0:
                max_t = 1.0
        else:
            max_t = 1.0

    t_norm = 1.0 - torch.clamp(timesteps_f / max_t, min=0.0, max=1.0)
    alpha_floor = 0.05
    alpha_gain = 0.65
    alpha = extra_scale * (alpha_gain * t_norm + alpha_floor)
    alpha = torch.clamp(alpha, max=2.0)

    noise_seed = torch.randn_like(noise, dtype=work_dtype)
    block_noise = torch.zeros_like(noise, dtype=work_dtype)

    for idx in contributing.tolist():
        scale = float(quantized_scale[idx].item())
        if scale <= 1.0:
            continue
        sample = noise_seed[idx : idx + 1]
        _, _, h, w = sample.shape
        inv_scale = 1.0 / scale
        down_h = max(1, int(round(h * inv_scale)))
        down_w = max(1, int(round(w * inv_scale)))
        downsampled = F.interpolate(sample, size=(down_h, down_w), mode="area")
        block = F.interpolate(downsampled, size=(h, w), mode="nearest")
        block = normalize_spatially(block)
        block_noise[idx] = block[0]

    alpha = alpha.to(device=device, dtype=work_dtype).view(-1, 1, 1, 1)
    base = noise.to(work_dtype)
    denom = torch.sqrt(1.0 + alpha.pow(2))
    mixed = (base + alpha * block_noise) / denom

    return mixed.to(noise.dtype)
