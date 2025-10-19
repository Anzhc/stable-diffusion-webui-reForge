import torch

from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.k_diffusion.sampling import append_dims


def _compute_unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = v.reshape(v.shape[0], -1)
    flat_float = flat.float()
    norms = torch.linalg.vector_norm(flat_float, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=eps)
    unit_flat = flat_float / norms
    return unit_flat.view_as(v).to(v.dtype)


def _split_radial_tangential(delta: torch.Tensor, unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, delta.ndim))
    radial_coeff = (delta * unit).sum(dim=dims, keepdim=True)
    radial = radial_coeff * unit
    tangential = delta - radial
    print(f"[TAG] radial norm {radial.norm().item():.6f}, tangential norm {tangential.norm().item():.6f}")
    return radial, tangential


def _prepare_sigma(sigma, x: torch.Tensor) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)

    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.repeat(x.shape[0])
    elif sigma_tensor.ndim == 1:
        if sigma_tensor.shape[0] == 1:
            sigma_tensor = sigma_tensor.repeat(x.shape[0])
        elif sigma_tensor.shape[0] != x.shape[0]:
            sigma_tensor = sigma_tensor.expand(x.shape[0])
    else:
        sigma_tensor = sigma_tensor.view(x.shape[0], *sigma_tensor.shape[1:])

    return sigma_tensor


def _apply_tag(x: torch.Tensor, denoised: torch.Tensor, sigma, eta: float) -> torch.Tensor:
    if eta is None or eta <= 1.0 + 1e-6:
        return denoised

    print(f"[TAG] applying eta {eta}")
    sigma_tensor = _prepare_sigma(sigma, x)
    sigma_view = append_dims(sigma_tensor, x.ndim)

    if torch.allclose(sigma_view, torch.zeros_like(sigma_view)):
        return denoised

    x_float = x.float()
    denoised_float = denoised.float()
    sigma_float = sigma_view.float()

    derivative = (x_float - denoised_float) / sigma_float
    unit = _compute_unit(x)
    radial, tangential = _split_radial_tangential(derivative, unit.float())
    derivative_tag = radial + eta * tangential
    print(f"[TAG] derivative radial norm {radial.norm().item():.6f}, tangential norm {tangential.norm().item():.6f}")

    adjusted = x_float - derivative_tag * sigma_float
    return adjusted.to(dtype=denoised.dtype)


def patch_model_with_tag(model: ModelPatcher, eta: float):
    print(f"[TAG] patch_model_with_tag eta={eta}")
    patched = model.clone()

    def tag_post_cfg(args):
        x = args.get("input")
        denoised = args.get("denoised")
        sigma = args.get("sigma")
        if x is None or denoised is None or sigma is None:
            return args.get("denoised")
        return _apply_tag(x, denoised, sigma, eta)

    patched.set_model_sampler_post_cfg_function(tag_post_cfg, disable_cfg1_optimization=True)
    return (patched,)
