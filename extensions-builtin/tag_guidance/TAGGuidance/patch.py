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
    adjusted = x_float - derivative_tag * sigma_float
    return adjusted.to(dtype=denoised.dtype)


def _apply_ctag(x, sigma, cond, uncond, cond_scale, ctag_eta):
    if ctag_eta is None or ctag_eta <= 0.0:
        return None
    if cond is None or uncond is None:
        return None
    if cond_scale is None:
        return None
    if not isinstance(cond_scale, (int, float)):
        try:
            cond_scale = float(cond_scale)
        except (TypeError, ValueError):
            return None
    sigma_tensor = _prepare_sigma(sigma, x)
    sigma_view = append_dims(sigma_tensor, x.ndim)
    if torch.allclose(sigma_view, torch.zeros_like(sigma_view)):
        return None
    x_float = x.float()
    sigma_float = sigma_view.float()
    cond_eps = (x_float - cond.float()) / sigma_float
    uncond_eps = (x_float - uncond.float()) / sigma_float
    g = cond_eps - uncond_eps
    unit = _compute_unit(x)
    _, g_tan = _split_radial_tangential(g, unit.float())
    dims = tuple(range(1, g_tan.ndim))
    denom = (g_tan ** 2).sum(dim=dims, keepdim=True)
    eps = torch.finfo(denom.dtype).eps
    alpha = (cond_eps * g_tan).sum(dim=dims, keepdim=True) / (denom + eps)
    extra = alpha * g_tan
    guided_eps = uncond_eps + cond_scale * g + ctag_eta * extra
    guided = x_float - guided_eps * sigma_float
    return guided.to(dtype=cond.dtype)


def patch_model_with_tag(model: ModelPatcher, eta: float, ctag_eta: float = 0.0):
    print(f"[TAG] patch_model_with_tag eta={eta} (C-TAG eta={ctag_eta})")
    patched = model.clone()

    def tag_post_cfg(args):
        x = args.get("input")
        denoised = args.get("denoised")
        sigma = args.get("sigma")
        if x is None or denoised is None or sigma is None:
            return args.get("denoised")

        if ctag_eta > 0.0:
            cond = args.get("cond_denoised")
            uncond = args.get("uncond_denoised")
            cond_scale = args.get("cond_scale")
            ctag_denoised = _apply_ctag(x, sigma, cond, uncond, cond_scale, ctag_eta)
            if ctag_denoised is not None:
                denoised = ctag_denoised

        return _apply_tag(x, denoised, sigma, eta)

    patched.set_model_sampler_post_cfg_function(tag_post_cfg, disable_cfg1_optimization=True)
    return (patched,)

