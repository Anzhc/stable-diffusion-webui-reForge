from ldm_patched.modules.model_patcher import ModelPatcher


class TagModelPatch:
    def __init__(self, eta: float):
        self.eta = eta

    def to(self, device=None, dtype=None):
        return self

    def __call__(self, model_inputs, model_kwargs):
        model_kwargs.setdefault("model_options", {})
        model_options = model_kwargs["model_options"]
        tag_opts = model_options.get("tag_guidance", {})
        tag_opts["solver"] = {"enabled": True, "tangential_scale": float(self.eta)}
        model_options["tag_guidance"] = tag_opts
        model_kwargs["model_options"] = model_options
        return model_inputs, model_kwargs
