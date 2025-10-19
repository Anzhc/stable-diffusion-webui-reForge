import gradio as gr
import logging
import traceback
from functools import partial

from modules import scripts, shared, script_callbacks
from TAGGuidance.patch import patch_model_with_tag

class TagGuidanceScript(scripts.Script):
    sorting_priority = 15.6

    def __init__(self):
        super().__init__()
        self.current_eta = getattr(shared.opts, "tag_guidance_eta", 1.0)
        self.ctag_enabled = False
        self.ctag_eta = 0.0

    def title(self):
        return "Tangential Amplifying Guidance (TAG)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        default_eta = getattr(shared.opts, "tag_guidance_eta", 1.0)
        with gr.Accordion(self.title(), open=False):
            eta_slider = gr.Slider(
                label="TAG Tangential Scale η",
                minimum=1.0,
                maximum=1.5,
                step=0.01,
                value=default_eta,
            )
            ctag_eta_slider = gr.Slider(
                label="C-TAG Gain",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=self.ctag_eta,
            )
            ctag_checkbox = gr.Checkbox(
                label="Enable Conditional TAG (C-TAG)",
                value=self.ctag_enabled,
            )

        return [eta_slider, ctag_eta_slider, ctag_checkbox]

    def process_before_every_sampling(self, p, eta_value, ctag_eta_value, ctag_enabled, *args, **kwargs):
        try:
            eta = float(eta_value)
        except (TypeError, ValueError):
            eta = 1.0
        self.current_eta = eta
        shared.opts.tag_guidance_eta = eta
        xyz = getattr(p, "_tag_xyz", {})
        if "tag_eta" in xyz:
            try:
                eta = float(xyz["tag_eta"])
            except (TypeError, ValueError):
                eta = 1.0
            shared.opts.tag_guidance_eta = eta
        try:
            ctag_eta = float(ctag_eta_value)
        except (TypeError, ValueError):
            ctag_eta = 0.0
        self.ctag_eta = ctag_eta
        if isinstance(ctag_enabled, str):
            ctag_flag = ctag_enabled.lower() == "true"
        else:
            ctag_flag = bool(ctag_enabled)
        if "tag_ctag" in xyz:
            ctag_flag = xyz["tag_ctag"] == "True"
        if "tag_ctag_eta" in xyz:
            try:
                ctag_eta = float(xyz["tag_ctag_eta"])
            except (TypeError, ValueError):
                ctag_eta = 0.0
        self.ctag_enabled = ctag_flag
        self.ctag_eta = ctag_eta
        ctag_eta_final = ctag_eta if ctag_flag and ctag_eta > 0.0 else 0.0
        patched = patch_model_with_tag(p.sd_model.forge_objects.unet, eta, ctag_eta_final)[0]
        p.sd_model.forge_objects.unet = patched
        p.extra_generation_params["TAG η"] = round(eta, 3)
        p.extra_generation_params["TAG C-TAG"] = ctag_flag
        if ctag_eta_final > 0.0:
            p.extra_generation_params["TAG C-TAG η"] = round(ctag_eta_final, 3)







def set_value(p, x, xs, *, field: str):
    if not hasattr(p, "_tag_xyz"):
        p._tag_xyz = {}
    p._tag_xyz[field] = x

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    axis = [
        xyz_grid.AxisOption(
            "(TAG) Tangential Scale η",
            float,
            partial(set_value, field="tag_eta"),
        ),
        xyz_grid.AxisOption(
            "(TAG) C-TAG Enabled",
            str,
            partial(set_value, field="tag_ctag"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(TAG) C-TAG η",
            float,
            partial(set_value, field="tag_ctag_eta"),
        ),
    ]
    if not any(opt.label.startswith("(TAG)") for opt in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        logging.error(f"[TAG] xyz_grid setup failed:\n{error}")

script_callbacks.on_before_ui(on_before_ui)

