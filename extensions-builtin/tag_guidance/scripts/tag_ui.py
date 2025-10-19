import gradio as gr

from modules import scripts, shared
from TAGGuidance.patch import patch_model_with_tag

class TagGuidanceScript(scripts.Script):
    sorting_priority = 15.6

    def __init__(self):
        super().__init__()
        self.current_eta = getattr(shared.opts, "tag_guidance_eta", 1.0)

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

        return [eta_slider]

    def process_before_every_sampling(self, p, eta_value, *args, **kwargs):
        try:
            eta = float(eta_value)
        except (TypeError, ValueError):
            eta = 1.0
        self.current_eta = eta
        shared.opts.tag_guidance_eta = eta
        patched = patch_model_with_tag(p.sd_model.forge_objects.unet, eta)[0]
        p.sd_model.forge_objects.unet = patched
        p.extra_generation_params["TAG η"] = round(eta, 3)






