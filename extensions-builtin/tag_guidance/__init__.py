import gradio as gr

from modules import script_callbacks, shared

TAG_OPTION_KEY = "tag_guidance_eta"


def _register_options():
    if TAG_OPTION_KEY not in shared.opts.data_labels:
        shared.opts.add_option(
            TAG_OPTION_KEY,
            shared.OptionInfo(
                1.0,
                "Tangential Amplifying Guidance η",
                gr.Slider,
                {"minimum": 1.0, "maximum": 1.5, "step": 0.01},
                section=("sampler", "Sampling"),
            ).info("η=1 disables TAG (Tangential Amplifying Guidance, arXiv:2510.04533)"),
        )

    quicksettings = getattr(shared.opts, "quicksettings_list", None)
    if isinstance(quicksettings, list) and TAG_OPTION_KEY not in quicksettings:
        quicksettings.append(TAG_OPTION_KEY)


script_callbacks.on_ui_settings(_register_options, name="TAG Guidance Options")
