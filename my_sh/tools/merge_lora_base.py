import sys
from argparse import Namespace


import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components

transformer_additional_kwargs = {
    "has_multi_term_memory_patch": True,
    "zero_history_timestep": True,
    "guidance_cross_attn": True,
    "restrict_self_attn": False,
    "is_train_restrict_lora": False,
    "restrict_lora": False,
    "restrict_lora_rank": 128,
    "is_amplify_history": False,
    "history_scale_mode": "per_head",
}
transformer = HeliosTransformer3DModel.from_pretrained(
    "/beijing-c/models/BestWishYSH/Helios-Base",
    subfolder="transformer_init",
    transformer_additional_kwargs=transformer_additional_kwargs,
)
pipe = HeliosPipeline.from_pretrained(
    "/beijing-c/models/BestWishYSH/Helios-Base",
    transformer=transformer,
)

pipe.load_lora_weights(
    "ablation_stage_1_spatialvid/checkpoint-7000/pytorch_lora_weights.safetensors",
    adapter_name="default",
)
pipe.set_adapters(["default"], adapter_weights=[1.0])


args = Namespace()
if not hasattr(args, "training_config"):
    args.training_config = Namespace()
args.training_config.is_enable_stage1 = True
args.training_config.restrict_self_attn = False
args.training_config.is_amplify_history = False
args.training_config.is_use_gan = False
load_extra_components(
    args,
    transformer,
    "ablation_stage_1_spatialvid/checkpoint-7000/transformer_partial.pth",
)

pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.transformer.save_pretrained(
    "ablation_stage_1_spatialvid/merge/step_7000/transformer"
)
