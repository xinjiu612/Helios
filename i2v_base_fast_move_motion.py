import argparse
import os
import time

import torch
from diffusers import HeliosPipeline
from diffusers.models import HeliosTransformer3DModel
from diffusers.utils import export_to_video, load_image


def build_prompt() -> str:
    return (
        "Cinematic image-to-video shot in a modern office environment with a tree initially far on the left side of the frame. "
        "0-3 seconds: execute a strong accelerating forward camera push toward the tree, with clear parallax and directional "
        "motion blur on curtains, desks, equipment racks, and ceiling rails. "
        "At exactly 3 seconds: perform a crisp hard stop around one meter in front of the tree. "
        "3-5 seconds: hold a stable lock-off shot with no residual drift or shake while preserving detailed texture on foliage, "
        "trunk, office background, right-side drapes, and floor cables."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helios-Base I2V motion script.")
    parser.add_argument("--base_model_path", type=str, default="/beijing-c/models/BestWishYSH/Helios-Base")
    parser.add_argument("--transformer_path", type=str, default="/beijing-c/models/BestWishYSH/Helios-Base")
    parser.add_argument(
        "--image_path",
        type=str,
        default="/beijing-c/workspace/zhumo/diff/task_design/language_control/fast_move.png",
    )
    parser.add_argument("--output_folder", type=str, default="./output_helios/helios-zhumo")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--num_frames", type=int, default=132)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=60)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--is_skip_first_chunk", action="store_true", default=True)
    parser.add_argument("--image_noise_sigma_min", type=float, default=0.20)
    parser.add_argument("--image_noise_sigma_max", type=float, default=0.30)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    os.makedirs(args.output_folder, exist_ok=True)

    if args.num_frames % 33 != 0:
        args.num_frames = ((args.num_frames + 32) // 33) * 33

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt.strip()
    else:
        prompt = build_prompt()
    negative_prompt = (
        "random jitter, violent shake, deformation, warping, flicker, frame tearing, low quality, overexposed, "
        "underexposed, unstable geometry, object morphing"
    )

    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.to(device)

    image = load_image(args.image_path).resize((args.width, args.height))

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            add_noise_to_image_latents=True,
            image_noise_sigma_min=args.image_noise_sigma_min,
            image_noise_sigma_max=args.image_noise_sigma_max,
            is_skip_first_chunk=args.is_skip_first_chunk,
        ).frames[0]

    ts = int(time.time())
    out_video = os.path.join(args.output_folder, f"i2v_base_motion_{args.tag}_{ts}.mp4")
    out_prompt = os.path.join(args.output_folder, f"i2v_base_motion_{args.tag}_{ts}.txt")
    export_to_video(output, out_video, fps=args.fps)
    with open(out_prompt, "w", encoding="utf-8") as f:
        f.write(prompt + "\n")

    print(f"Saved video: {out_video}")
    print(f"Saved prompt: {out_prompt}")


if __name__ == "__main__":
    main()
