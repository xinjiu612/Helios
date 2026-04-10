import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_video, load_image

vae = AutoModel.from_pretrained("/beijing-c/models/BestWishYSH/Helios-Distilled", subfolder="vae", torch_dtype=torch.float32)

pipeline = HeliosPyramidPipeline.from_pretrained(
    "/beijing-c/models/BestWishYSH/Helios-Distilled",
    vae=vae,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

# --- T2V ---
prompt = """
A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue 
and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with 
a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, 
allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades 
of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and 
the vivid colors of its surroundings. A close-up shot with dynamic movement.
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=240,
    pyramid_num_inference_steps_list=[2, 2, 2],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_distilled_t2v_output.mp4", fps=24)

# --- I2V ---
i2v_prompt = """
A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, 
illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, 
casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes 
apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and 
relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and 
respect for nature’s might.
"""
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg"

output = pipeline(
    prompt=i2v_prompt,
    negative_prompt=negative_prompt,
    image=load_image(image_path).resize((640, 384)),
    num_frames=240,
    pyramid_num_inference_steps_list=[2, 2, 2],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_distilled_i2v_output.mp4", fps=24)

# --- V2V ---
v2v_prompt = """
A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees 
under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, 
emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to 
the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. 
A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery.
"""
video_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4"

output = pipeline(
    prompt=v2v_prompt,
    negative_prompt=negative_prompt,
    video=load_video(video_path),
    num_frames=240,
    pyramid_num_inference_steps_list=[2, 2, 2],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_distilled_v2v_output.mp4", fps=24)