
export PYTHONPATH=$PWD:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29508 infer_helios_batch.py \
    --enable_parallelism \
    --cp_backend "ulysses" \
    --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --transformer_path "ablation_stage_1_post/merge/step_8000" \
    --sample_type "i2v" \
    --num_frames 165 \
    --fps 24 \
    --task_dir "eval/ours/task_design" \
    --guidance_scale 5.0 \
    --enable_compile \
    --output_folder "eval/ours/output_8000"
    
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

# CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
#     --base_model_path "BestWishYsh/Helios-Base" \
#     --transformer_path "BestWishYsh/Helios-Base" \
#     --sample_type "i2v" \
#     --num_frames 99 \
#     --fps 24 \
#     --image_path "example/wave.jpg" \
#     --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and respect for nature’s might." \
#     --guidance_scale 5.0 \
#     --enable_compile \
#     --output_folder "./output_helios/helios-base"
