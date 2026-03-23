# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29508 infer_helios_batch.py \
    --enable_parallelism \
    --cp_backend "ulysses" \
    --base_model_path "/beijing-c/models/BestWishYSH/Helios-Mid" \
    --transformer_path "ablation_stage_2_post/merge/step_8000" \
    --sample_type "i2v" \
    --num_frames 99 \
    --fps 24 \
    --task_dir eval/intergs/samples/captions_frames \
    --image_noise_sigma_min 0.111 \
    --image_noise_sigma_max 0.135 \
    --guidance_scale 5.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 20 20 20 \
    --use_zero_init \
    --zero_steps 1 \
    --output_folder "./eval/intergs/samples/mid_output_8000"
    # --enable_compile \


    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group
    # --pyramid_num_inference_steps_list 17 17 17 \