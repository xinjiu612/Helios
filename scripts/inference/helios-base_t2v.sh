# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

export PYTHONPATH=$PWD:$PYTHONPATH
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29505 infer_helios_batch_copy.py \
        --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
        --transformer_path "/beijing-c/models/BestWishYSH/Helios-Base" \
        --sample_type "t2v" \
        --num_frames 144 \
        --guidance_scale 3.0 \
        --fps 24 \
        --prompt_txt_path "./prompts14.txt" \
        --output_folder "./output_helios/helios-base-run7"


    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \