# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

export PYTHONPATH=$PWD:$PYTHONPATH
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29505 infer_helios.py \
        --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
        --transformer_path "ablation_stage_1_post_10000_intergs/merge" \
        --enable_parallelism \
        --cp_backend "ulysses" \
        --sample_type "t2v" \
        --num_frames 99 \
        --fps 24 \
        --prompt "The camera glides forward across a smooth, tiled floor within a narrow kitchen corridor lined with cabinetry. It approaches a large, rectangular stainless steel refrigerator standing against the far wall, positioned between a countertop and a doorway. The camera moves directly toward the appliance, passes closely by its right side, and continues its trajectory into the adjacent space." \
        --guidance_scale 5.0 \
        --output_folder "./output_helios/helios-base/test_2"




    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \