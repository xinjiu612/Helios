# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --transformer_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --sample_type "i2v" \
    --num_frames 99 \
    --fps 24 \
    --image_path "output_helios/helios-base/test/fast_move.png" \
    --prompt "Cinematic fast truck-in and dolly-in toward the left-side tree in a modern office scene. The first three seconds must show continuously increasing speed toward the tree, with strong depth compression and obvious background streaking (white curtains, desks, lighting rails, equipment racks). At the 3-second boundary, the camera brakes instantly and locks in front of the tree. For the next two seconds, keep a rigid lock-off composition and stable details in leaves, trunk, drapes, and floor cables." \
    --guidance_scale 5.0 \
    --enable_compile \
    --output_folder "./output_helios/helios-base"


    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \