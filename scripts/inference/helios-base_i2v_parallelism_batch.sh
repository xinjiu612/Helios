
export PYTHONPATH=$PWD:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29505 infer_helios_batch.py \
    --enable_parallelism \
    --cp_backend "ulysses" \
    --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --transformer_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --sample_type "i2v" \
    --num_frames 99 \
    --fps 24 \
    --task_dir "output_helios/helios-base/test" \
    --guidance_scale 5.0 \
    --enable_compile \
    --output_folder "./output_helios/helios-base"

    # --is_skip_first_chunk \

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
#     --guidance_scale 5.0 \
#     --enable_compile \
#     --output_folder "./output_helios/helios-base"
