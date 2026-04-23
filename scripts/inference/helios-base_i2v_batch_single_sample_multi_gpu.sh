export PYTHONPATH=$PWD:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29505 infer_helios_batch.py \
    --enable_parallelism \
    --cp_backend "ulysses" \
    --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --transformer_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --sample_type "i2v" \
    --num_frames 99 \
    --fps 24 \
    --task_dir "eval/intergs/samples/captions_frames" \
    --guidance_scale 5.0 \
    --enable_compile \
    --output_folder "eval/intergs/samples/output_single_sample_multi_gpu"
