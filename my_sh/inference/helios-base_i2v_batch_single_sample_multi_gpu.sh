export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/beijing-c/workspace/hxj/.cache/huggingface"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29508 infer_helios_batch.py \
    --enable_parallelism \
    --cp_backend "ulysses" \
    --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --transformer_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --sample_type "i2v" \
    --num_frames 241 \
    --fps 24 \
    --task_dir "/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/eval/spatialvid/samples" \
    --guidance_scale 5.0 \
    --output_folder "/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/eval/spatialvid/samples/output/base_single_sample_multi_gpu"
