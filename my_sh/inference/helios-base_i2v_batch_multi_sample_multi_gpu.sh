export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/beijing-c/workspace/hxj/.cache/huggingface"
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master_port 29509 infer_helios_batch.py \
    --base_model_path "/beijing-c/models/BestWishYSH/Helios-Base" \
    --transformer_path "/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/ablation_stage_1_spatialvid/merge/step_5000" \
    --sample_type "i2v" \
    --num_frames 121 \
    --fps 24 \
    --task_dir "/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/eval/real_world/nothing" \
    --guidance_scale 5.0 \
    --output_folder "/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/eval/real_world/output/nothing"
