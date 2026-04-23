export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME="/beijing-c/workspace/hxj/.cache/huggingface"
# export KERNELS_CACHE="${HF_HOME}/hub"
# export HF_HUB_OFFLINE=1

BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/beijing-c/models/BestWishYSH/Helios-Base"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/ablation_stage_1_spatialvid/checkpoint-8000"}
TASK_DIR=${TASK_DIR:-"/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/eval/spatialvid/samples_16_8000steps/samples"}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-"/beijing-c/workspace/hxj/videomodel/Helios_new/Helios/eval/spatialvid/samples_16_8000steps/output"}

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 --master_port 29509 infer_helios_batch.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --transformer_path "$BASE_MODEL_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --fuse_lora \
    --sample_type "i2v" \
    --num_frames 241 \
    --fps 24 \
    --task_dir "$TASK_DIR" \
    --guidance_scale 5.0 \
    --output_folder "$OUTPUT_FOLDER"
