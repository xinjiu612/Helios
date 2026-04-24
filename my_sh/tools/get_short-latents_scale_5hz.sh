export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR
#################################################################
## Torch
#################################################################
export TOKENIZERS_PARALLELISM=false
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export TORCH_NCCL_ENABLE_MONITORING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
#################################################################


#################################################################
## NCCL
#################################################################
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=3600000

export NCCL_DEBUG=WARN  # disable the verbose NCCL logs
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0  # was 1
export NCCL_SHM_DISABLE=0  # was 1
export NCCL_P2P_LEVEL=NVL

export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
#################################################################

#################################################################
## DIST
#################################################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12345}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
if [ -z "${GPUS_PER_NODE:-}" ]; then
    clean_cuda_visible_devices=$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')
    GPUS_PER_NODE=$(echo "${clean_cuda_visible_devices}" | awk -F',' '{print NF}')
fi

# export CUDA_VISIBLE_DEVICES=1
# MASTER_PORT=12345
# GPUS_PER_NODE=1
# NNODES=1
# NODE_RANK=0

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
if [ ! -z $RDZV_BACKEND ]; then
    DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_id 9863 --rdzv_backend c10d"
    export NCCL_SHM_DISABLE=1
fi

echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"

#################################################################
# Add project root to PYTHONPATH to find 'helios' module
export PYTHONPATH=$PYTHONPATH:$(pwd)

BASE_VIDEO_PATH="${BASE_VIDEO_PATH:-/beijing-c/datasets/hxj_video_model/SpatialVID/SpatialVID}"
BASE_CSV_PATH="${BASE_CSV_PATH:-/beijing-c/datasets/hxj_video_model/SpatialVID/SpatialVID}"
CSV_PATH="${CSV_PATH:-helios_data_straight_5hz.part_002_of_002.json}"
BASE_OUTPUT_LATENT_PATH="${BASE_OUTPUT_LATENT_PATH:-/world_model_data/SpatialVID/5fps}"
OUTPUT_LATENT_PATH="${OUTPUT_LATENT_PATH:-latents_short_straight_lw3}"
LATENT_WINDOW_SIZE="${LATENT_WINDOW_SIZE:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"

EXTRA_ARGS=()
if [ "${DISABLE_PERSISTENT_WORKERS:-1}" = "1" ]; then
    EXTRA_ARGS+=(--disable_persistent_workers)
fi

torchrun $DISTRIBUTED_ARGS \
    my_sh/tools/get_short-latents_scale_5hz.py \
    --base_video_path "$BASE_VIDEO_PATH" \
    --base_csv_path "$BASE_CSV_PATH" \
    --csv_path "$CSV_PATH" \
    --base_output_latent_path "$BASE_OUTPUT_LATENT_PATH" \
    --output_latent_path "$OUTPUT_LATENT_PATH" \
    --latent_window_size "$LATENT_WINDOW_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR" \
    "${EXTRA_ARGS[@]}" \
    "$@"
