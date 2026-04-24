#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# 避免共享工作区里旧 Triton 产物与线上 worker glibc 不匹配（见 GLIBC_2.34 / cuda_utils.so）
if [ -z "${TRITON_CACHE_DIR:-}" ]; then
    _h="$(hostname 2>/dev/null | tr -cd 'a-zA-Z0-9_.-' || echo nohost)"
    export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_autotune_helios_${_h}"
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="/beijing-c/workspace/hxj/.cache/huggingface"
# Optional GPU selection:
#   GPU_IDS=0,1,2,3 bash my_sh/train/train_ddp.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash my_sh/train/train_ddp.sh
if [ ! -z "${GPU_IDS}" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU_IDS}
fi

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
if [ ! -z "${ARNOLD_RDMA_DEVICE}" ]; then
    export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE
fi
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=3600000

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=NVL

export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
#################################################################

# #################################################################
# ## DIST
# #################################################################
# MASTER_ADDR=$ARNOLD_WORKER_0_HOST
# ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# export MASTER_PORT=${ports[0]}
# NNODES=$ARNOLD_WORKER_NUM
# NODE_RANK=$ARNOLD_ID
# GPUS_PER_NODE=$ARNOLD_WORKER_GPU
# # GPUS_PER_NODE=1
# # NNODES=1
# # NODE_RANK=0
# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# if [ ! -z $RDZV_BACKEND ]; then
#     DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_id 9863 --rdzv_backend c10d"
#     export NCCL_SHM_DISABLE=1
# fi

# echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"

#################################################################
## ACCELERATE CONFIG (supports PAI-DLC / Arnold / standalone)
#################################################################
# PAI-DLC sets: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
# Arnold  sets: ARNOLD_WORKER_0_HOST, METIS_WORKER_0_PORT, ARNOLD_WORKER_NUM, ARNOLD_ID
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"

if [ -z "${MASTER_PORT}" ]; then
    if [ ! -z "${METIS_WORKER_0_PORT}" ]; then
        ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
        export MASTER_PORT=${ports[0]}
    else
        export MASTER_PORT=29500
    fi
fi

NUM_MACHINES="${NUM_MACHINES:-${WORLD_SIZE:-${ARNOLD_WORKER_NUM:-1}}}"
MACHINE_RANK="${MACHINE_RANK:-${RANK:-${ARNOLD_ID:-0}}}"

if [ ! -z "${CUDA_VISIBLE_DEVICES}" ]; then
    clean_cuda_visible_devices=$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')
    NUM_PROCESSES_PER_MACHINE=$(echo "${clean_cuda_visible_devices}" | awk -F',' '{print NF}')
elif [ ! -z "${NPROC_PER_NODE}" ]; then
    NUM_PROCESSES_PER_MACHINE="${NPROC_PER_NODE}"
elif [ ! -z "${ARNOLD_WORKER_GPU}" ]; then
    NUM_PROCESSES_PER_MACHINE="${ARNOLD_WORKER_GPU}"
else
    NUM_PROCESSES_PER_MACHINE=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${NUM_PROCESSES_PER_MACHINE}" -eq 0 ]; then
        NUM_PROCESSES_PER_MACHINE=1
    fi
fi

# Override for debugging (uncomment as needed):
# export CUDA_VISIBLE_DEVICES=0
# NUM_PROCESSES_PER_MACHINE=1
# NUM_MACHINES=1
# MACHINE_RANK=0

ACCELERATE_ARGS="--num_machines $NUM_MACHINES --machine_rank $MACHINE_RANK --num_processes $((NUM_PROCESSES_PER_MACHINE*NUM_MACHINES)) --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"

echo -e "\033[31mCUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}\033[0m"
echo -e "\033[31mMASTER_ADDR: ${MASTER_ADDR}\033[0m"
echo -e "\033[31mMASTER_PORT: ${MASTER_PORT}\033[0m"
echo -e "\033[31mNUM_MACHINES: ${NUM_MACHINES}\033[0m"
echo -e "\033[31mMACHINE_RANK: ${MACHINE_RANK}\033[0m"
echo -e "\033[31mNUM_PROCESSES_PER_MACHINE: ${NUM_PROCESSES_PER_MACHINE}\033[0m"
echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"

accelerate launch \
    $ACCELERATE_ARGS \
    train_helios_val_i2v_5hz.py \
    --config my_sh/train/config_new/stage_1_init_sptialvid_5hz_lw3_8card.yaml \
    2>&1 | tee ./train_5hz.log
# accelerate launch \
#     $ACCELERATE_ARGS \
#     --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
#     train_helios.py \
#     --config scripts/training/configs/stage_1_init.yaml \
#     2>&1 | tee ./train.log
