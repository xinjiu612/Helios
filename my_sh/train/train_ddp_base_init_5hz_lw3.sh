#!/bin/bash
# 多机训练：2 台 worker × 8×A100 = 16 进程（Accelerate + torch DDP）
#
# 火山 MLP：平台会注入 MLP_*，入口一般只需 conda + bash 本脚本，无需再配 MASTER / 端口。
#
# 手动裸机（两台各执行一次，自行 export）：
#   机器0（主节点，需可被机器1访问的 IP）:
#     export MASTER_ADDR=<机器0的网卡IP>   # 不要用127.0.0.1
#     export MASTER_PORT=29500             # 防火墙需放行 TCP
#     export MACHINE_RANK=0
#     bash my_sh/train/train_ddp_base_post_multinode_2x8.sh
#
#   机器1:
#     export MASTER_ADDR=<同上>
#     export MASTER_PORT=29500
#     export MACHINE_RANK=1
#     bash my_sh/train/train_ddp_base_post_multinode_2x8.sh
#
# 可选覆盖：NUM_MACHINES NPROC_PER_NODE GPU_IDS NCCL_SOCKET_IFNAME
#
# 火山引擎机器学习平台（自定义任务 / 多机）会注入环境变量，一般无需再配 MASTER 与端口：
#   MLP_WORKER_0_HOST  MLP_WORKER_0_PORT  MLP_ROLE_INDEX  MLP_WORKER_NUM  MLP_WORKER_GPU
# 说明见：https://www.volcengine.com/docs/6459/75222
# 入口示例：source .../miniconda3/bin/activate helios && bash 本脚本
# （或 source .../bin/activate 后 conda activate helios，与现脚本等价）
set -euo pipefail

source /beijing-c/workspace/hxj/miniconda3/bin/activate
conda activate helios

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_ENDPOINT=https://hf-mirror.com
# Triton 缓存若在共享盘且曾在 glibc 较新环境编译，在火山等 glibc 较旧 worker 会报
# GLIBC_2.34 not found（cuda_utils.so）。默认用本机 /tmp，每机独立重编译。
# 需要固定目录时可事先 export TRITON_CACHE_DIR=...
if [ -z "${TRITON_CACHE_DIR:-}" ]; then
    _h="$(hostname 2>/dev/null | tr -cd 'a-zA-Z0-9_.-' || echo nohost)"
    export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_autotune_helios_${_h}_${MLP_ROLE_INDEX:-0}"
fi
# 默认关闭 wandb（无 TTY/无 key 的集群上避免 init 报错）；需要时 export WANDB_MODE=online WANDB_API_KEY=...
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

# 每台8 卡；也可用 GPU_IDS 选子集
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [ -n "${GPU_IDS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export TORCH_NCCL_ENABLE_MONITORING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"

export NCCL_IB_GID_INDEX=3
if [ -n "${ARNOLD_RDMA_DEVICE:-}" ]; then
    export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_TIMEOUT=3600000
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
# 容器内 IB 设备不可用时会出现 ibv_modify_qp / errno 19，可在任务环境设 NCCL_IB_DISABLE=1 走 Socket（略慢但稳）
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22

# ---------- 分布式（与单机脚本的关键区别）----------
# 机器数不要用 PyTorch 的 WORLD_SIZE（那是总进程数）；优先用手动 NUM_MACHINES 或 MLP_WORKER_NUM。
export NUM_MACHINES="${NUM_MACHINES:-${MLP_WORKER_NUM:-2}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${MLP_WORKER_GPU:-8}}"
# 机器编号：火山用 MLP_ROLE_INDEX；勿与部分平台上「进程级」的 RANK 混用。
export MACHINE_RANK="${MACHINE_RANK:-${MLP_ROLE_INDEX:-0}}"

export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
if [ "${MASTER_ADDR}" = "127.0.0.1" ] && [ "${NUM_MACHINES}" -gt 1 ]; then
    echo "ERROR: 多机时请设置 MASTER_ADDR（或依赖 MLP_WORKER_0_HOST），不要用 127.0.0.1。" >&2
    exit 1
fi

if [ -z "${MASTER_PORT:-}" ]; then
    if [ -n "${MLP_WORKER_0_PORT:-}" ]; then
        export MASTER_PORT="$(echo "${MLP_WORKER_0_PORT}" | cut -d',' -f1)"
    else
        export MASTER_PORT=29500
    fi
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    clean_cuda_visible_devices=$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')
    NUM_PROCESSES_PER_MACHINE=$(echo "${clean_cuda_visible_devices}" | awk -F',' '{print NF}')
else
    NUM_PROCESSES_PER_MACHINE="${NPROC_PER_NODE}"
fi

if [ "${NUM_PROCESSES_PER_MACHINE}" != "${NPROC_PER_NODE}" ]; then
    echo "WARN: 当前 CUDA_VISIBLE_DEVICES 对应每机进程数=${NUM_PROCESSES_PER_MACHINE}，与 NPROC_PER_NODE=${NPROC_PER_NODE} 不一致；将以可见 GPU 数为准。" >&2
fi

TOTAL_PROCESSES=$((NUM_PROCESSES_PER_MACHINE * NUM_MACHINES))

ACCELERATE_ARGS="--num_machines ${NUM_MACHINES} --machine_rank ${MACHINE_RANK} --num_processes ${TOTAL_PROCESSES} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"

if [ -n "${MLP_WORKER_0_HOST:-}" ] || [ -n "${MLP_WORKER_NUM:-}" ]; then
    echo -e "\033[32mMLP 注入: MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST:-} MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT:-} MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-} MLP_WORKER_NUM=${MLP_WORKER_NUM:-} MLP_WORKER_GPU=${MLP_WORKER_GPU:-}\033[0m"
fi

echo -e "\033[31mCUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}\033[0m"
echo -e "\033[31mMASTER_ADDR: ${MASTER_ADDR}\033[0m"
echo -e "\033[31mMASTER_PORT: ${MASTER_PORT}\033[0m"
echo -e "\033[31mNUM_MACHINES: ${NUM_MACHINES}\033[0m"
echo -e "\033[31mMACHINE_RANK: ${MACHINE_RANK}\033[0m"
echo -e "\033[31mNUM_PROCESSES_PER_MACHINE: ${NUM_PROCESSES_PER_MACHINE}\033[0m"
echo -e "\033[31mTOTAL_PROCESSES (world): ${TOTAL_PROCESSES}\033[0m"
echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"

# accelerate launch \
#     ${ACCELERATE_ARGS} \
#     train_helios_val_i2v.py \
#     --config my_sh/train/config_new/stage_1_post_sptialvid.yaml \
#     2>&1 | tee "./train_multinode_rank${MACHINE_RANK}.log"
accelerate launch \
    $ACCELERATE_ARGS \
    train_helios_val_i2v_5hz.py \
    --config my_sh/train/config_new/stage_1_init_sptialvid_5hz_lw3.yaml \
    2>&1 | tee "./train_multinode_rank_5hz${MACHINE_RANK}.log"