#!/bin/bash
# run_harmbench_agent.sh

# Get absolute path of repo root
REPO_ROOT=$(pwd)
CONFIG_PATH="${REPO_ROOT}/config"

# --- CRITICAL RAY FIXES ---
# 1. Disable Dashboard (prevents startup crash)
export RAY_INCLUDE_DASHBOARD=false
export RAY_DEDUP_LOGS=0

# 2. Fix Socket Error on NFS/Bigtemp
# Ray crashes on network drives. We symlink the repo's tmp/ray to local /tmp
LOCAL_RAY_DIR="/tmp/ray_local_$USER"
REPO_RAY_DIR="${REPO_ROOT}/tmp/ray"

echo "Setting up local Ray storage at $LOCAL_RAY_DIR..."
# Clean up previous links/dirs
rm -rf "$REPO_RAY_DIR" 
mkdir -p "$LOCAL_RAY_DIR"
# Ensure parent tmp dir exists
mkdir -p "$(dirname "$REPO_RAY_DIR")"
# Create the symlink: writes to REPO_RAY_DIR will go to LOCAL_RAY_DIR
ln -s "$LOCAL_RAY_DIR" "$REPO_RAY_DIR"

# 3. Cleanup & Limits
ulimit -n 65536
pkill -u $USER -f ray
pkill -u $USER -f retrieval_server.py
# Sleep to let ports clear
sleep 3

# --- CONFIGURATION ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=4
TP=1
ROLLOUT_MEM=0.6

MODEL="Qwen/Qwen2.5-3B-Instruct"
DATA_DIR="${REPO_ROOT}/data/harmbench"
VAL_DATA="$DATA_DIR/test.parquet"
RESULTS_DIR="${REPO_ROOT}/results/harmbench_agent_t0"
TOOL_CONFIG="${CONFIG_PATH}/search_tool_config.yaml"

# --- 1. START RETRIEVAL SERVER ---
echo "Starting Retrieval Server..."
python search/retrieval_server.py \
    --index_path='./corpus/e5_Flat.index' \
    --corpus_path='./corpus/wiki-18.jsonl' \
    --retriever_model='intfloat/e5-base-v2' \
    --retriever_name='e5' \
    --faiss_gpu \
    --topk 3 &

# Wait for server to initialize
sleep 20

# --- 2. RUN AGENT ROLLOUT ---
echo "Starting Agent Rollout..."

python -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='search_multiturn_grpo' \
    data.train_files="$VAL_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MEM \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.logger='["console"]' \
    trainer.project_name="dr-zero-safety" \
    trainer.experiment_name="harmbench_t0" \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    ++trainer.val_only=True \
    ++trainer.val_before_train=True \
    trainer.validation_data_dir=$RESULTS_DIR \
    +retriever.url="http://127.0.0.1:8000/retrieve" \
    +retriever.topk=3

# --- 3. CLEANUP ---
echo "Killing Retrieval Server..."
pkill -u $USER -f retrieval_server.py