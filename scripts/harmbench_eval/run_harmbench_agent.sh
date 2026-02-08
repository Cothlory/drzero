#!/bin/bash
# run_harmbench_agent.sh

REPO_ROOT=$(pwd)
CONFIG_PATH="${REPO_ROOT}/config"

# --- 1. CLEANUP ---
# Kill any lingering processes
pkill -u $USER -f ray
pkill -u $USER -f retrieval_server.py

# Remove the 'tmp/ray' directory completely to clear any broken symlinks or locks
rm -rf "${REPO_ROOT}/tmp/ray"
# Ensure the local /tmp dir we tried to use before is gone too, to avoid confusion
rm -rf "/tmp/ray_local_$USER"

# --- 2. THE SIMPLEST FIX ---
# Tell Ray: "I know the disk is slow/networked, please don't crash."
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
# Disable dashboard to prevent port conflicts/startup hangs
export RAY_INCLUDE_DASHBOARD=0
export RAY_DEDUP_LOGS=0

# --- 3. CONFIGURATION ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=4
TP=1
ROLLOUT_MEM=0.6

MODEL="Qwen/Qwen2.5-3B-Instruct"
DATA_DIR="${REPO_ROOT}/data/harmbench"
VAL_DATA="$DATA_DIR/test.parquet"
RESULTS_DIR="${REPO_ROOT}/results/harmbench_agent_t0"
TOOL_CONFIG="${CONFIG_PATH}/search_tool_config.yaml"

# --- 4. START RETRIEVAL SERVER ---
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

# --- 5. RUN AGENT ---
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

# --- 6. CLEANUP ---
echo "Killing Retrieval Server..."
pkill -u $USER -f retrieval_server.py