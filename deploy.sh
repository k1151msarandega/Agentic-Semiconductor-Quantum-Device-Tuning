#!/bin/bash
set -e

CONDA_ENV="qdots"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
VLLM_PORT=8000
STREAMLIT_PORT=8501

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SimQuantum — MI300X Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ----------------------------------------------------------------------------
# 0. Ensure script runs from its own directory
# ----------------------------------------------------------------------------
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# 1. Update repo
# ----------------------------------------------------------------------------
echo "► Updating repository..."
git fetch --all
git reset --hard origin/main
echo "  ✓ Repo updated"

# ----------------------------------------------------------------------------
# 2. Proper conda initialization (Claude Fix #1)
# ----------------------------------------------------------------------------
echo "► Initializing conda..."
source /root/miniconda3/etc/profile.d/conda.sh

# ----------------------------------------------------------------------------
# 3. Activate environment
# ----------------------------------------------------------------------------
echo "► Activating conda env '$CONDA_ENV'..."
conda activate "$CONDA_ENV"
echo "  ✓ Python: $(python --version)"

# ----------------------------------------------------------------------------
# 4. Install ONLY non-vLLM dependencies (Claude Fix #2)
# ----------------------------------------------------------------------------
echo "► Ensuring Python dependencies are installed..."
pip install --upgrade pip
pip install streamlit plotly openai
echo "  ✓ Dependencies ready"

# ----------------------------------------------------------------------------
# 5. Start vLLM if not already running
# ----------------------------------------------------------------------------
echo "► Checking vLLM..."
if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
    echo "  ✓ vLLM already running on port $VLLM_PORT"
else
    echo "  Starting vLLM..."
    conda deactivate
    export HIP_VISIBLE_DEVICES=0
    export ROCR_VISIBLE_DEVICES=0
    export VLLM_TARGET_DEVICE=rocm
    nohup vllm serve $MODEL \
        --host 0.0.0.0 --port $VLLM_PORT \
        --gpu-memory-utilization 0.4 \
        > /tmp/vllm.log 2>&1 &
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    echo -n "  Waiting for vLLM..."
    for i in $(seq 1 90); do
        curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1 && echo " ✓" && break
        printf "."; sleep 1
    done
fi

# ----------------------------------------------------------------------------
# 6. Kill orphaned Streamlit processes
# ----------------------------------------------------------------------------
echo "► Killing orphaned Streamlit processes..."
pkill -9 streamlit || true
pkill -9 python || true

# ----------------------------------------------------------------------------
# 7. Start Streamlit
# ----------------------------------------------------------------------------
echo "► Starting Streamlit on port $STREAMLIT_PORT..."
nohup streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    > /tmp/streamlit.log 2>&1 &

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App:  http://$(curl -s ifconfig.me):$STREAMLIT_PORT"
echo "  vLLM: http://localhost:$VLLM_PORT/v1/models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
