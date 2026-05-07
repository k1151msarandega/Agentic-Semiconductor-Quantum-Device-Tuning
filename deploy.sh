#!/bin/bash
# deploy.sh — SimQuantum startup for AMD MI300X
# Run from repo root: bash deploy.sh

set -e

CONDA_ENV="qdots"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
VLLM_PORT=8000
STREAMLIT_PORT=8501

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SimQuantum — AMD MI300X startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Ensure script runs from its own directory (repo root)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# ─────────────────────────────────────────────────────────────────────────────
# Auto-update repo from GitHub
# ─────────────────────────────────────────────────────────────────────────────
echo "► Updating repository from GitHub..."
git fetch --all
git reset --hard origin/main
echo "  ✓ Repo is up to date"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ensure conda is initialized (even in non-interactive shells)
# ─────────────────────────────────────────────────────────────────────────────
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
else
    echo "✗ Could not find conda.sh — is Miniconda installed?"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Auto-create conda env if missing
# ─────────────────────────────────────────────────────────────────────────────
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo "► Creating conda env '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

echo "► Activating conda env '$CONDA_ENV'..."
conda activate "$CONDA_ENV"
echo "  Python: $(python --version)"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install required dependencies (GPU-safe)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "► Ensuring dependencies are installed..."
pip install -q streamlit plotly openai tqdm pyyaml joblib
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
pip install -q vllm
echo "  ✓ Dependencies ready"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Kill anything already using ports
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "► Clearing ports $VLLM_PORT and $STREAMLIT_PORT..."
fuser -k ${VLLM_PORT}/tcp 2>/dev/null && echo "  killed process on $VLLM_PORT" || echo "  port $VLLM_PORT was free"
fuser -k ${STREAMLIT_PORT}/tcp 2>/dev/null && echo "  killed process on $STREAMLIT_PORT" || echo "  port $STREAMLIT_PORT was free"
sleep 1

# ─────────────────────────────────────────────────────────────────────────────
# 5. Set environment variables for the app
# ─────────────────────────────────────────────────────────────────────────────
export QDOT_LLM_BASE_URL="http://localhost:${VLLM_PORT}"
export QDOT_LLM_MODEL="$MODEL"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Start vLLM in background
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "► Starting vLLM ($MODEL) on port $VLLM_PORT..."
nohup vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.4 \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID (logs: tail -f /tmp/vllm.log)"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Wait for vLLM to be ready
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "► Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "  ✓ vLLM is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ✗ vLLM did not start in 60s. Check: tail -f /tmp/vllm.log"
        echo "    Streamlit will still launch — Dr. Q will show as offline."
    fi
    printf "."
    sleep 1
done
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 8. Launch Streamlit
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "► Starting Streamlit on port $STREAMLIT_PORT..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App:  http://$(hostname -I | awk '{print $1}'):$STREAMLIT_PORT"
echo "  vLLM: http://localhost:$VLLM_PORT/v1/models"
echo "  Logs: tail -f /tmp/vllm.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run app.py \
    --server.port $STREAMLIT_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
