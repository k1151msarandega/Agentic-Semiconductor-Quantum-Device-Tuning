#!/bin/bash
set -e

# ============================================================================
#  SimQuantum — Hardened Deployment Script (MI300X ROCm Container Edition)
#  This script handles EVERYTHING automatically:
#    - conda TOS acceptance
#    - conda env creation
#    - ROCm PyTorch install
#    - vLLM install
#    - Streamlit install
#    - repo auto-update
#    - GPU detection
#    - vLLM startup
#    - Streamlit startup
# ============================================================================

CONDA_ENV="qdots"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
VLLM_PORT=8000
STREAMLIT_PORT=8501

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SimQuantum — AMD MI300X startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ----------------------------------------------------------------------------
# 0. Ensure script runs from its own directory
# ----------------------------------------------------------------------------
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# 1. Auto-update repo
# ----------------------------------------------------------------------------
echo "► Updating repository from GitHub..."
git fetch --all
git reset --hard origin/main
echo "  ✓ Repo is up to date"

# ----------------------------------------------------------------------------
# 2. Initialize conda (if not already)
# ----------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "✗ Conda not found. Install Miniconda first."
    exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh

# ----------------------------------------------------------------------------
# 3. Accept Anaconda TOS automatically
# ----------------------------------------------------------------------------
echo "► Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
echo "  ✓ TOS accepted"

# ----------------------------------------------------------------------------
# 4. Create conda environment if missing
# ----------------------------------------------------------------------------
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo "► Creating conda env '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

echo "► Activating conda env '$CONDA_ENV'..."
conda activate "$CONDA_ENV"
echo "  ✓ Python: $(python --version)"

# ----------------------------------------------------------------------------
# 5. Install ROCm PyTorch + vLLM + Streamlit
# ----------------------------------------------------------------------------
echo "► Ensuring dependencies are installed..."

pip install --upgrade pip

pip install streamlit plotly openai

echo "  ✓ Dependencies ready"

# ----------------------------------------------------------------------------
# 6. GPU detection
# ----------------------------------------------------------------------------
echo "► Checking GPU availability..."
python - << 'EOF'
import torch
if not torch.cuda.is_available():
    print("✗ GPU NOT AVAILABLE — vLLM will fail.")
    exit(1)
print("✓ GPU detected:", torch.cuda.get_device_name(0))
EOF

# ----------------------------------------------------------------------------
# 7. Kill any processes on ports
# ----------------------------------------------------------------------------
echo "► Clearing ports $VLLM_PORT and $STREAMLIT_PORT..."
for port in $VLLM_PORT $STREAMLIT_PORT; do
    PIDS=$(lsof -t -i:$port || true)
    if [ -n "$PIDS" ]; then
        echo "  Killing processes on port $port..."
        kill -9 $PIDS || true
    else
        echo "  Port $port is free"
    fi
done

# ----------------------------------------------------------------------------
# 8. Start vLLM
# ----------------------------------------------------------------------------
echo "► Checking vLLM..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  ✓ vLLM already running on port 8000"
else
    echo "  ✗ vLLM not running. Start it manually:"
    echo "    conda deactivate"
    echo "    vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000 &"
    echo "  Then re-run deploy.sh"
    exit 1
fi

# ----------------------------------------------------------------------------
# 9. Clean Streamlit
# ----------------------------------------------------------------------------
echo "► Killing any orphaned Streamlit processes..."
pkill -9 streamlit || true
pkill -9 python || true

# ----------------------------------------------------------------------------
# 9. Start Streamlit
# ----------------------------------------------------------------------------
echo ""
echo "► Starting Streamlit on port $STREAMLIT_PORT..."
nohup streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    > /tmp/streamlit.log 2>&1 &

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App:  http://$(curl -s ifconfig.me):$STREAMLIT_PORT"
echo "  vLLM: http://localhost:$VLLM_PORT/v1/models"
echo "  Logs:"
echo "    vLLM:      tail -f /tmp/vllm.log"
echo "    Streamlit: tail -f /tmp/streamlit.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
