#!/bin/bash
# start.sh — SimQuantum daily launch script
# ONE command to start vLLM + Streamlit on every droplet boot.
# Assumes deploy.sh was already run once on this droplet.

CONDA_ENV="qdots"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
VLLM_PORT=8000
STREAMLIT_PORT=8501

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SimQuantum — Launch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$(dirname "$0")"

# ── Init conda ────────────────────────────────────────────────────────────────
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Get the FULL path to Python and vLLM in this env — no ambiguity
PYTHON="$(which python)"
VLLM_BIN="$(which vllm)"
echo "  Python : $PYTHON"
echo "  vLLM   : $VLLM_BIN"

# ── Step 1: Start vLLM (if not already running) ───────────────────────────────
if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
    echo "► vLLM already running on :$VLLM_PORT ✓"
else
    echo "► Starting vLLM..."
    export HIP_VISIBLE_DEVICES=0
    export ROCR_VISIBLE_DEVICES=0
    export VLLM_TARGET_DEVICE=rocm
    export HSA_OVERRIDE_GFX_VERSION=9.4.2   # MI300X

    # Launch using the FULL PATH to vllm in the conda env
    nohup "$VLLM_BIN" serve "$MODEL" \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --gpu-memory-utilization 0.45 \
        --max-model-len 4096 \
        > /tmp/vllm.log 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"

    echo -n "  Waiting for vLLM to be ready (up to 120s)..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
            echo " ✓"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo ""
            echo "  ✗ vLLM process died. Last 20 lines of log:"
            tail -20 /tmp/vllm.log
            exit 1
        fi
        printf "."; sleep 1
    done

    # Final check
    if ! curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        echo ""
        echo "  ✗ vLLM did not start in time. Check /tmp/vllm.log"
        exit 1
    fi
fi

# ── Step 2: Kill any leftover Streamlit (not vLLM!) ──────────────────────────
echo "► Clearing old Streamlit processes..."
pkill -f "streamlit run" 2>/dev/null || true
sleep 1

# ── Step 3: Start Streamlit ───────────────────────────────────────────────────
echo "► Starting Streamlit on :$STREAMLIT_PORT..."
export QDOT_LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export QDOT_LLM_MODEL="$MODEL"

nohup "$PYTHON" -m streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo -n "  Waiting for Streamlit..."
for i in $(seq 1 30); do
    if curl -s http://localhost:$STREAMLIT_PORT > /dev/null 2>&1; then
        echo " ✓"
        break
    fi
    printf "."; sleep 1
done

# ── Done ─────────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_DROPLET_IP")
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App   → http://${PUBLIC_IP}:${STREAMLIT_PORT}"
echo "  vLLM  → http://localhost:${VLLM_PORT}/v1/models"
echo ""
echo "  Logs:"
echo "    tail -f /tmp/vllm.log"
echo "    tail -f /tmp/streamlit.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
