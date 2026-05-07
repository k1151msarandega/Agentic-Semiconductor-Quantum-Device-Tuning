#!/bin/bash
# deploy.sh — SimQuantum startup for AMD MI300X
# Run from repo root: bash deploy.sh
# Access the app at http://<your-mi300x-ip>:8501

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

# ── 1. Kill anything already using these ports ─────────────────────────────
echo "► Clearing ports $VLLM_PORT and $STREAMLIT_PORT..."
fuser -k ${VLLM_PORT}/tcp 2>/dev/null && echo "  killed process on $VLLM_PORT" || echo "  port $VLLM_PORT was free"
fuser -k ${STREAMLIT_PORT}/tcp 2>/dev/null && echo "  killed process on $STREAMLIT_PORT" || echo "  port $STREAMLIT_PORT was free"
sleep 1

# ── 2. Activate conda env ──────────────────────────────────────────────────
echo ""
echo "► Activating conda env '$CONDA_ENV'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "  Python: $(python --version)"

# ── 3. Install any missing deps quietly ───────────────────────────────────
echo ""
echo "► Checking dependencies..."
pip install -q streamlit plotly openai 2>/dev/null
echo "  Done."

# ── 4. Set environment variables ───────────────────────────────────────────
export QDOT_LLM_BASE_URL="http://localhost:${VLLM_PORT}"
export QDOT_LLM_MODEL="$MODEL"

# ── 5. Start vLLM in background ────────────────────────────────────────────
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

# ── 6. Wait for vLLM to be ready ──────────────────────────────────────────
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

# ── 7. Launch Streamlit ────────────────────────────────────────────────────
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
