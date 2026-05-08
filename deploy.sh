#!/bin/bash
# deploy.sh — SimQuantum MI300X setup
# Run ONCE on a fresh droplet to install everything.
# After this, use start.sh for all subsequent launches.
set -e

CONDA_ENV="qdots"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SimQuantum — MI300X First-Time Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$(dirname "$0")"

# 1. Conda init
source /root/miniconda3/etc/profile.d/conda.sh

# 2. Create env if missing
if ! conda env list | grep -q "^${CONDA_ENV}"; then
    echo "► Creating conda env '$CONDA_ENV' (Python 3.11)..."
    conda create -y -n "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"
echo "  ✓ Python: $(python --version)"

# 3. Install ROCm PyTorch (skip if already present)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "► Installing ROCm PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
fi

# 4. Install vLLM for ROCm
if ! python -c "import vllm" 2>/dev/null; then
    echo "► Installing vLLM (ROCm)..."
    pip install vllm
fi

# 5. Install app dependencies
echo "► Installing app dependencies..."
pip install streamlit==1.57.0 plotly openai numpy scipy scikit-learn tqdm
pip install -e . --quiet

# 6. Pull latest code
echo "► Pulling latest code..."
git fetch --all
git reset --hard origin/main

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete. Now run: bash start.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
