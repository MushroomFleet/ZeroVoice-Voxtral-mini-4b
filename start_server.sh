#!/bin/bash
# Voxtral TTS vLLM Server Launcher
# Run inside WSL2 Ubuntu-22.04

export HOME=/home/genuine
source $HOME/.local/bin/env
cd $HOME/voxtral-tts
source .venv/bin/activate

# Add conda-forge gcc to PATH for triton compilation
export PATH="$HOME/.micromamba/envs/gcc/bin:$PATH"
export CC="$HOME/.micromamba/envs/gcc/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$HOME/.micromamba/envs/gcc/bin/x86_64-conda-linux-gnu-g++"

# Disable cuDNN (incompatible with driver 566.36 on WSL2)
export TORCH_CUDNN_ENABLED=0

export PYTHONUNBUFFERED=1

echo "=== Starting Voxtral TTS Server ==="
echo "CC=$CC"
echo "Python: $(python --version)"
echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)')"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "==================================="

exec vllm serve /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603 --omni --enforce-eager
