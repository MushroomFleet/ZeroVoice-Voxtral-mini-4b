<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:MEDIUM -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:WINDOWS+WSL2 -->
<!-- ZS:LANGUAGE:PYTHON,BASH -->

# Voxtral-4B-TTS-2603 Installation & Deployment

## Description

Complete installation, configuration, and deployment guide for **Mistral AI's Voxtral-4B-TTS-2603** text-to-speech model on a Windows 11 system with NVIDIA RTX 4090 GPU. Voxtral TTS is a frontier, open-weights text-to-speech model producing lifelike speech across 9 languages with 20 preset voices, 24 kHz audio output, and very low latency.

Because vLLM does not support Windows natively, the entire inference stack runs inside **WSL2 (Ubuntu)**. The model weights have already been cloned from HuggingFace and reside locally at `./Voxtral-4B-TTS-2603/`. This guide covers everything from WSL2 preparation through serving the model and running client inference requests.

**Target audience:** Developers and researchers evaluating Voxtral TTS for inference pipeline testing on a local RTX 4090 workstation.

**License:** CC BY-NC 4.0 (inherited from bundled voice reference datasets).

---

## Functionality

### Core Capabilities

- **Text-to-Speech inference** via an OpenAI-compatible HTTP API served by vLLM-Omni
- **20 preset voices** with instant adaptation (no fine-tuning required)
- **9 languages supported:** English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi
- **24 kHz audio output** in WAV, PCM, FLAC, MP3, AAC, and Opus formats
- **BF16 weights** running on a single GPU with >= 16GB VRAM (RTX 4090 has 24GB)
- **Streaming and batch inference** support
- **Optional Gradio web demo** for interactive testing

### Available Voices

| Voice ID | Language | Gender |
|---|---|---|
| `casual_male` | English | Male |
| `casual_female` | English | Female |
| `cheerful_female` | English | Female |
| `neutral_male` | English | Male |
| `neutral_female` | English | Female |
| `fr_male` | French | Male |
| `fr_female` | French | Female |
| `es_male` | Spanish | Male |
| `es_female` | Spanish | Female |
| `de_male` | German | Male |
| `de_female` | German | Female |
| `it_male` | Italian | Male |
| `it_female` | Italian | Female |
| `pt_male` | Portuguese | Male |
| `pt_female` | Portuguese | Female |
| `nl_male` | Dutch | Male |
| `nl_female` | Dutch | Female |
| `ar_male` | Arabic | Male |
| `hi_male` | Hindi | Male |
| `hi_female` | Hindi | Female |

### User Flow

```
[1] Prepare WSL2 + CUDA drivers
         |
[2] Create Python 3.12 environment (uv)
         |
[3] Install vLLM >= 0.18.0
         |
[4] Install vllm-omni from GitHub
         |
[5] Verify mistral_common >= 1.10.0
         |
[6] Start vLLM server with --omni flag
         |
[7] Send TTS requests via HTTP client
         |
[8] (Optional) Launch Gradio demo UI
```

---

## Technical Implementation

### Architecture Overview

```
+-------------------------------------------------------+
|  Windows 11 Host (RTX 4090)                           |
|                                                       |
|  +--------------------------------------------------+ |
|  |  WSL2 (Ubuntu 22.04+)                            | |
|  |                                                  | |
|  |  +-------------------+   +--------------------+  | |
|  |  | vLLM Server       |   | Model Weights      |  | |
|  |  | (port 8000)       |<--| ./Voxtral-4B-      |  | |
|  |  | --omni flag       |   |   TTS-2603/        |  | |
|  |  +-------------------+   +--------------------+  | |
|  |          |                                        | |
|  |  +-------------------+   +--------------------+  | |
|  |  | vllm-omni plugin  |   | CUDA 12.8+         |  | |
|  |  | (TTS endpoint)    |   | (via NVIDIA driver) |  | |
|  |  +-------------------+   +--------------------+  | |
|  +--------------------------------------------------+ |
|          |                                             |
|  +-------------------+   +------------------------+   |
|  | HTTP Client       |   | Gradio Demo (optional) |   |
|  | (Python/httpx)    |   | (port 7860)            |   |
|  +-------------------+   +------------------------+   |
+-------------------------------------------------------+
```

### System Requirements

| Component | Requirement |
|---|---|
| OS | Windows 11 with WSL2 enabled |
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| NVIDIA Driver | >= 535.xx (CUDA 12.8+ compatible) |
| WSL Distro | Ubuntu 22.04 or 24.04 |
| Python | 3.12 (managed via `uv`) |
| Disk Space | ~10GB for model weights + ~5GB for dependencies |
| RAM | >= 16GB system RAM recommended |

### Model Specifications

```javascript
{
  model_type: "voxtral_tts",
  base_model: "mistralai/Ministral-3-3B-Base-2512",
  dim: 3072,                          // Model dimension
  n_layers: 26,                       // Transformer layers
  n_heads: 32,                        // Attention heads
  n_kv_heads: 8,                      // Key-value heads
  head_dim: 128,                      // Head dimension
  hidden_dim: 9216,                   // FFN hidden dimension
  vocab_size: 131072,                 // Token vocabulary size
  max_seq_len: 65536,                 // Maximum sequence length
  max_position_embeddings: 128000,    // Maximum position embeddings
  weights_format: "BF16",             // Brain Float 16
  sampling_rate: 24000,               // 24 kHz output audio
  frame_rate: 12.5,                   // Audio frame rate
  num_codebooks: 37,                  // Total audio codebooks
  semantic_codebook_size: 8192,       // Semantic codebook entries
  acoustic_codebook_size: 21,         // Acoustic codebook entries
  n_acoustic_codebook: 36,            // Number of acoustic codebooks
  preset_voices: 20                   // Number of built-in voices
}
```

---

## Step-by-Step Installation

### Step 1: Verify Windows Prerequisites

Confirm WSL2 is enabled and an NVIDIA GPU driver supporting CUDA 12.8+ is installed. Run these commands from **Windows PowerShell (Admin)**:

```powershell
# Check WSL version
wsl --version

# If WSL is not installed or needs updating:
wsl --install
wsl --update

# Verify NVIDIA driver is visible from Windows
nvidia-smi
```

The `nvidia-smi` output must show **Driver Version >= 535.xx** and **CUDA Version >= 12.8**. The NVIDIA GPU driver installed on the Windows host automatically provides CUDA support inside WSL2 -- do NOT install a separate CUDA driver inside WSL.

### Step 2: Prepare WSL2 Ubuntu Environment

Launch your WSL2 Ubuntu terminal. All remaining commands run inside WSL2.

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools and git
sudo apt install -y build-essential git curl wget

# Verify GPU is accessible from WSL2
nvidia-smi
```

Expected: `nvidia-smi` shows the RTX 4090 with CUDA version >= 12.8.

### Step 3: Install uv (Python Environment Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (reload shell or source profile)
source $HOME/.cargo/env

# Verify installation
uv --version
```

### Step 4: Create Python 3.12 Virtual Environment

```bash
# Navigate to the project directory (adjust path to your WSL mount)
cd /mnt/k/voxtral-mini-4b

# Create a Python 3.12 virtual environment
uv venv --python 3.12 --seed --managed-python

# Activate the environment
source .venv/bin/activate

# Verify Python version
python --version  # Should print Python 3.12.x
```

### Step 5: Install vLLM >= 0.18.0

```bash
# Install vLLM with automatic CUDA backend detection
uv pip install vllm --torch-backend=auto

# Verify vLLM version
python -c "import vllm; print(vllm.__version__)"  # Must print >= 0.18.0
```

**Troubleshooting:** If you encounter CUDA compatibility issues:

```bash
# Force a specific CUDA version (e.g., 12.8)
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=128
export CPU_ARCH=$(uname -m)
uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl" --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
```

### Step 6: Install vllm-omni

```bash
# Install vllm-omni directly from the GitHub main branch
uv pip install "git+https://github.com/vllm-project/vllm-omni.git" --upgrade
```

### Step 7: Verify mistral_common

Installing vLLM >= 0.18.0 should automatically install `mistral_common >= 1.10.0`:

```bash
python -c "import mistral_common; print(mistral_common.__version__)"
# Must print >= 1.10.0
```

If the version is too old:

```bash
uv pip install -U mistral_common
```

### Step 8: Install Client Dependencies

```bash
# For making TTS requests and handling audio
uv pip install httpx soundfile

# Optional: for audio playback during testing
uv pip install sounddevice
```

---

## Serving the Model

### Start the vLLM Server

The model weights are available locally. Start the server pointing to the local path or the HuggingFace model ID:

```bash
# Option A: Serve from HuggingFace model ID (will use cached/local weights if available)
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni

# Option B: Serve from local path (if you want to explicitly use the local clone)
vllm serve ./Voxtral-4B-TTS-2603 --omni
```

The `--omni` flag is **required** -- it enables the vllm-omni plugin which provides the `/v1/audio/speech` TTS endpoint.

**Expected startup output:** The server will load model weights (~8GB in BF16), initialize CUDA kernels, and begin listening on `http://0.0.0.0:8000`. The RTX 4090 with 24GB VRAM has ample headroom for this 4B parameter model.

### Server Health Check

```bash
# From another terminal (with the venv activated)
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

---

## Client Usage

### Basic TTS Request (Python)

```python
import io
import httpx
import soundfile as sf

BASE_URL = "http://localhost:8000/v1"

payload = {
    "input": "Paris is a beautiful city!",
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "response_format": "wav",
    "voice": "casual_male",
}

response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=120.0)
response.raise_for_status()

audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
print(f"Got audio: {len(audio_array)} samples at {sr} Hz")

# Save to file
sf.write("output.wav", audio_array, sr)
```

### Multilingual Example

```python
import io
import httpx
import soundfile as sf

BASE_URL = "http://localhost:8000/v1"

samples = [
    {"input": "Hello, how are you today?", "voice": "neutral_female"},
    {"input": "Bonjour, comment allez-vous?", "voice": "fr_female"},
    {"input": "Hola, como estas hoy?", "voice": "es_male"},
    {"input": "Guten Tag, wie geht es Ihnen?", "voice": "de_male"},
    {"input": "Ciao, come stai oggi?", "voice": "it_female"},
]

for i, sample in enumerate(samples):
    payload = {
        "input": sample["input"],
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "response_format": "wav",
        "voice": sample["voice"],
    }
    response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=120.0)
    response.raise_for_status()
    audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    sf.write(f"sample_{i}_{sample['voice']}.wav", audio_array, sr)
    print(f"[{sample['voice']}] {len(audio_array)} samples at {sr} Hz -> sample_{i}_{sample['voice']}.wav")
```

### Audio Playback (Optional)

```python
import sounddevice as sd

# After generating audio_array and sr from the examples above:
sd.play(audio_array, sr)
sd.wait()  # Block until playback finishes
```

---

## Optional: Gradio Demo UI

### Install and Launch

```bash
# Clone vllm-omni repo (if not already cloned)
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni

# Install Gradio
uv pip install gradio==5.50

# Launch the demo (server must already be running on port 8000)
python examples/online_serving/voxtral_tts/gradio_demo.py \
  --host localhost \
  --port 8000
```

The Gradio interface will be available at `http://localhost:7860` and provides a web UI for selecting voices, entering text, and playing generated audio interactively.

---

## Verification & Testing

### Verification Checklist

Run these commands sequentially after installation to confirm everything works:

```bash
# 1. GPU accessible
nvidia-smi | grep "RTX 4090"

# 2. Python environment
python --version  # 3.12.x

# 3. vLLM installed
python -c "import vllm; print(f'vLLM {vllm.__version__}')"  # >= 0.18.0

# 4. vllm-omni installed
python -c "import vllm_omni; print('vllm-omni OK')"

# 5. mistral_common version
python -c "import mistral_common; print(f'mistral_common {mistral_common.__version__}')"  # >= 1.10.0

# 6. Client libraries
python -c "import httpx, soundfile; print('Client deps OK')"

# 7. Model weights present
ls -la ./Voxtral-4B-TTS-2603/consolidated.safetensors
ls -la ./Voxtral-4B-TTS-2603/params.json
ls -la ./Voxtral-4B-TTS-2603/tekken.json
ls ./Voxtral-4B-TTS-2603/voice_embedding/  # Should list 20 .pt files
```

### End-to-End Smoke Test

With the server running, execute this minimal test:

```bash
python -c "
import io, httpx, soundfile as sf
r = httpx.post('http://localhost:8000/v1/audio/speech', json={
    'input': 'Testing Voxtral TTS.',
    'model': 'mistralai/Voxtral-4B-TTS-2603',
    'response_format': 'wav',
    'voice': 'neutral_male',
}, timeout=120.0)
r.raise_for_status()
a, sr = sf.read(io.BytesIO(r.content), dtype='float32')
print(f'SUCCESS: {len(a)} samples at {sr} Hz ({len(a)/sr:.2f}s audio)')
sf.write('smoke_test.wav', a, sr)
"
```

Expected output: `SUCCESS: <N> samples at 24000 Hz (<duration>s audio)` and a playable `smoke_test.wav` file.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `nvidia-smi` not found in WSL | NVIDIA driver not installed on Windows host or WSL not updated | Install latest NVIDIA Game Ready / Studio driver on Windows; run `wsl --update` |
| CUDA version mismatch | vLLM compiled against different CUDA than available | Use the explicit CUDA version wheel install from Step 5 troubleshooting |
| `git` not found during vllm-omni install | Git not installed in WSL | Run `sudo apt install -y git` |
| `ModuleNotFoundError: vllm_omni` | vllm-omni not installed | Run `uv pip install git+https://github.com/vllm-project/vllm-omni.git --upgrade` |
| Server OOM on startup | Insufficient GPU memory | RTX 4090 (24GB) should be sufficient; ensure no other GPU processes are running (`nvidia-smi` to check) |
| Slow first request | CUDA kernel compilation on first inference | Normal behavior; subsequent requests will be fast |
| `Connection refused` on client | Server not fully started | Wait for server log to show "Uvicorn running on http://0.0.0.0:8000" |
| PyTorch NCCL errors | Conda-installed PyTorch with static NCCL | Use the uv-managed environment as described; do not mix conda |

---

## Performance Expectations

Benchmarked on NVIDIA H200 (official results from model card). RTX 4090 performance will vary but should be comparable for single-request latency:

| Concurrency | Latency | RTF (lower=better) | Throughput (char/s/GPU) |
|:-----------:|:-------:|:-------------------:|:-----------------------:|
| 1 | 70 ms | 0.103 | 119.14 |
| 16 | 331 ms | 0.237 | 879.11 |
| 32 | 552 ms | 0.302 | 1430.78 |

*RTF = Real-Time Factor. Input: 500-character text with 10-second audio reference. vLLM v0.18.0.*

---

## File Manifest

```
voxtral-mini-4b/
+-- Voxtral-4B-TTS-2603/           # Model weights (pre-cloned from HuggingFace)
|   +-- consolidated.safetensors    # BF16 model weights
|   +-- params.json                 # Model architecture parameters
|   +-- tekken.json                 # Tokenizer configuration
|   +-- README.md                   # Original HuggingFace model card
|   +-- voice_embedding/            # 20 preset voice embeddings
|       +-- casual_male.pt
|       +-- casual_female.pt
|       +-- cheerful_female.pt
|       +-- neutral_male.pt
|       +-- neutral_female.pt
|       +-- fr_male.pt, fr_female.pt
|       +-- es_male.pt, es_female.pt
|       +-- de_male.pt, de_female.pt
|       +-- it_male.pt, it_female.pt
|       +-- pt_male.pt, pt_female.pt
|       +-- nl_male.pt, nl_female.pt
|       +-- ar_male.pt
|       +-- hi_male.pt, hi_female.pt
+-- model-card.md                   # Source model card documentation
+-- VLLM-GPU-install.md            # Source vLLM GPU installation docs
+-- VoxtralTTS-2603-TINS.md        # This file
```
