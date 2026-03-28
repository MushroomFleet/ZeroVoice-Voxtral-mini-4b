<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:WINDOWS+WSL2 -->
<!-- ZS:LANGUAGE:PYTHON,BASH -->

# Voxtral-4B-TTS-2603 Installation & Deployment (Post-Install Corrected)

## Description

Complete, battle-tested installation guide for **Mistral AI's Voxtral-4B-TTS-2603** text-to-speech model on a Windows 11 system with NVIDIA RTX 4090 GPU. This document supersedes the original `VoxtralTTS-2603-TINS.md` and reflects the exact steps, workarounds, and patches that produced a working deployment on 2026-03-27.

Voxtral TTS is a frontier, open-weights text-to-speech model producing lifelike speech across 9 languages with 20 preset voices, 24 kHz audio output, and very low latency. The inference stack runs inside **WSL2 (Ubuntu 22.04)** because vLLM does not support Windows natively. The model weights are pre-cloned from HuggingFace and reside at `./Voxtral-4B-TTS-2603/` on the Windows filesystem, accessed via `/mnt/` inside WSL2.

**Target audience:** Developers and researchers evaluating Voxtral TTS for inference pipeline testing on a local RTX 4090 workstation.

**License:** CC BY-NC 4.0 (inherited from bundled voice reference datasets).

---

## Critical Notes Before You Begin

These are hard-won lessons from the actual installation. Read all of them before starting.

1. **The Python venv MUST live on the native Linux filesystem** (e.g., `~/voxtral-tts/.venv`), NOT on a `/mnt/` NTFS mount. PyTorch fails with `ModuleNotFoundError: No module named 'torch._utils_internal'` when the venv is on NTFS via WSL2. Model weights on `/mnt/` are fine.

2. **PyTorch version must be pinned** after installing vllm-omni. vllm-omni upgrades torch to a newer version that breaks vLLM's compiled `_C.abi3.so` with an ABI mismatch (`undefined symbol`).

3. **`--enforce-eager` is required** when launching the server. The Voxtral TTS model does not support `torch.compile` and the inductor/triton compilation crashes during memory profiling without this flag.

4. **cuDNN must be disabled** if your NVIDIA driver is < 570.x. Driver 566.36 is incompatible with cuDNN 9.19 bundled in torch 2.10.0+cu129, causing `CUDNN_STATUS_NOT_INITIALIZED` on Conv1d operations.

5. **A C compiler (gcc) is required** for triton kernel compilation during vLLM memory profiling. If `sudo apt install build-essential` is not possible, gcc can be installed via micromamba without root.

6. **A source code patch is required** for vllm-omni's `VoxtralTTSConfig` class to work with transformers 5.x. The `text_config` attribute must be set before `super().__init__()`.

---

## Functionality

### Core Capabilities

- **Text-to-Speech inference** via an OpenAI-compatible HTTP API (`POST /v1/audio/speech`)
- **Voice listing** via `GET /v1/audio/voices`
- **20 preset voices** across 9 language categories (no fine-tuning required)
- **9 languages supported:** English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi
- **24 kHz audio output** in WAV format (MP3/AAC/Opus require ffmpeg installed separately)
- **BF16 weights** running on a single GPU with >= 16GB VRAM (uses ~22.5 GB on RTX 4090 with KV cache)
- **Gradio web demo** with cascading language/voice dropdowns and audio playback

### Available Voices

| Language | Voices |
|---|---|
| English | `neutral_male`, `neutral_female`, `casual_male`, `casual_female`, `cheerful_female` |
| French | `fr_male`, `fr_female` |
| Spanish | `es_male`, `es_female` |
| German | `de_male`, `de_female` |
| Italian | `it_male`, `it_female` |
| Portuguese | `pt_male`, `pt_female` |
| Dutch | `nl_male`, `nl_female` |
| Arabic | `ar_male` |
| Hindi | `hi_male`, `hi_female` |

### User Flow

```
[1] Enable BIOS virtualization + WSL2
         |
[2] Install uv + create Python 3.12 venv (on native Linux FS)
         |
[3] Install vLLM 0.18.0
         |
[4] Install vllm-omni + pin PyTorch back
         |
[5] Apply source patches (config init order + cuDNN disable)
         |
[6] Install gcc (via apt or micromamba)
         |
[7] Start server with --omni --enforce-eager
         |
[8] Send TTS requests via HTTP client
         |
[9] (Optional) Launch Gradio demo UI
```

---

## Technical Implementation

### Architecture Overview

```
+-------------------------------------------------------+
|  Windows 11 Host (RTX 4090, Driver >= 535.xx)         |
|                                                       |
|  +--------------------------------------------------+ |
|  |  WSL2 (Ubuntu 22.04)                             | |
|  |  Venv: ~/voxtral-tts/.venv (ext4 filesystem)     | |
|  |                                                   | |
|  |  +-------------------+   +---------------------+  | |
|  |  | vLLM 0.18.0       |   | Model Weights       |  | |
|  |  | --omni             |   | /mnt/<drive>/        |  | |
|  |  | --enforce-eager    |   |   Voxtral-4B-TTS/   |  | |
|  |  | port 8000         |   | (NTFS mount, OK)     |  | |
|  |  +-------------------+   +---------------------+  | |
|  |          |                                         | |
|  |  +-------------------+   +---------------------+  | |
|  |  | vllm-omni plugin  |   | torch 2.10.0+cu129  |  | |
|  |  | (TTS endpoint)    |   | (pinned, cuDNN off)  |  | |
|  |  +-------------------+   +---------------------+  | |
|  +--------------------------------------------------+ |
|          |                                             |
|  +-------------------+   +------------------------+   |
|  | HTTP Client       |   | Gradio Demo            |   |
|  | (any OS, httpx)   |   | port 7860 + share URL  |   |
|  +-------------------+   +------------------------+   |
+-------------------------------------------------------+
```

### System Requirements

| Component | Requirement | Verified Value |
|---|---|---|
| OS | Windows 11 with WSL2 | Windows 11 Home 10.0.22631 |
| BIOS | Virtualization enabled (VT-x / AMD-V) | Required for WSL2 |
| GPU | NVIDIA with >= 16GB VRAM, compute capability >= 7.0 | RTX 4090, 24GB |
| NVIDIA Driver | >= 535.xx | 566.36 (works with cuDNN workaround) |
| WSL Distro | Ubuntu 22.04 or 24.04 | Ubuntu 22.04 |
| Python | 3.12 (managed via `uv`) | 3.12.13 |
| Disk (Linux FS) | ~15GB for venv + dependencies | ~/voxtral-tts/ |
| Disk (Windows FS) | ~10GB for model weights | Voxtral-4B-TTS-2603/ |
| RAM | >= 32GB recommended | Model + KV cache uses ~22.5GB VRAM |

### Verified Software Versions

| Package | Version | Notes |
|---|---|---|
| uv | 0.11.2 | Python environment manager |
| vLLM | 0.18.0 | Must be this exact version |
| vllm-omni | 0.18.0rc2 | From GitHub main branch |
| torch | 2.10.0+cu129 | MUST be pinned after vllm-omni install |
| torchaudio | 2.10.0+cu129 | Pinned with torch |
| torchvision | 0.25.0+cu129 | Pinned with torch |
| transformers | 5.4.0 | Requires config patch (see Step 8) |
| mistral_common | 1.10.0 | Installed automatically with vLLM |
| httpx | 0.28.1 | Client HTTP library |
| soundfile | 0.13.1 | Audio file I/O |
| gradio | 5.50.0 | Installed automatically with vllm-omni |
| gcc | 15.2.0 | Via apt or micromamba conda-forge |

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

### Step 1: Enable Virtualization and WSL2

WSL2 requires hardware virtualization. If not already enabled:

1. **Enter BIOS/UEFI** (restart, press DEL or F2 during POST).
2. **Enable** Intel VT-x (or AMD-V / SVM Mode).
3. **Save and reboot** into Windows.

Then from **Windows PowerShell (Admin)**:

```powershell
# Enable WSL2 and Virtual Machine Platform
wsl --install
wsl --update

# If Ubuntu is not yet installed:
wsl --install -d Ubuntu-22.04

# Verify
wsl --list --verbose
# Should show Ubuntu-22.04 with VERSION 2
```

Verify GPU is accessible from WSL2 (run inside a WSL2 terminal):

```bash
nvidia-smi
# Must show RTX 4090 with CUDA version >= 12.x
```

> **If `nvidia-smi` fails inside WSL:** Ensure the Windows NVIDIA driver is >= 535.xx. Do NOT install a separate CUDA driver inside WSL2 -- the Windows host driver provides CUDA support automatically.

### Step 2: Install uv (Python Environment Manager)

Inside WSL2:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version  # Should print >= 0.10.x
```

### Step 3: Create Python 3.12 Virtual Environment

**CRITICAL:** The venv MUST be on the native Linux ext4 filesystem, NOT on `/mnt/`. PyTorch's internal modules fail to load from NTFS mounts.

```bash
# Create project directory on native Linux filesystem
mkdir -p ~/voxtral-tts
cd ~/voxtral-tts

# Create Python 3.12 venv with uv-managed Python
uv venv --python 3.12 --seed --managed-python .venv

# Activate
source .venv/bin/activate
python --version  # Should print Python 3.12.x
```

### Step 4: Install vLLM 0.18.0

```bash
uv pip install vllm --torch-backend=auto

# Verify
python -c "import vllm; print(vllm.__version__)"  # Must print 0.18.0
```

Note the torch version installed -- you will need it in the next step:

```bash
python -c "import torch; print(torch.__version__)"
# Record this value (e.g., 2.10.0+cu129)
```

### Step 5: Install vllm-omni and Pin PyTorch

vllm-omni's dependencies will upgrade PyTorch to an incompatible version. You must pin it back immediately after.

```bash
# Install vllm-omni from GitHub main branch
uv pip install "git+https://github.com/vllm-project/vllm-omni.git" --upgrade

# IMMEDIATELY pin PyTorch back to the vLLM-compatible version
uv pip install "torch==2.10.0+cu129" "torchaudio==2.10.0+cu129" "torchvision==0.25.0+cu129" --torch-backend=cu129
```

Verify the pin worked:

```bash
python -c "import torch; print(torch.__version__)"   # Must print 2.10.0+cu129
python -c "import vllm; print(vllm.__version__)"     # Must print 0.18.0
```

### Step 6: Install Client Dependencies

```bash
uv pip install httpx sounddevice
```

### Step 7: Verify mistral_common

```bash
python -c "import mistral_common; print(mistral_common.__version__)"
# Must print >= 1.10.0
```

If too old: `uv pip install -U mistral_common`

### Step 8: Patch vllm-omni VoxtralTTSConfig (transformers 5.x fix)

Transformers 5.x calls `get_text_config()` during `PretrainedConfig.__init__()` validation, before the subclass assigns `self.text_config`. This crashes the server with `'VoxtralTTSConfig' object has no attribute 'text_config'`.

Apply this patch:

```bash
python << 'PYEOF'
import site, os

site_packages = site.getsitepackages()[0]
fpath = os.path.join(site_packages, "vllm_omni", "model_executor", "models", "voxtral_tts", "configuration_voxtral_tts.py")

with open(fpath, "r") as f:
    content = f.read()

old = """    ) -> None:
        super().__init__(**kwargs)

        if isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            self.text_config = PretrainedConfig.from_dict(text_config)
        else:
            self.text_config = PretrainedConfig()

        self.audio_config = audio_config or {}"""

new = """    ) -> None:
        # Set text_config BEFORE super().__init__ because transformers 5.x
        # calls get_text_config() during validation inside __init__.
        if isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            self.text_config = PretrainedConfig.from_dict(text_config)
        else:
            self.text_config = PretrainedConfig()

        self.audio_config = audio_config or {}

        super().__init__(**kwargs)"""

assert old in content, "Patch target not found -- file may already be patched or vllm-omni version changed"
content = content.replace(old, new)

with open(fpath, "w") as f:
    f.write(content)

print(f"Patched: {fpath}")
PYEOF
```

### Step 9: Patch vLLM cuDNN Disable (driver < 570.x)

cuDNN 9.19 bundled with torch 2.10.0+cu129 is incompatible with NVIDIA drivers < 570.x under WSL2, causing `CUDNN_STATUS_NOT_INITIALIZED` on Conv1d. This patch disables cuDNN in the EngineCore subprocess.

> **Skip this step** if your NVIDIA driver is >= 570.x.

```bash
python << 'PYEOF'
import site, os

site_packages = site.getsitepackages()[0]
fpath = os.path.join(site_packages, "vllm", "platforms", "cuda.py")

with open(fpath, "r") as f:
    content = f.read()

old = "import torch\nfrom torch.distributed import PrefixStore, ProcessGroup"
new = "import torch\ntorch.backends.cudnn.enabled = False  # Workaround: cuDNN incompatible with driver < 570.x on WSL2\nfrom torch.distributed import PrefixStore, ProcessGroup"

if "cudnn.enabled = False" in content:
    print("Already patched")
else:
    assert old in content, "Patch target not found"
    content = content.replace(old, new, 1)
    with open(fpath, "w") as f:
        f.write(content)
    print(f"Patched: {fpath}")
PYEOF
```

### Step 10: Install gcc (C Compiler)

Triton requires a C compiler for kernel compilation during vLLM's memory profiling phase.

**Option A: With sudo access (preferred)**

```bash
sudo apt update && sudo apt install -y build-essential
```

**Option B: Without sudo (via micromamba)**

```bash
# Install micromamba
mkdir -p /tmp/mamba && cd /tmp/mamba
curl -L -o micromamba.tar.bz2 "https://micro.mamba.pm/api/micromamba/linux-64/latest"
python3 -c "
import bz2
with open('micromamba.tar.bz2', 'rb') as f:
    data = bz2.decompress(f.read())
with open('micromamba.tar', 'wb') as f:
    f.write(data)
"
tar -xf micromamba.tar
cp bin/micromamba ~/.local/bin/
chmod +x ~/.local/bin/micromamba

# Install gcc from conda-forge
export MAMBA_ROOT_PREFIX=~/.micromamba
~/.local/bin/micromamba create -n gcc -c conda-forge gcc_linux-64 gxx_linux-64 -y
```

### Step 11: Install ffmpeg (Optional)

Required only for MP3, AAC, and Opus output formats. WAV output works without it.

```bash
sudo apt install -y ffmpeg   # Requires sudo
```

---

## Serving the Model

### Create a Launch Script

The server requires several environment variables for the gcc PATH, cuDNN workaround, and unbuffered output. Use this launch script:

```bash
cat > ~/voxtral-tts/start_server.sh << 'EOF'
#!/bin/bash
# Voxtral TTS vLLM Server Launcher
source $HOME/.local/bin/env
cd $HOME/voxtral-tts
source .venv/bin/activate

# Add gcc to PATH (choose one based on your install method)
# Option A (apt): gcc is already in PATH
# Option B (micromamba): uncomment the following lines
# export PATH="$HOME/.micromamba/envs/gcc/bin:$PATH"
# export CC="$HOME/.micromamba/envs/gcc/bin/x86_64-conda-linux-gnu-gcc"
# export CXX="$HOME/.micromamba/envs/gcc/bin/x86_64-conda-linux-gnu-g++"

export PYTHONUNBUFFERED=1

echo "=== Starting Voxtral TTS Server ==="
echo "Python: $(python --version)"
echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "==================================="

# --omni: enables vllm-omni TTS plugin
# --enforce-eager: required, model does not support torch.compile
exec vllm serve /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603 --omni --enforce-eager
EOF

chmod +x ~/voxtral-tts/start_server.sh
```

> **Adjust the model path** `/mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603` to match your Windows drive letter and directory.

### Start the Server

From a **Windows terminal** (Git Bash, PowerShell, or CMD):

```bash
wsl -d Ubuntu-22.04 -- bash ~/voxtral-tts/start_server.sh
```

Or from inside a WSL2 terminal directly:

```bash
~/voxtral-tts/start_server.sh
```

**Expected startup sequence:**
1. vLLM logo banner and version info
2. `Initializing a V1 LLM engine (v0.18.0)`
3. `Available voice embeddings: ['casual_female', 'casual_male', ...]` (20 voices)
4. `Loading safetensors checkpoint shards: 100%` (~3-4 seconds)
5. `Model loading took 7.78 GiB memory`
6. CUDA graph captures for acoustic transformer (6 batch sizes)
7. KV cache allocation
8. `Uvicorn running on http://0.0.0.0:8000`

Total startup time: **3-5 minutes** (longer on first run due to triton compilation).

GPU memory usage when fully loaded: **~22.5 GB / 24 GB**.

### Server Health Check

```bash
curl http://localhost:8000/health
# Returns HTTP 200 with empty body when ready

curl http://localhost:8000/v1/audio/voices
# Returns JSON: {"voices":["ar_male","casual_female",...], "uploaded_voices":[]}
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
    "model": "/mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603",
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

> **Note:** The `model` field must match the path used when starting the server (the local path, not the HuggingFace ID).

### Multilingual Example

```python
import io
import httpx
import soundfile as sf

BASE_URL = "http://localhost:8000/v1"
MODEL = "/mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603"

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
        "model": MODEL,
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

## Gradio Demo UI

### Launch

The vllm-omni repo contains a Gradio frontend. Gradio 5.50.0 is already installed as a vllm-omni dependency.

```bash
cd ~/voxtral-tts
source .venv/bin/activate

# Clone the demo repo (shallow, only need the example scripts)
git clone --depth 1 https://github.com/vllm-project/vllm-omni.git ~/vllm-omni

# Launch (server must already be running on port 8000)
cd ~/vllm-omni
python examples/online_serving/voxtral_tts/gradio_demo.py \
  --host localhost \
  --port 8000 \
  --model /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603
```

### Access

- **Local:** http://localhost:7860
- **Public share link:** Gradio auto-generates a `*.gradio.live` URL (expires in 1 week)

### UI Layout

```
+---------------------------------------------------+
|  Voxtral TTS                                      |
+---------------------------------------------------+
| [Language v]     |  Generated audio                |
| [Voice v]        |  [  ▶  audio player  ⬇ ]       |
|                  |                                 |
| Text prompt:     |  Shareable link:                |
| [                |  [  copy link  ]                |
|   Enter text...  |                                 |
|                ] |                                 |
|                  |                                 |
| [Clear] [Generate audio]                          |
+---------------------------------------------------+
```

The **Language** dropdown selects the language category (English, French, Spanish, etc.). The **Voice** dropdown updates to show only voices for the selected language. All 20 voices are available across 9 language categories.

---

## Verification & Testing

### Verification Checklist

Run inside WSL2 with the venv activated (`source ~/voxtral-tts/.venv/bin/activate`):

```bash
# 1. GPU accessible
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2. Python version
python --version  # 3.12.x

# 3. vLLM version
python -c "import vllm; print(vllm.__version__)"  # 0.18.0

# 4. vllm-omni import
python -c "import vllm_omni; print('vllm-omni OK')"

# 5. torch version (must be pinned)
python -c "import torch; print(torch.__version__)"  # 2.10.0+cu129

# 6. mistral_common
python -c "import mistral_common; print(mistral_common.__version__)"  # >= 1.10.0

# 7. Client libraries
python -c "import httpx, soundfile; print('Client deps OK')"

# 8. Model weights (adjust path to your mount)
ls -lh /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603/consolidated.safetensors
ls /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603/voice_embedding/ | wc -l  # Should print 20
```

### End-to-End Smoke Test

With the server running:

```bash
python -c "
import io, httpx, soundfile as sf
r = httpx.post('http://localhost:8000/v1/audio/speech', json={
    'input': 'Testing Voxtral TTS. Hello world!',
    'model': '/mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603',
    'response_format': 'wav',
    'voice': 'neutral_male',
}, timeout=120.0)
r.raise_for_status()
a, sr = sf.read(io.BytesIO(r.content), dtype='float32')
print(f'SUCCESS: {len(a)} samples at {sr} Hz ({len(a)/sr:.2f}s audio)')
sf.write('smoke_test.wav', a, sr)
"
```

Expected: `SUCCESS: 82560 samples at 24000 Hz (3.44s audio)` and a playable `smoke_test.wav`.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `WSL2 is not supported with your current machine configuration` | Virtualization disabled in BIOS | Enter BIOS, enable VT-x / AMD-V / SVM Mode, reboot |
| `ModuleNotFoundError: No module named 'torch._utils_internal'` | Venv is on NTFS mount (`/mnt/`) | Recreate venv on native Linux filesystem (`~/voxtral-tts/.venv`) |
| `undefined symbol: _ZN3c1013MessageLoggerC1EPKciib` | PyTorch version mismatch after vllm-omni install | Pin torch: `uv pip install "torch==2.10.0+cu129" --torch-backend=cu129` |
| `'VoxtralTTSConfig' object has no attribute 'text_config'` | transformers 5.x init order bug | Apply the Step 8 patch (move `self.text_config` before `super().__init__`) |
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` | NVIDIA driver < 570.x + cuDNN 9.19 | Apply the Step 9 patch (disable cuDNN in `vllm/platforms/cuda.py`) |
| `Failed to find C compiler` / triton compilation error | gcc not installed | Install via `sudo apt install build-essential` or micromamba (Step 10) |
| `torch.compile is turned on, but the model does not support it` (crash) | Inductor compilation fails on Voxtral TTS | Use `--enforce-eager` flag when launching server |
| `flash_attn is not installed. Falling back to PyTorch SDPA` | flash-attn not installed | Warning only. Optional: `uv pip install flash-attn` (requires build tools) |
| `Couldn't find ffmpeg or avconv` | ffmpeg not installed | Warning only. Only needed for MP3/AAC/Opus output: `sudo apt install ffmpeg` |
| `nvidia-smi` not found in WSL | Driver issue | Install latest NVIDIA driver on Windows host; run `wsl --update` |
| Server OOM on startup | Other GPU processes consuming VRAM | Close other GPU apps; model needs ~22.5GB. Check with `nvidia-smi` |
| `Connection refused` on client | Server not fully started | Wait for log to show `Uvicorn running on http://0.0.0.0:8000` (~3-5 min) |
| WSL background process dies | WSL kills processes when shell exits | Keep the WSL terminal open, or run via `wsl -- bash ~/voxtral-tts/start_server.sh` from Windows |

---

## Performance Expectations

Benchmarked on NVIDIA H200 (official results from model card). RTX 4090 results will differ due to lower memory bandwidth and the `--enforce-eager` flag.

| Concurrency | Latency (H200) | RTF (lower=better) | Throughput (char/s/GPU) |
|:-----------:|:---------------:|:-------------------:|:-----------------------:|
| 1 | 70 ms | 0.103 | 119.14 |
| 16 | 331 ms | 0.237 | 879.11 |
| 32 | 552 ms | 0.302 | 1430.78 |

*RTF = Real-Time Factor. Input: 500-character text with 10-second audio reference. vLLM v0.18.0.*

**Observed RTX 4090 results** (with `--enforce-eager`, cuDNN disabled):

| Voice | Input Text | Audio Duration | Status |
|---|---|---|---|
| neutral_male | "Testing Voxtral TTS. Hello world!" | 3.44s | PASS |
| casual_female | "Hello, how are you today?" | 3.84s | PASS |
| fr_male | "Bonjour, comment allez-vous?" | 2.08s | PASS |
| es_female | "Hola, como estas?" | 1.84s | PASS |

---

## File Manifest

```
Windows filesystem (e.g., K:\voxtral-mini-4b\):
+-- Voxtral-4B-TTS-2603/           # Model weights (pre-cloned from HuggingFace)
|   +-- consolidated.safetensors    # BF16 model weights (7.5 GB)
|   +-- params.json                 # Model architecture parameters
|   +-- tekken.json                 # Tokenizer configuration (15 MB)
|   +-- README.md                   # Original HuggingFace model card
|   +-- voice_embedding/            # 20 preset voice embeddings (.pt files)
+-- model-card.md                   # Source model card documentation
+-- VLLM-GPU-install.md            # Source vLLM GPU installation docs
+-- VoxtralTTS-2603-TINS.md        # Original pre-install plan
+-- post-install-TINS.md           # This file (corrected plan)
+-- stage1-incomplete-steps.md     # Installation assessment log
+-- start_server.sh                # Server launch script
+-- smoke_test.wav                 # Generated test audio (24kHz WAV)
+-- test_casual_female.wav         # Test output
+-- test_fr_male.wav               # Test output
+-- test_es_female.wav             # Test output

WSL2 Linux filesystem:
~/voxtral-tts/
+-- .venv/                          # Python 3.12 virtual environment (MUST be here, not /mnt/)
+-- start_server.sh                 # Server launch script (copy)
~/vllm-omni/                        # Cloned repo (for Gradio demo)
+-- examples/online_serving/voxtral_tts/
    +-- gradio_demo.py              # Gradio frontend
    +-- text_preprocess.py          # Text sanitization
~/.local/bin/
+-- uv                             # Python environment manager
+-- micromamba                      # Conda alternative (if sudo unavailable)
~/.micromamba/envs/gcc/bin/         # gcc from conda-forge (if sudo unavailable)
```
