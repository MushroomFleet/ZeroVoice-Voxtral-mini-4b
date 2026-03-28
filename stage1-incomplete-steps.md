# Stage 1 Assessment — Voxtral-4B-TTS-2603 Installation

**Date:** 2026-03-27
**Assessed against:** `VoxtralTTS-2603-TINS.md`
**Status:** OPERATIONAL — Model serving and generating audio successfully

---

## Summary

| TINS Plan Step | Status | Notes |
|---|---|---|
| Step 1: Verify Windows Prerequisites | COMPLETE | GPU/driver OK; WSL2 functional after BIOS virtualization enabled |
| Step 2: Prepare WSL2 Ubuntu Environment | COMPLETE | Ubuntu-22.04 running; gcc installed via micromamba (no sudo) |
| Step 3: Install uv | COMPLETE | uv 0.11.2 in WSL2; uv 0.10.9 on Windows host |
| Step 4: Create Python 3.12 venv | COMPLETE | Python 3.12.13 venv at `~/voxtral-tts/.venv` (native Linux FS) |
| Step 5: Install vLLM >= 0.18.0 | COMPLETE | vLLM 0.18.0 with torch 2.10.0+cu129 |
| Step 6: Install vllm-omni | COMPLETE | vllm-omni 0.18.0rc2 from GitHub main |
| Step 7: Verify mistral_common | COMPLETE | mistral_common 1.10.0 |
| Step 8: Install client dependencies | COMPLETE | httpx 0.28.1, soundfile 0.13.1, sounddevice 0.5.5 |
| Serve the model | COMPLETE | Server running on port 8000 with `--omni --enforce-eager` |
| Client smoke test | COMPLETE | 3.44s audio generated at 24kHz, multi-voice/multi-language verified |
| Gradio demo | INCOMPLETE | Not attempted |

---

## Workarounds Applied During Installation

### Workaround 1: VoxtralTTSConfig `text_config` init order (transformers 5.x compatibility)

**File:** `vllm_omni/model_executor/models/voxtral_tts/configuration_voxtral_tts.py`

**Problem:** Transformers 5.4.0 calls `get_text_config()` during `PretrainedConfig.__init__()` validation, before the subclass has set `self.text_config`.

**Fix:** Move `self.text_config` and `self.audio_config` assignments BEFORE `super().__init__(**kwargs)`.

```python
# BEFORE (broken with transformers 5.x):
def __init__(self, text_config=None, audio_config=None, **kwargs):
    super().__init__(**kwargs)          # <-- calls get_text_config() here
    self.text_config = ...              # <-- too late

# AFTER (fixed):
def __init__(self, text_config=None, audio_config=None, **kwargs):
    self.text_config = ...              # <-- set first
    self.audio_config = audio_config or {}
    super().__init__(**kwargs)          # <-- now get_text_config() works
```

### Workaround 2: cuDNN disabled (driver 566.36 incompatible with cuDNN 9.19)

**File:** `vllm/platforms/cuda.py`

**Problem:** cuDNN 9.19 (bundled with torch 2.10.0+cu129) fails `CUDNN_STATUS_NOT_INITIALIZED` on Conv1d operations with NVIDIA driver 566.36 under WSL2.

**Fix:** Added `torch.backends.cudnn.enabled = False` after `import torch` in vLLM's CUDA platform module. This propagates to the EngineCore subprocess.

**Alternative resolution:** Update NVIDIA driver to >= 570.x, or downgrade to torch with cu126.

### Workaround 3: PyTorch version pinning (vllm-omni dependency conflict)

**Problem:** vllm-omni upgrades torch from 2.10.0+cu129 to 2.11.0+cu130, causing ABI mismatch with vLLM's compiled `_C.abi3.so` (`undefined symbol: _ZN3c1013MessageLoggerC1EPKciib`).

**Fix:** Pin torch back after vllm-omni install:
```bash
uv pip install "torch==2.10.0+cu129" "torchaudio==2.10.0+cu129" "torchvision==0.25.0+cu129" --torch-backend=cu129
```

### Workaround 4: `--enforce-eager` flag required

**Problem:** torch.compile/inductor crashes during memory profiling. The model doesn't support torch.compile (vLLM itself warns about this).

**Fix:** Always launch with `--enforce-eager`:
```bash
vllm serve /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603 --omni --enforce-eager
```

### Workaround 5: Venv must be on native Linux filesystem

**Problem:** PyTorch fails with `ModuleNotFoundError: No module named 'torch._utils_internal'` when the venv is on `/mnt/k/` (NTFS via WSL2 mount).

**Fix:** Create the venv on the native ext4 filesystem (e.g., `~/voxtral-tts/.venv`). The model weights on `/mnt/k/` are fine — only the Python venv is affected.

### Workaround 6: gcc installed via micromamba (no sudo available)

**Problem:** Triton/inductor requires a C compiler (`gcc`), but `sudo apt install build-essential` requires a password.

**Fix:** Installed gcc via micromamba (conda-forge) without sudo:
```bash
~/.local/bin/micromamba create -n gcc -c conda-forge gcc_linux-64 gxx_linux-64 -y
export CC=~/.micromamba/envs/gcc/bin/x86_64-conda-linux-gnu-gcc
```

---

## Incomplete Steps

### 1. Gradio Demo Not Launched

**TINS plan section:** "Optional: Gradio Demo UI"

**Status:** Not attempted. All prerequisites are met (vllm-omni includes gradio as a dependency, server is running). Would require:
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
python examples/online_serving/voxtral_tts/gradio_demo.py --host localhost --port 8000
```

### 2. sudo apt packages not installed (build-essential, libsndfile1-dev)

**Status:** Bypassed via micromamba for gcc. `libsndfile1` system library was not installed but `soundfile` Python package works without it (uses bundled shared library). User should run when password is available:
```bash
sudo apt install -y build-essential libsndfile1-dev ffmpeg
```

### 3. flash-attn not installed

**Status:** vLLM falls back to PyTorch SDPA. The server warns:
```
flash_attn is not installed. Falling back to PyTorch SDPA for audio tokenizer attention.
Install flash-attn for better performance.
```
Installing flash-attn requires compilation and build-essential. Optional performance improvement.

### 4. ffmpeg not installed

**Status:** pydub warns `Couldn't find ffmpeg or avconv`. Not needed for WAV output but required for MP3/AAC/Opus format output. Install via:
```bash
sudo apt install -y ffmpeg
```

### 5. TINS plan needs updating with discovered workarounds

**Status:** The `VoxtralTTS-2603-TINS.md` does not document the 6 workarounds above. Server launch command should be updated from `vllm serve ... --omni` to `vllm serve ... --omni --enforce-eager`, and the venv location guidance should specify native Linux FS only.

---

## Current System State

### WSL2 Environment (`~/voxtral-tts/.venv`)

| Component | Version | Status |
|---|---|---|
| WSL2 Ubuntu | 22.04 | Running |
| Python | 3.12.13 | Native Linux venv |
| vLLM | 0.18.0 | Operational (enforce-eager) |
| vllm-omni | 0.18.0rc2 | Operational (patched config) |
| torch | 2.10.0+cu129 | Pinned for ABI compat |
| mistral_common | 1.10.0 | OK |
| httpx | 0.28.1 | OK |
| soundfile | 0.13.1 | OK |
| gcc | 15.2.0 (conda-forge) | Via micromamba |

### Server

| Property | Value |
|---|---|
| Endpoint | `http://localhost:8000` |
| Health | `GET /health` returns 200 |
| TTS API | `POST /v1/audio/speech` |
| Model | `/mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603` |
| GPU Memory | ~22.5 GB / 24 GB |
| Flags | `--omni --enforce-eager` |
| Launch script | `K:\voxtral-mini-4b\start_server.sh` |

### Test Results

| Voice | Language | Input | Duration | Status |
|---|---|---|---|---|
| neutral_male | English | "Testing Voxtral TTS. Hello world!" | 3.44s | PASS |
| casual_female | English | "Hello, how are you today?" | 3.84s | PASS |
| fr_male | French | "Bonjour, comment allez-vous?" | 2.08s | PASS |
| es_female | Spanish | "Hola, como estas?" | 1.84s | PASS |

All output: 24kHz WAV format, saved to `K:\voxtral-mini-4b\`.
