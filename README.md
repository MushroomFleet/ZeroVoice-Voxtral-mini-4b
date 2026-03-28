# ZeroVoice for Voxtral-4B-TTS-2603

**Procedural voice generation for Mistral AI's Voxtral TTS model.** From 20 preset voices and zero stored bytes of additional voice data, ZeroVoice derives thousands of unique voices using position-is-seed coordinate hashing and spherical interpolation (SLERP).

Any 3D coordinate `(x, y, z)` deterministically produces a unique blended voice. The same coordinate always produces the same voice, on any machine, in any order. No database, no stored state -- the voice space springs complete from coordinates.

## How It Works

ZeroVoice applies the [ZeroBytes](https://github.com/MushroomFleet/ZeroBytes-Manifesto) position-is-seed procedural generation paradigm to text-to-speech voice synthesis:

1. **Z-axis** selects the voice family: z<100 = English, z=100-199 = European, z>=200 = Asian/Arabic
2. **Position hash** (xxHash64 of packed coordinates) selects a **cross-family voice pair** -- the primary voice (A) and a tint voice (B) are always from different language families
3. **Coherent noise** on the X/Y plane derives a smooth SLERP blend weight (capped at 0.20) so nearby coordinates sound similar
4. **Row-wise SLERP** blends the two voice embeddings on the 3072-dimensional hypersphere with magnitude preservation and norm calibration

The result is a voice that sounds like the primary preset with subtle characteristics from the secondary, producing natural-sounding variation without artifacts.

### Key Numbers

| Metric | Value |
|---|---|
| Base preset voices | 20 (9 languages, 10 male / 10 female) |
| Cross-family voice pairs | 156 (A and B always from different families) |
| Perceptually distinct voices per seed | ~3,276 |
| Total addressable voices (all seeds) | ~3.28 billion |
| Coordinate addresses (practical range) | 301 million+ |
| Additional voice data stored | 0 bytes |
| Audio output | 24 kHz WAV |

Full statistics and voice inventory: [ZeroVoice-stats.md](ZeroVoice-stats.md)

### ZeroBytes Law Compliance

| Law | Status |
|---|---|
| O(1) Access | Any voice computed directly from (x,y,z) -- no iteration |
| Parallelism | Each coordinate depends only on its own values, never neighbors |
| Coherence | Adjacent coordinates produce similar voices via multi-octave noise |
| Hierarchy | Z selects family, X/Y explore variation within |
| Determinism | Same inputs produce identical output on any machine |

## Installation

### Prerequisites

- **Windows 11** with WSL2 enabled (Ubuntu 22.04)
- **NVIDIA GPU** with >= 16GB VRAM (tested on RTX 4090, 24GB)
- **NVIDIA Driver** >= 535.xx
- **BIOS virtualization** enabled (VT-x / AMD-V) for WSL2

### Step 1: Clone This Repo

```bash
git clone https://github.com/MushroomFleet/ZeroVoice-Voxtral-mini-4b.git
cd ZeroVoice-Voxtral-mini-4b
```

### Step 2: Clone the Voxtral TTS Model

Download the model weights from HuggingFace (requires `git-lfs`):

```bash
git lfs install
git clone https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
```

This creates the `Voxtral-4B-TTS-2603/` directory containing model weights (~7.5GB), tokenizer, and 20 preset voice embeddings.

### Step 3: Set Up WSL2 Environment

All remaining commands run inside WSL2 Ubuntu. Open a WSL2 terminal:

```bash
# Install uv (Python environment manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create Python 3.12 venv on native Linux filesystem (NOT on /mnt/)
mkdir -p ~/voxtral-tts && cd ~/voxtral-tts
uv venv --python 3.12 --seed --managed-python .venv
source .venv/bin/activate
```

### Step 4: Install vLLM + vllm-omni

```bash
# Install vLLM 0.18.0
uv pip install vllm --torch-backend=auto

# Install vllm-omni (TTS plugin)
uv pip install "git+https://github.com/vllm-project/vllm-omni.git" --upgrade

# IMPORTANT: Pin PyTorch back (vllm-omni upgrades it to an incompatible version)
uv pip install "torch==2.10.0+cu129" "torchaudio==2.10.0+cu129" "torchvision==0.25.0+cu129" --torch-backend=cu129

# Install ZeroVoice dependencies
uv pip install xxhash httpx soundfile sounddevice
```

### Step 5: Install ZeroVoice Modules

Copy the ZeroVoice engine and SLERP library into the venv:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp /mnt/<drive>/ZeroVoice-Voxtral-mini-4b/slerp_voices.py "$SITE/"
cp /mnt/<drive>/ZeroVoice-Voxtral-mini-4b/zerovoice.py "$SITE/"
```

Replace `<drive>` with your Windows drive letter (e.g., `k` for `K:\`).

### Step 6: Apply Required Patches

The Voxtral TTS model requires three patches to work with ZeroVoice and on WSL2. See [post-install-TINS.md](post-install-TINS.md) for the full patch scripts. The patches address:

1. **VoxtralTTSConfig init order** -- transformers 5.x compatibility fix for vllm-omni
2. **cuDNN disable** -- workaround for NVIDIA driver < 570.x on WSL2
3. **ZeroVoice resolver** -- adds `zv_` prefix handling to the model's preprocessing
4. **Voice validation bypass** -- allows `zv_` coordinate voices through the API
5. **Tokenizer extension** -- computes correct token count for blended voices
6. **max_new_tokens safety cap** -- prevents runaway audio generation

### Step 7: Start the Server

Edit `start_server.sh` to set the correct model path, then:

```bash
bash /mnt/<drive>/ZeroVoice-Voxtral-mini-4b/start_server.sh
```

The server takes 3-5 minutes to start. Wait for `Uvicorn running on http://0.0.0.0:8000`.

## Using ZeroVoice

### Launch the Frontend

In a second WSL2 terminal:

```bash
cd ~/voxtral-tts && source .venv/bin/activate
cp /mnt/<drive>/ZeroVoice-Voxtral-mini-4b/zerovoice_frontend.py .
cp /mnt/<drive>/ZeroVoice-Voxtral-mini-4b/zerovoice.py .
cp /mnt/<drive>/ZeroVoice-Voxtral-mini-4b/slerp_voices.py .

# Copy the text preprocessor from vllm-omni
git clone --depth 1 https://github.com/vllm-project/vllm-omni.git ~/vllm-omni
cp ~/vllm-omni/examples/online_serving/voxtral_tts/text_preprocess.py .

python zerovoice_frontend.py --host localhost --port 8000 \
  --model /mnt/<drive>/ZeroVoice-Voxtral-mini-4b/Voxtral-4B-TTS-2603
```

Open **http://localhost:7860** in your browser. Gradio also provides a public share link.

### The Interface

The frontend has two tabs:

**Preset Voices** -- The original 20 Voxtral voices organized by language (English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi). Select a language, pick a voice, type text, click Generate.

**Voice Explorer** -- Three sliders (X, Y, Z) that map to a ZeroVoice coordinate. The recipe panel shows which voices are being blended and at what weight. Navigator buttons (X+/X-/Y+/Y-/Z+/Z-) let you step through the voice space. A history table tracks your recent coordinates.

### Tips for Best Results

- **Start at z=0-99** for English-primary voices with European tinting -- these produce the most natural output
- **Use the navigator buttons** with step=10 to explore smoothly. Adjacent coordinates sound similar.
- **z=100-199** gives European-primary voices (French, German, Spanish, Italian, Portuguese, Dutch) with English tinting
- **z=200+** gives Asian/Arabic-primary voices with European tinting
- **Keep text short** for testing (under 100 characters). Longer text works but generation takes proportionally longer.
- **The World Seed** control (default: 42) reshuffles the entire voice universe. Change it to explore completely different voice mappings.
- **Preset voices** are always available on the first tab for comparison and reliable high-quality output.

### API Usage

You can also use ZeroVoice directly via the API without the frontend:

```python
import io, httpx, soundfile as sf

response = httpx.post("http://localhost:8000/v1/audio/speech", json={
    "input": "Hello world, this is ZeroVoice.",
    "model": "/path/to/Voxtral-4B-TTS-2603",
    "response_format": "wav",
    "voice": "zv_50_30_10",  # Any coordinate works
}, timeout=60.0)

audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")
sf.write("output.wav", audio, sr)
```

## How This Was Built

The implementation plans used to build ZeroVoice are included in this repository. The system was developed iteratively through four stages:

1. **[VoxtralTTS-2603-TINS.md](VoxtralTTS-2603-TINS.md)** -- Initial installation plan for the Voxtral TTS model
2. **[post-install-TINS.md](post-install-TINS.md)** -- Corrected installation guide with all workarounds discovered during deployment
3. **[SLERP-voxtral-voice-plan.md](SLERP-voxtral-voice-plan.md)** -- Voice embedding SLERP blending design
4. **[ZeroVoice-Voxtral-plan.md](ZeroVoice-Voxtral-plan.md)** -- ZeroBytes coordinate-to-voice engine design
5. **[ZeroVoiceVoxtral-FrontendV2-plan.md](ZeroVoiceVoxtral-FrontendV2-plan.md)** -- Gradio frontend V2 design
6. **[stage2-quadratic-layer-plan.md](stage2-quadratic-layer-plan.md)** -- Quadratic interpolation layer (experimental)
7. **[stage3-code-review-plan.md](stage3-code-review-plan.md)** -- Code review, diagnostics, and safety caps
8. **[stage4-ZB-refactor.md](stage4-ZB-refactor.md)** -- Cross-family pairing refactor

[ZeroBytes Family Skills](https://github.com/MushroomFleet/ZeroBytes-Manifesto) were used in the planning and design of the ZeroVoice feature, following the success of [ZeroVoice-KokoroTTS](https://github.com/MushroomFleet/ZeroVoice-KokoroTTS).

## License

The Voxtral-4B-TTS-2603 model and its bundled voice embeddings are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) by Mistral AI. The ZeroVoice system code in this repository is provided as-is for research and evaluation purposes.

---

## Citation

### Academic Citation

If you use this codebase in your research or project, please cite:

```bibtex
@software{zerovoice_voxtral,
  title = {ZeroVoice-Voxtral: Procedural Voice Space for Voxtral-4B-TTS-2603},
  author = {Drift Johnson},
  year = {2025},
  url = {https://github.com/MushroomFleet/ZeroVoice-Voxtral-mini-4b},
  version = {1.0.0}
}
```

### Donate:

[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)
