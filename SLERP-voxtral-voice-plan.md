<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:LINUX+WSL2 -->
<!-- ZS:LANGUAGE:PYTHON -->

# SLERP Voice Interpolation for Voxtral TTS

## Description

Implement **Spherical Linear Interpolation (SLERP)** between Voxtral TTS voice embeddings to create blended synthetic voices from any two existing preset voices. This enables generating voices that exist "between" two reference speakers -- for example, 70% `casual_male` / 30% `neutral_male`, or a cross-language blend like 50% `fr_female` / 50% `es_female`.

Voxtral's 20 preset voice embeddings are **2D PyTorch tensors** of shape `[N, 3072]` where N is a variable-length sequence dimension (67-218 tokens depending on the voice) and 3072 is the model hidden dimension. Because the sequence lengths differ between voices, naive element-wise interpolation is impossible. The implementation must handle **length alignment** before applying SLERP row-wise across the hidden dimension.

The feature is implemented as:
1. A standalone **Python utility module** (`slerp_voices.py`) with the core SLERP math and length-alignment logic
2. A **patch to the vllm-omni model code** (`voxtral_tts.py`) so the server can load and serve SLERP-blended voices at runtime
3. A **CLI tool** (`generate_slerp_voice.py`) for pre-generating blended `.pt` files offline
4. An **API extension** adding a `POST /v1/audio/voices/blend` endpoint for on-the-fly voice creation

**Target audience:** Developers extending the Voxtral TTS deployment from `post-install-TINS.md`.

---

## Functionality

### Core Capabilities

- **Offline SLERP blending:** Generate a new `.pt` voice embedding file from any two voices at any interpolation weight `t` in `[0.0, 1.0]`
- **Runtime SLERP blending:** Request a blended voice via API without pre-generating files
- **Variable-length alignment:** Automatically handle the differing sequence lengths (N dimension) between voice embeddings using three strategies: truncate-to-shorter, pad-to-longer, or resample-via-interpolation
- **Row-wise SLERP:** Apply spherical interpolation independently to each row (token position) across the 3072-dimensional hidden space, preserving the geometric structure of the embedding manifold
- **Batch pre-generation:** Generate a grid of blended voices across multiple weight steps (e.g., 0.1 increments) for any voice pair
- **Seamless integration:** Blended voices work identically to preset voices -- same API, same `"voice"` parameter

### Voice Embedding Data Model

```javascript
{
  // Each .pt file contains a single tensor:
  tensor: torch.Tensor,              // shape: [N, 3072], dtype: torch.bfloat16
  //   N: variable per voice (67-218), represents reference audio token count
  //   3072: model hidden dimension (dim from params.json)
  //
  // Observed ranges across all 20 voices:
  //   N: min=67 (ar_male), max=218 (neutral_female)
  //   values: min=-4.4375, max=5.7188
  //   norm: 39.0 - 66.0
}
```

### Preset Voice Sequence Lengths (reference)

| Voice | N (tokens) | Voice | N (tokens) |
|---|---|---|---|
| `ar_male` | 67 | `nl_female` | 138 |
| `hi_female` | 86 | `casual_male` | 147 |
| `fr_male` | 97 | `it_female` | 146 |
| `es_female` | 94 | `cheerful_female` | 163 |
| `hi_male` | 132 | `de_female` | 168 |
| `es_male` | 144 | `de_male` | 169 |
| `pt_male` | 172 | `fr_female` | 175 |
| `nl_male` | 138 | `it_male` | 144 |
| `pt_female` | 208 | `neutral_male` | 214 |
| `casual_female` | 146 | `neutral_female` | 218 |

### User Flow: Offline Blending

```
[1] User runs CLI: generate_slerp_voice.py --voice_a casual_male --voice_b neutral_male --t 0.6
         |
[2] Tool loads both .pt files from voice_embedding/
         |
[3] Length alignment applied (default: resample-to-longer)
         |
[4] Row-wise SLERP at t=0.6 across all N rows of [N, 3072]
         |
[5] Result saved as voice_embedding/slerp_casual_male_neutral_male_0.60.pt
         |
[6] Server restart picks up new voice automatically (or hot-reload via API)
```

### User Flow: Runtime Blending via API

```
[1] Client sends POST /v1/audio/voices/blend
    Body: {"voice_a": "casual_male", "voice_b": "fr_female", "t": 0.5, "name": "my_blend"}
         |
[2] Server computes SLERP blend in-memory
         |
[3] Blended embedding registered in voice_to_embedding dict
         |
[4] Server responds: {"voice": "my_blend", "status": "created"}
         |
[5] Client uses the blend: POST /v1/audio/speech {"voice": "my_blend", ...}
```

### Edge Cases

| Scenario | Behavior |
|---|---|
| `t = 0.0` | Returns voice_a unchanged (but length-aligned) |
| `t = 1.0` | Returns voice_b unchanged (but length-aligned) |
| Same voice for both A and B | Returns a copy of that voice (SLERP is identity when inputs are equal) |
| Voices with identical N | Skip length alignment, apply SLERP directly |
| Near-parallel vectors (dot product ~ 1.0) | Fall back to linear interpolation (LERP) to avoid division by near-zero sin(omega) |
| Near-antiparallel vectors (dot product ~ -1.0) | Clamp dot product to [-1+eps, 1-eps] before computing omega |
| `t` outside [0.0, 1.0] | Clamp to [0.0, 1.0] with a warning log |
| Unknown voice name | Return HTTP 404 with message: `"Voice '{name}' not found"` |
| Blend name conflicts with preset voice | Return HTTP 409: `"Cannot overwrite preset voice '{name}'"` |

---

## Technical Implementation

### Architecture

```
voxtral-mini-4b/
+-- slerp_voices.py                    # NEW: Core SLERP math + alignment utilities
+-- generate_slerp_voice.py            # NEW: CLI tool for offline voice generation
+-- Voxtral-4B-TTS-2603/
    +-- voice_embedding/
        +-- casual_male.pt             # Existing preset voices (20 files)
        +-- ...
        +-- slerp_*.pt                 # NEW: Generated blended voices

Patched files (in WSL2 venv site-packages):
+-- vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py
    (add SLERP blend loading + runtime blend support)
+-- vllm_omni/entrypoints/openai/serving_speech.py
    (add POST /v1/audio/voices/blend endpoint)
```

### Algorithm: SLERP for Voice Embeddings

#### Step 1: Length Alignment

Given voice_a with shape `[Na, 3072]` and voice_b with shape `[Nb, 3072]`, when `Na != Nb`:

**Strategy: Resample to target length (default)**

Use `torch.nn.functional.interpolate` on the sequence dimension to resample the shorter tensor to match the longer one (or to a user-specified target length). This preserves the overall embedding structure while creating compatible shapes.

```python
def align_lengths(a: torch.Tensor, b: torch.Tensor, strategy: str = "resample") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align two voice embeddings to the same sequence length.

    Args:
        a: tensor of shape [Na, D]
        b: tensor of shape [Nb, D]
        strategy: one of "resample" (interpolate shorter to longer),
                  "truncate" (cut both to min length),
                  "pad" (zero-pad shorter to match longer)

    Returns:
        Tuple of two tensors with shape [N_aligned, D]
    """
    Na, Nb = a.shape[0], b.shape[0]
    if Na == Nb:
        return a, b

    if strategy == "truncate":
        N = min(Na, Nb)
        return a[:N], b[:N]

    elif strategy == "pad":
        N = max(Na, Nb)
        if Na < N:
            a = torch.nn.functional.pad(a, (0, 0, 0, N - Na))
        if Nb < N:
            b = torch.nn.functional.pad(b, (0, 0, 0, N - Nb))
        return a, b

    elif strategy == "resample":
        N = max(Na, Nb)
        # interpolate operates on [batch, channels, length], so transpose
        if Na < N:
            a = a.float().unsqueeze(0).permute(0, 2, 1)        # [1, D, Na]
            a = torch.nn.functional.interpolate(a, size=N, mode="linear", align_corners=True)
            a = a.permute(0, 2, 1).squeeze(0).to(torch.bfloat16)  # [N, D]
        if Nb < N:
            b = b.float().unsqueeze(0).permute(0, 2, 1)        # [1, D, Nb]
            b = torch.nn.functional.interpolate(b, size=N, mode="linear", align_corners=True)
            b = b.permute(0, 2, 1).squeeze(0).to(torch.bfloat16)  # [N, D]
        return a, b

    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")
```

#### Step 2: Row-wise SLERP

After alignment, both tensors have shape `[N, D]` where D=3072. Apply SLERP independently to each of the N rows. Each row is a point on a 3072-dimensional hypersphere (after normalization).

```python
def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation between two tensors.

    Operates row-wise: for tensors of shape [N, D], each of the N rows
    is independently interpolated on the D-dimensional unit hypersphere.

    Args:
        a: tensor of shape [N, D] (voice embedding A, already length-aligned)
        b: tensor of shape [N, D] (voice embedding B, already length-aligned)
        t: interpolation weight in [0.0, 1.0]. t=0 returns a, t=1 returns b.

    Returns:
        Interpolated tensor of shape [N, D]
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Work in float32 for numerical stability
    a_f = a.float()
    b_f = b.float()

    # Normalize each row to unit length
    a_norm = torch.nn.functional.normalize(a_f, dim=-1)
    b_norm = torch.nn.functional.normalize(b_f, dim=-1)

    # Compute per-row cosine similarity (dot product of unit vectors)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True)  # [N, 1]

    # Clamp to avoid numerical issues with acos
    dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    # Compute the angle omega between vectors
    omega = torch.acos(dot)  # [N, 1]

    # Compute SLERP weights
    sin_omega = torch.sin(omega)  # [N, 1]

    # For near-parallel vectors (sin(omega) ~ 0), fall back to LERP
    use_lerp = (sin_omega.abs() < 1e-6).squeeze(-1)  # [N]

    # SLERP formula: (sin((1-t)*omega) / sin(omega)) * a + (sin(t*omega) / sin(omega)) * b
    weight_a = torch.sin((1.0 - t) * omega) / sin_omega  # [N, 1]
    weight_b = torch.sin(t * omega) / sin_omega           # [N, 1]

    result = weight_a * a_f + weight_b * b_f  # [N, D]

    # Apply LERP fallback for near-parallel rows
    if use_lerp.any():
        lerp_result = (1.0 - t) * a_f + t * b_f
        result[use_lerp] = lerp_result[use_lerp]

    # Restore original magnitude (SLERP on unit sphere, then rescale)
    a_mag = a_f.norm(dim=-1, keepdim=True)  # [N, 1]
    b_mag = b_f.norm(dim=-1, keepdim=True)  # [N, 1]
    target_mag = (1.0 - t) * a_mag + t * b_mag  # Linearly interpolate magnitude
    result_mag = result.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result * (target_mag / result_mag)

    return result.to(a.dtype)  # Cast back to bfloat16
```

#### Step 3: Full Pipeline

```python
def blend_voices(
    voice_a_path: str,
    voice_b_path: str,
    t: float,
    alignment: str = "resample",
) -> torch.Tensor:
    """
    Load two voice embeddings and produce a SLERP-blended result.

    Args:
        voice_a_path: path to first .pt file
        voice_b_path: path to second .pt file
        t: interpolation weight [0.0, 1.0]
        alignment: length alignment strategy ("resample", "truncate", "pad")

    Returns:
        Blended voice embedding tensor of shape [N_aligned, 3072], dtype bfloat16
    """
    t = max(0.0, min(1.0, t))  # Clamp

    a = torch.load(voice_a_path, map_location="cpu")
    b = torch.load(voice_b_path, map_location="cpu")

    a_aligned, b_aligned = align_lengths(a, b, strategy=alignment)
    blended = slerp(a_aligned, b_aligned, t)

    return blended
```

### Component 1: `slerp_voices.py` — Core Utility Module

**Location:** `K:\voxtral-mini-4b\slerp_voices.py` (also copied to WSL2 at `~/voxtral-tts/slerp_voices.py`)

**Contents:**
- `align_lengths(a, b, strategy)` — Length alignment (as defined above)
- `slerp(a, b, t)` — Row-wise SLERP (as defined above)
- `blend_voices(voice_a_path, voice_b_path, t, alignment)` — Full pipeline (as defined above)
- `list_voices(voice_dir)` — List available `.pt` files and their shapes
- `voice_name_from_blend(voice_a, voice_b, t)` — Generate canonical name: `"slerp_{a}_{b}_{t:.2f}"`

**Dependencies:** `torch` only. No vLLM or vllm-omni imports.

### Component 2: `generate_slerp_voice.py` — CLI Tool

**Location:** `K:\voxtral-mini-4b\generate_slerp_voice.py`

**Usage:**

```bash
# Single blend
python generate_slerp_voice.py \
  --voice_a casual_male \
  --voice_b neutral_male \
  --t 0.6 \
  --voice_dir /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603/voice_embedding \
  --alignment resample

# Grid generation (multiple t values)
python generate_slerp_voice.py \
  --voice_a casual_female \
  --voice_b fr_female \
  --t_steps 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --voice_dir /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603/voice_embedding

# All-pairs sweep
python generate_slerp_voice.py \
  --all_pairs \
  --t 0.5 \
  --voice_dir /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603/voice_embedding
```

**CLI Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--voice_a` | str | required (unless `--all_pairs`) | First voice name (without `.pt`) |
| `--voice_b` | str | required (unless `--all_pairs`) | Second voice name (without `.pt`) |
| `--t` | float | 0.5 | Interpolation weight |
| `--t_steps` | float[] | None | Multiple t values (overrides `--t`) |
| `--all_pairs` | flag | False | Generate blends for all voice pair combinations |
| `--voice_dir` | str | `./Voxtral-4B-TTS-2603/voice_embedding` | Path to voice embedding directory |
| `--alignment` | str | `resample` | Length alignment strategy |
| `--output_dir` | str | same as `--voice_dir` | Where to save output .pt files |
| `--output_name` | str | auto-generated | Custom output filename |

**Output naming convention:** `slerp_{voice_a}_{voice_b}_{t:.2f}.pt`

Example: `slerp_casual_male_neutral_male_0.60.pt`

**Behavior:**
- Prints tensor info for each blend: shape, dtype, norm, min/max
- Skips if output file already exists (use `--overwrite` to force)
- Validates that both input voices exist before blending

### Component 3: Patch to `voxtral_tts.py` — Server-side Voice Loading

**File:** `vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py`

**Changes to `VoxtralTTSForConditionalGeneration.__init__()` (around line 128-140):**

After loading the preset voices from `speaker_id`, also scan for and load any `slerp_*.pt` files in the same `voice_embedding/` directory:

```python
# --- EXISTING CODE (lines 128-140): load preset voices from speaker_id ---
# speaker_id = config.audio_config.get("speaker_id", None)
# if speaker_id:
#     self.voice_to_embedding = {}
#     for sid in speaker_id:
#         ...load and store...

# --- NEW CODE: also load any slerp_*.pt files ---
voice_dir = Path(self.repo_id) / "voice_embedding"
if voice_dir.is_dir():
    for pt_file in sorted(voice_dir.glob("slerp_*.pt")):
        voice_name = pt_file.stem  # e.g., "slerp_casual_male_neutral_male_0.60"
        if voice_name not in self.voice_to_embedding:
            self.voice_to_embedding[voice_name] = torch.load(pt_file, map_location="cpu")
            logger.info("Loaded SLERP voice: %s (shape=%s)", voice_name, self.voice_to_embedding[voice_name].shape)
```

This means any `.pt` file placed in `voice_embedding/` with a `slerp_` prefix is auto-loaded on server start and immediately usable via `"voice": "slerp_casual_male_neutral_male_0.60"`.

### Component 4: Patch to `serving_speech.py` — API Blend Endpoint

**File:** `vllm_omni/entrypoints/openai/serving_speech.py`

**New endpoint: `POST /v1/audio/voices/blend`**

**Request body:**

```javascript
{
  "voice_a": string,        // Required. Name of first voice (must exist in supported_speakers)
  "voice_b": string,        // Required. Name of second voice (must exist in supported_speakers)
  "t": float,               // Required. Interpolation weight [0.0, 1.0]
  "name": string | null,    // Optional. Custom name for the blend. Default: auto-generated
  "alignment": string,      // Optional. "resample" | "truncate" | "pad". Default: "resample"
  "persist": boolean         // Optional. If true, save .pt file to disk. Default: false
}
```

**Response body (success, HTTP 201):**

```javascript
{
  "voice": string,           // Name of the created blend (usable in /audio/speech requests)
  "voice_a": string,
  "voice_b": string,
  "t": float,
  "shape": [int, int],       // e.g., [218, 3072]
  "alignment": string,
  "persisted": boolean
}
```

**Error responses:**

| Status | Condition | Body |
|---|---|---|
| 404 | voice_a or voice_b not found | `{"error": "Voice 'xyz' not found"}` |
| 409 | name conflicts with preset voice | `{"error": "Cannot overwrite preset voice 'casual_male'"}` |
| 422 | t not in [0.0, 1.0] or missing fields | `{"error": "t must be between 0.0 and 1.0"}` |

**Implementation details:**

The endpoint handler must:
1. Validate both voice names exist in `supported_speakers`
2. Retrieve the raw embedding tensors from the model's `voice_to_embedding` dict (requires access to the engine's model instance, or a copy of the embeddings stored at the serving layer)
3. Run `align_lengths()` + `slerp()` from the `slerp_voices` module
4. Register the result in `voice_to_embedding` on the model AND in `supported_speakers` on the serving layer
5. Optionally save to disk if `persist=true`

**Access to model embeddings from the serving layer:**

The serving layer (`serving_speech.py`) does not have direct access to `voice_to_embedding` on the model. Two approaches:

**Approach A (recommended): Cache embeddings at the serving layer.**
At server startup, after the engine loads the model, copy the `voice_to_embedding` dict to the `OpenAISpeechServing` instance. The blend endpoint operates on this copy and pushes new blends into both the serving-layer cache and the engine model via an RPC or shared reference.

**Approach B (simpler, for single-GPU): Load embeddings independently.**
The serving layer loads the `.pt` files directly from disk (same path the model uses). Blends are computed at the serving layer, saved to disk as `.pt` files, and the model picks them up on the next request via a modified `tts_preprocess` that checks the `voice_embedding/` directory for new files.

### Component 5: Gradio Demo Extension (Optional)

**File:** Patch to `vllm-omni/examples/online_serving/voxtral_tts/gradio_demo.py`

Add a new "Voice Blender" tab:

```
+---------------------------------------------------+
|  Voxtral TTS  |  Voice Blender                    |
+---------------------------------------------------+
| Voice A: [Language v] [Voice v]                    |
| Voice B: [Language v] [Voice v]                    |
|                                                    |
| Blend Weight (t): [====O=======] 0.50             |
|                                                    |
| Preview Text: [Hello, this is a blended voice.]   |
|                                                    |
| [Create & Preview]                                 |
|                                                    |
| Result: [  audio player  ]                         |
| Blend name: slerp_casual_male_fr_female_0.50      |
+---------------------------------------------------+
```

---

## Testing Scenarios

### Unit Tests for `slerp_voices.py`

| Test | Input | Expected |
|---|---|---|
| Identity at t=0 | `slerp(a, b, 0.0)` | Result equals `a` (within float tolerance) |
| Identity at t=1 | `slerp(a, b, 1.0)` | Result equals `b` (within float tolerance) |
| Midpoint symmetry | `slerp(a, b, 0.5)` vs `slerp(b, a, 0.5)` | Results are equal |
| Shape preservation | `slerp(a, b, 0.5)` | Output shape matches aligned input shape |
| Dtype preservation | bfloat16 inputs | Output is bfloat16 |
| Same-voice SLERP | `slerp(a, a, 0.5)` | Result equals `a` |
| Magnitude interpolation | `slerp(a, b, 0.5)` | Row norms are between norms of `a` and `b` |
| Truncate alignment | Two tensors with N=100 and N=200, strategy="truncate" | Both become [100, D] |
| Pad alignment | Two tensors with N=100 and N=200, strategy="pad" | Both become [200, D], shorter padded with zeros |
| Resample alignment | Two tensors with N=100 and N=200, strategy="resample" | Both become [200, D], shorter resampled smoothly |
| LERP fallback | Two parallel vectors | No NaN, returns linear interpolation |

### Integration Tests

| Test | Action | Expected |
|---|---|---|
| Offline blend + serve | Generate `slerp_casual_male_neutral_male_0.50.pt`, restart server | Voice appears in `/v1/audio/voices` |
| Offline blend + TTS | Request speech with `"voice": "slerp_casual_male_neutral_male_0.50"` | Returns valid 24kHz WAV audio |
| API blend + TTS | `POST /v1/audio/voices/blend`, then `POST /v1/audio/speech` with blend name | Both succeed, audio is valid |
| Blend of blend | SLERP between a preset and a previously blended voice | Works normally |
| All-pairs at t=0.5 | Generate 190 blends (20 choose 2), serve all | All produce valid audio |

### Perceptual Tests (Manual)

| Test | Expectation |
|---|---|
| `casual_male` -> `neutral_male` at t=0.25, 0.50, 0.75 | Gradual shift in vocal character, no artifacts |
| `casual_female` -> `fr_female` at t=0.5 | Accent blending between English and French vocal qualities |
| `de_male` -> `es_male` at t=0.5 | Cross-language blend, should sound natural |
| t sweep from 0.0 to 1.0 in 0.1 steps | Smooth audible progression between two voices |

---

## Performance Goals

| Metric | Target |
|---|---|
| SLERP computation (CPU, per blend) | < 10ms for any voice pair |
| Length alignment (resample) | < 5ms for [67, 3072] -> [218, 3072] |
| Disk I/O for .pt save | < 50ms |
| Server startup with 20 preset + 50 SLERP voices | < 10s additional |
| TTS latency with SLERP voice vs preset voice | No measurable difference (embedding is pre-computed) |
| Memory per additional SLERP voice | ~2.5 MB (218 * 3072 * 2 bytes bfloat16) |

---

## Implementation Steps (Ordered)

### Phase 1: Core Library

1. Create `slerp_voices.py` with `align_lengths()`, `slerp()`, `blend_voices()`, `list_voices()`, `voice_name_from_blend()`
2. Write and run unit tests for all SLERP math edge cases
3. Create `generate_slerp_voice.py` CLI tool
4. Generate a test blend: `casual_male` + `neutral_male` at t=0.5

### Phase 2: Server Integration

5. Patch `voxtral_tts.py` to auto-load `slerp_*.pt` files from `voice_embedding/`
6. Restart server, verify blend appears in `/v1/audio/voices`
7. Run TTS with the blended voice, validate audio output
8. Generate a sweep (t=0.1 to 0.9) for `casual_male` + `neutral_male`, test all

### Phase 3: API Endpoint

9. Add `slerp_voices.py` to the vllm-omni package path (or install as module)
10. Patch `serving_speech.py` to add `POST /v1/audio/voices/blend`
11. Implement embedding cache at serving layer for runtime blending
12. Test the full API flow: blend creation -> TTS generation

### Phase 4: Gradio Extension (Optional)

13. Add "Voice Blender" tab to gradio_demo.py
14. Wire blend slider + voice dropdowns to the blend API endpoint
15. Test interactive blending with audio preview

---

## File Manifest

```
K:\voxtral-mini-4b\
+-- slerp_voices.py                  # NEW: Core SLERP + alignment utilities
+-- generate_slerp_voice.py          # NEW: CLI tool for offline blending
+-- SLERP-voxtral-voice-plan.md      # This file
+-- Voxtral-4B-TTS-2603/
    +-- voice_embedding/
        +-- casual_male.pt            # Existing (20 files)
        +-- ...
        +-- slerp_*.pt               # NEW: Generated blends

Patched files (WSL2 venv):
~/voxtral-tts/.venv/lib/python3.12/site-packages/
+-- vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py   # Patched: auto-load slerp_*.pt
+-- vllm_omni/entrypoints/openai/serving_speech.py                # Patched: blend endpoint
```
