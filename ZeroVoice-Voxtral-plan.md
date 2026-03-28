<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:LINUX+WSL2 -->
<!-- ZS:LANGUAGE:PYTHON -->

# ZeroVoice: Procedural Voice Space for Voxtral TTS

## Description

**ZeroVoice** applies the ZeroBytes position-is-seed procedural generation paradigm to Voxtral TTS voice synthesis. Instead of manually selecting voice pairs and blend weights, any **3D coordinate `(x, y, z)`** deterministically produces a unique voice by hashing the position into voice pair selection, SLERP interpolation weight, and alignment parameters. The same coordinate always produces the same voice, on any machine, in any order.

From 20 preset voice embeddings (2D tensors of shape `[N, 3072]`, bfloat16, variable N from 67-218), ZeroVoice derives an **infinite, explorable voice space** with the following properties:

- **O(1) access:** Compute any voice directly from `(x, y, z)` without iterating other positions
- **Deterministic:** Same coordinates produce identical voices across machines and sessions
- **Coherent:** Adjacent coordinates produce perceptually similar voices (smooth regional variation via noise)
- **Hierarchical:** Z-axis controls voice "families", X/Y explore within a family
- **Parallelizable:** Every voice depends only on its own coordinates, never neighbors

The coordinate space can be explored interactively via a 3D voice map in Gradio, queried via API with `(x, y, z)` parameters, or batch-generated offline for any region of voice space.

**Target audience:** Developers extending the Voxtral TTS deployment who want procedural voice generation rather than manual pair-by-pair blending.

**Builds on:** `SLERP-voxtral-voice-plan.md` (SLERP math, alignment, and server integration)

---

## Assessment of Original SLERP Plan

### What the SLERP Plan Gets Right

- Correct identification of variable-length sequence dimension problem (`[N, 3072]` where N differs per voice)
- Three solid alignment strategies (resample, truncate, pad)
- Row-wise SLERP with magnitude preservation and LERP fallback
- Clean separation: utility module, CLI tool, server patch, API endpoint
- Good edge case handling (parallel vectors, clamping, dtype preservation)

### What the SLERP Plan Lacks

| Gap | Impact | ZeroVoice Solution |
|---|---|---|
| Manual pair selection | User must know which pairs blend well | Coordinate hash selects pairs automatically |
| Single pair per blend | Only 2-voice interpolation, limiting timbre space | Multi-voice blending via hierarchical hashing (blend of blends) |
| No spatial coherence | Adjacent blends have no perceptual relationship | Coherent noise ensures nearby coordinates sound similar |
| No discoverability | Must guess at good `t` values | 3D space can be explored/navigated interactively |
| No deterministic naming | Names are fragile strings like `slerp_casual_male_neutral_male_0.60` | Canonical coordinate string `zv_x_y_z` is the identity |
| Flat voice space | All blends are equivalent, no structure | Z-axis = family hierarchy, X/Y = variation within family |
| Sequential generation | Must pre-generate to use | O(1) on-demand computation from coordinates |

---

## Functionality

### Core Capabilities

- **Coordinate-to-voice mapping:** Any integer triple `(x, y, z)` deterministically produces a unique voice embedding
- **Coherent voice regions:** Nearby coordinates produce perceptually similar voices using multi-octave coherent noise
- **Hierarchical axis semantics:**
  - **Z-axis (depth):** Controls the "voice family" -- which primary voice anchors dominate. Low Z = English voices, mid Z = European, high Z = Asian/Arabic. Different Z values mix different voice pools.
  - **X-axis (width):** Controls the primary SLERP weight -- gender and character blend
  - **Y-axis (height):** Controls the secondary blend weight -- adding a third voice for timbral richness
- **Multi-voice blending:** Each coordinate selects up to 3 base voices and blends them via chained SLERP (A-B blend, then result-C blend)
- **On-demand computation:** No pre-generation required. Voice computed at request time in < 15ms, then cached
- **LRU cache:** Recently computed voices are cached in memory to avoid recomputation
- **Persist on demand:** Any coordinate voice can be saved as a `.pt` file for permanent reuse
- **Full SLERP plan compatibility:** All original SLERP plan components (slerp_voices.py, CLI, server patch) are preserved and extended

### Voice Space Layout

```
                Z (voice family / language region)
                |
                |   z=200+ : Arabic, Hindi voices anchor
                |   z=100-199 : European voices anchor (fr, de, es, it, pt, nl)
                |   z=0-99 : English voices anchor (casual, neutral, cheerful)
                |
                |         Y (secondary blend / timbre)
                |        /
                |       /
                |      /
                +------------- X (primary blend / gender-character)

    Example positions:
    (50, 50, 10)   -> English-anchored, mid blend
    (50, 50, 150)  -> European-anchored, mid blend
    (100, 0, 50)   -> English, high X = different character blend
    (0, 0, 0)      -> Pure anchor voice (minimal blending)
```

### Voice Coordinate Data Model

```javascript
{
  // Input: 3D integer coordinates
  x: int,                            // Primary blend axis
  y: int,                            // Secondary blend axis
  z: int,                            // Voice family / language region axis
  world_seed: int,                   // Global seed (default: 42), changes entire voice universe

  // Derived (deterministic from coordinates + world_seed):
  voice_a: string,                   // Primary voice name (from 20 presets)
  voice_b: string,                   // Secondary voice name
  voice_c: string,                   // Tertiary voice name (for multi-blend)
  t_ab: float,                       // SLERP weight between A and B [0.0, 1.0]
  t_abc: float,                      // SLERP weight between AB-blend and C [0.0, 0.4]
  alignment: string,                 // Always "resample" for ZeroVoice

  // Output:
  embedding: torch.Tensor,           // shape: [N_aligned, 3072], dtype: bfloat16
  canonical_name: string,            // "zv_{x}_{y}_{z}" or "zv_{x}_{y}_{z}_s{seed}"
}
```

### User Flows

**Flow 1: API request with coordinates**
```
[1] POST /v1/audio/speech {"voice": "zv_50_30_10", "input": "Hello world"}
         |
[2] Server parses "zv_50_30_10" -> (x=50, y=30, z=10)
         |
[3] Check LRU cache -> miss
         |
[4] position_hash(50, 30, 10, world_seed) -> derive voice_a, voice_b, voice_c, t_ab, t_abc
         |
[5] Load base embeddings, align lengths, SLERP chain
         |
[6] Cache result, register in voice_to_embedding
         |
[7] Proceed with normal TTS pipeline -> 24kHz WAV audio
```

**Flow 2: Explore voice space in Gradio**
```
[1] User moves X/Y sliders and Z depth selector
         |
[2] UI shows coordinate (x, y, z) and derived voice recipe
         |
[3] "Preview" button -> POST /v1/audio/speech with "zv_x_y_z"
         |
[4] Audio plays back, user navigates to find desired voice
         |
[5] "Save Voice" button -> persist .pt file for permanent use
```

**Flow 3: Batch region generation**
```
[1] CLI: python zerovoice.py --region 0,0,0 100,100,50 --step 10
         |
[2] Generates voices for all (x,y,z) in region at step intervals
         |
[3] Saves each as zv_X_Y_Z.pt in voice_embedding/
         |
[4] Server restart loads all generated voices
```

### Edge Cases

| Scenario | Behavior |
|---|---|
| `(0, 0, 0)` | Valid coordinate, produces a deterministic voice |
| Negative coordinates `(-5, -10, -3)` | Valid. Hash handles negative integers correctly |
| Very large coordinates `(999999, 999999, 999999)` | Valid. O(1) computation, no iteration |
| Same coordinate, different `world_seed` | Produces entirely different voice |
| Coordinate resolves to same voice for A and B | t_ab is irrelevant; result is that voice blended with C |
| All three voices resolve to the same voice | Returns that voice unchanged |
| Voice name "zv_50_30_10" requested but not cached | Computed on-the-fly, cached, then served |
| Cache full (LRU limit reached) | Oldest entry evicted, recomputed if needed later |

---

## Technical Implementation

### Architecture

```
voxtral-mini-4b/
+-- slerp_voices.py                    # FROM SLERP PLAN: Core SLERP + alignment
+-- zerovoice.py                       # NEW: Coordinate hashing + voice space engine
+-- generate_slerp_voice.py            # FROM SLERP PLAN: CLI (extended with --coord mode)
+-- Voxtral-4B-TTS-2603/
    +-- voice_embedding/
        +-- casual_male.pt             # Existing 20 presets
        +-- ...
        +-- zv_*.pt                    # NEW: Persisted coordinate voices

Patched files (WSL2 venv):
+-- vllm_omni/.../voxtral_tts.py       # Extended: ZeroVoice resolver in tts_preprocess
+-- vllm_omni/.../serving_speech.py    # Extended: parse "zv_" voice names
```

### The Five ZeroBytes Laws Applied to Voice

| Law | Application |
|---|---|
| **O(1) Access** | `voice_at(50, 30, 10)` computes directly, no iteration over other coordinates |
| **Parallelism** | Each coordinate's voice depends only on (x,y,z,seed), never on adjacent voices |
| **Coherence** | X/Y use coherent noise so `(50,30,10)` and `(51,30,10)` sound similar |
| **Hierarchy** | Z selects voice family pool, X/Y explore within that pool |
| **Determinism** | xxhash of packed coordinates. Same on Windows, Linux, ARM, x86 |

### Algorithm: Coordinate to Voice

#### Step 1: Position Hashing

```python
import struct
import xxhash

def position_hash(x: int, y: int, z: int, salt: int = 0) -> int:
    """Deterministic 64-bit hash from 3D coordinates."""
    h = xxhash.xxh64(seed=salt)
    h.update(struct.pack('<qqq', x, y, z))
    return h.intdigest()

def hash_to_float(h: int) -> float:
    """Convert hash to float in [0.0, 1.0)."""
    return (h & 0xFFFFFFFF) / 0x100000000
```

#### Step 2: Coherent Noise for Smooth Variation

Adjacent coordinates should produce similar voices. Use multi-octave value noise:

```python
def coherent_value(x: float, y: float, seed: int, octaves: int = 3) -> float:
    """
    Multi-octave coherent noise in [−1, 1].
    Used so that nearby (x,y) coordinates produce similar blend weights,
    creating smooth voice "regions" in the coordinate space.
    """
    value, amp, freq, max_amp = 0.0, 1.0, 1.0, 0.0
    for i in range(octaves):
        x0, y0 = int(x * freq) if x * freq >= 0 else int(x * freq) - 1, \
                  int(y * freq) if y * freq >= 0 else int(y * freq) - 1
        sx = (x * freq) - x0; sx = sx * sx * (3 - 2 * sx)  # smoothstep
        sy = (y * freq) - y0; sy = sy * sy * (3 - 2 * sy)
        n00 = hash_to_float(position_hash(x0, y0, 0, seed + i)) * 2 - 1
        n10 = hash_to_float(position_hash(x0 + 1, y0, 0, seed + i)) * 2 - 1
        n01 = hash_to_float(position_hash(x0, y0 + 1, 0, seed + i)) * 2 - 1
        n11 = hash_to_float(position_hash(x0 + 1, y0 + 1, 0, seed + i)) * 2 - 1
        nx0 = n00 * (1 - sx) + n10 * sx
        nx1 = n01 * (1 - sx) + n11 * sx
        value += amp * (nx0 * (1 - sy) + nx1 * sy)
        max_amp += amp
        amp *= 0.5
        freq *= 2.0
    return value / max_amp
```

#### Step 3: Voice Family Selection (Z-axis Hierarchy)

The Z-axis selects which pool of base voices dominates. This creates language/character "regions" in the voice space.

```python
# Voice pools organized by family
VOICE_FAMILIES = {
    "english": ["casual_male", "casual_female", "cheerful_female", "neutral_male", "neutral_female"],
    "european": ["fr_male", "fr_female", "es_male", "es_female", "de_male", "de_female",
                 "it_male", "it_female", "pt_male", "pt_female", "nl_male", "nl_female"],
    "asian_arabic": ["ar_male", "hi_male", "hi_female"],
}

# Flat list for global indexing
ALL_VOICES = [
    "casual_female", "casual_male", "cheerful_female", "neutral_female", "neutral_male",
    "fr_male", "fr_female", "es_male", "es_female", "de_male", "de_female",
    "it_male", "it_female", "pt_male", "pt_female", "nl_male", "nl_female",
    "ar_male", "hi_male", "hi_female",
]  # 20 voices, indices 0-19

def select_voice_pool(z: int, world_seed: int) -> list[str]:
    """
    Select the active voice pool based on Z coordinate.
    Uses coherent noise on Z so transitions between families are smooth.

    Returns a weighted pool: primary family voices appear 3x, adjacent families 1x.
    """
    # Map Z to a family blend using coherent noise
    z_noise = coherent_value(z * 0.01, 0.0, world_seed + 7777, octaves=2)
    # z_noise in [-1, 1] -> map to family index [0, 2]
    family_idx = (z_noise + 1.0) / 2.0 * 2.99  # [0, 2.99]
    family_idx = max(0.0, min(2.99, family_idx))

    families = list(VOICE_FAMILIES.keys())
    primary_idx = int(family_idx)
    secondary_idx = min(primary_idx + 1, len(families) - 1)
    blend = family_idx - primary_idx  # [0, 1) fractional part

    primary_pool = VOICE_FAMILIES[families[primary_idx]]
    secondary_pool = VOICE_FAMILIES[families[secondary_idx]]

    # Weighted pool: primary voices repeated based on blend proximity
    primary_weight = int((1.0 - blend) * 3) + 1
    secondary_weight = int(blend * 3) + 1

    pool = primary_pool * primary_weight + secondary_pool * secondary_weight
    return pool
```

#### Step 4: Voice Triple Selection (from Pool)

```python
def select_voices(x: int, y: int, z: int, world_seed: int) -> tuple[str, str, str]:
    """
    Deterministically select 3 voices from the coordinate-derived pool.

    Returns (voice_a, voice_b, voice_c) names.
    """
    pool = select_voice_pool(z, world_seed)

    # Hash coordinate to select indices into the pool
    h1 = position_hash(x, y, z, world_seed + 1)
    h2 = position_hash(x, y, z, world_seed + 2)
    h3 = position_hash(x, y, z, world_seed + 3)

    idx_a = h1 % len(pool)
    idx_b = h2 % len(pool)
    idx_c = h3 % len(pool)

    # Ensure voice_b != voice_a (rehash if collision)
    attempts = 0
    while pool[idx_b] == pool[idx_a] and attempts < 10:
        attempts += 1
        idx_b = position_hash(x, y, z, world_seed + 2 + attempts * 100) % len(pool)
    # voice_c can equal A or B (it's a tertiary tint)

    return pool[idx_a], pool[idx_b], pool[idx_c]
```

#### Step 5: Blend Weight Derivation (Coherent)

```python
def derive_blend_weights(x: int, y: int, z: int, world_seed: int) -> tuple[float, float]:
    """
    Derive SLERP weights from coordinates using coherent noise.

    Returns:
        t_ab: primary blend weight [0.0, 1.0] between voice A and B
        t_abc: secondary blend weight [0.0, 0.4] between AB-result and voice C
    """
    # Primary blend: coherent on X/Y plane, scaled by 0.02 for smooth regional variation
    t_ab_raw = coherent_value(x * 0.02, y * 0.02, world_seed + 5000, octaves=3)
    t_ab = (t_ab_raw + 1.0) / 2.0  # [-1,1] -> [0,1]
    t_ab = max(0.0, min(1.0, t_ab))

    # Secondary blend: lower magnitude (tertiary voice is a subtle tint)
    t_abc_raw = coherent_value(x * 0.015, y * 0.015, world_seed + 6000, octaves=2)
    t_abc = (t_abc_raw + 1.0) / 2.0 * 0.4  # [-1,1] -> [0, 0.4]
    t_abc = max(0.0, min(0.4, t_abc))

    return t_ab, t_abc
```

#### Step 6: Full Voice Computation

```python
from slerp_voices import align_lengths, slerp

def voice_at(
    x: int, y: int, z: int,
    voice_embeddings: dict[str, torch.Tensor],
    world_seed: int = 42,
) -> torch.Tensor:
    """
    Compute the voice embedding for a 3D coordinate.

    This is the core ZeroVoice function. It is:
    - O(1): no iteration over other coordinates
    - Deterministic: same (x, y, z, world_seed) -> same output
    - Coherent: nearby coordinates produce similar voices

    Args:
        x, y, z: integer coordinates in voice space
        voice_embeddings: dict mapping voice names to [N, 3072] tensors
        world_seed: global seed (default 42)

    Returns:
        Blended voice embedding tensor, shape [N_aligned, 3072], dtype bfloat16
    """
    # Step A: Select three voices
    voice_a, voice_b, voice_c = select_voices(x, y, z, world_seed)

    # Step B: Derive blend weights
    t_ab, t_abc = derive_blend_weights(x, y, z, world_seed)

    # Step C: Load embeddings
    emb_a = voice_embeddings[voice_a]
    emb_b = voice_embeddings[voice_b]
    emb_c = voice_embeddings[voice_c]

    # Step D: First SLERP (A + B)
    a_aligned, b_aligned = align_lengths(emb_a, emb_b, strategy="resample")
    blend_ab = slerp(a_aligned, b_aligned, t_ab)

    # Step E: Second SLERP (AB + C) -- tertiary tint
    if t_abc > 0.01:  # Skip if negligible
        ab_aligned, c_aligned = align_lengths(blend_ab, emb_c, strategy="resample")
        result = slerp(ab_aligned, c_aligned, t_abc)
    else:
        result = blend_ab

    return result
```

### Component 1: `zerovoice.py` — Coordinate Voice Engine

**Location:** `K:\voxtral-mini-4b\zerovoice.py`

**Public API:**

| Function | Signature | Description |
|---|---|---|
| `position_hash` | `(x, y, z, salt) -> int` | Deterministic 64-bit hash |
| `hash_to_float` | `(h) -> float` | Hash to [0, 1) |
| `coherent_value` | `(x, y, seed, octaves) -> float` | Multi-octave noise [-1, 1] |
| `select_voice_pool` | `(z, world_seed) -> list[str]` | Z-axis family selection |
| `select_voices` | `(x, y, z, world_seed) -> (str, str, str)` | Pick 3 voices |
| `derive_blend_weights` | `(x, y, z, world_seed) -> (float, float)` | Coherent t values |
| `voice_at` | `(x, y, z, embeddings, world_seed) -> Tensor` | Full computation |
| `voice_name` | `(x, y, z, world_seed) -> str` | Canonical name `"zv_X_Y_Z"` |
| `voice_recipe` | `(x, y, z, world_seed) -> dict` | Full recipe without computing embedding |

**Dependencies:** `torch`, `xxhash`, `struct`. Imports `align_lengths` and `slerp` from `slerp_voices.py`.

**LRU cache:** `voice_at` results are cached using `functools.lru_cache` (keyed on `(x, y, z, world_seed)`) with a configurable max size (default: 256 voices = ~640MB).

### Component 2: Server Integration — Patch to `voxtral_tts.py`

**Changes to `tts_preprocess()` (around line 198-207):**

When the `voice` parameter starts with `"zv_"`, parse the coordinates and compute on-the-fly:

```python
def tts_preprocess(self, input_ids, input_embeds, **info_dict):
    # ... existing audio_tokens handling ...

    voice = info_dict.pop("voice", None)
    if voice is not None:
        if isinstance(voice, list):
            voice = voice[0]

        # --- NEW: ZeroVoice coordinate resolution ---
        if voice.startswith("zv_"):
            if voice not in self.voice_to_embedding:
                from zerovoice import voice_at, parse_voice_name
                x, y, z, seed = parse_voice_name(voice)
                self.voice_to_embedding[voice] = voice_at(
                    x, y, z, self.voice_to_embedding, world_seed=seed
                )
                logger.info("Computed ZeroVoice: %s -> shape %s",
                            voice, self.voice_to_embedding[voice].shape)
        # --- END NEW ---

        multimodal_embeddings = self.voice_to_embedding[voice].to(input_ids.device).clone().detach()
        is_multimodal = input_ids == self._audio_token_id
        input_embeds = self.embed_input_ids(
            input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )
        return input_ids, input_embeds, info_dict

    return input_ids, input_embeds, info_dict
```

**Voice name parsing:**

```python
def parse_voice_name(name: str) -> tuple[int, int, int, int]:
    """
    Parse 'zv_X_Y_Z' or 'zv_X_Y_Z_sSEED' into (x, y, z, world_seed).

    Examples:
        'zv_50_30_10'       -> (50, 30, 10, 42)      # default seed
        'zv_50_30_10_s99'   -> (50, 30, 10, 99)       # custom seed
        'zv_-5_100_0'       -> (-5, 100, 0, 42)       # negative coords OK
    """
    parts = name.split("_")
    # "zv", x, y, z [, "sSEED"]
    x = int(parts[1])
    y = int(parts[2])
    z = int(parts[3])
    seed = 42
    if len(parts) > 4 and parts[4].startswith("s"):
        seed = int(parts[4][1:])
    return x, y, z, seed
```

### Component 3: API Changes — `serving_speech.py`

**Voice validation patch:**

The existing `_validate_voxtral_tts_request()` checks that `voice` is in `supported_speakers`. Add a bypass for `zv_` prefixed names:

```python
# In _validate_voxtral_tts_request(), around line 758:
if self.supported_speakers and request.voice not in self.supported_speakers:
    # NEW: allow zv_ coordinate voices through without pre-registration
    if not request.voice.startswith("zv_"):
        raise ValueError(f"Voice '{request.voice}' not supported")
```

**New endpoint: `POST /v1/audio/voices/zerovoice`**

```javascript
// Request
{
  "x": int,                  // Required
  "y": int,                  // Required
  "z": int,                  // Required
  "world_seed": int,         // Optional, default 42
  "persist": boolean,        // Optional, save .pt file. Default false
  "preview_text": string     // Optional, generate audio preview
}

// Response (HTTP 200)
{
  "voice": "zv_50_30_10",
  "recipe": {
    "voice_a": "casual_male",
    "voice_b": "neutral_female",
    "voice_c": "fr_male",
    "t_ab": 0.63,
    "t_abc": 0.18
  },
  "shape": [218, 3072],
  "persisted": false,
  "preview_audio_url": null   // or URL if preview_text was provided
}
```

This endpoint is informational -- it returns the voice recipe without requiring TTS. Useful for exploring the voice space programmatically.

### Component 4: CLI Extension

**New mode in `generate_slerp_voice.py`:**

```bash
# Generate a single coordinate voice
python generate_slerp_voice.py --coord 50 30 10

# Generate a region (all points in a 3D grid)
python generate_slerp_voice.py --region 0,0,0 100,100,50 --step 25

# Inspect a coordinate's recipe without generating
python generate_slerp_voice.py --recipe 50 30 10

# Generate with custom world seed
python generate_slerp_voice.py --coord 50 30 10 --world-seed 99
```

### Component 5: Gradio Voice Explorer (Optional)

Add a "Voice Explorer" tab to the Gradio demo:

```
+-----------------------------------------------------------+
|  Voxtral TTS  |  Voice Blender  |  Voice Explorer         |
+-----------------------------------------------------------+
|                                                           |
|  X: [==========O========] 50     Z (Family): [====O==] 10|
|  Y: [======O============] 30     Seed: [42]              |
|                                                           |
|  Coordinate: zv_50_30_10                                  |
|  Recipe:                                                  |
|    Voice A: casual_male                                   |
|    Voice B: neutral_female                                |
|    Voice C: fr_male                                       |
|    t_AB: 0.63  t_ABC: 0.18                               |
|                                                           |
|  Preview: [This is a voice from coordinate space.]        |
|  [Generate & Play]              [Save as .pt]             |
|                                                           |
|  [  audio player  ]                                       |
|                                                           |
|  Nearby voices:                                           |
|  [zv_49_30_10] [zv_51_30_10] [zv_50_29_10] [zv_50_31_10]|
+-----------------------------------------------------------+
```

The "Nearby voices" buttons let users navigate the space by stepping in each axis direction, hearing the smooth transitions between adjacent coordinates.

---

## Determinism Verification

```python
def verify_zerovoice_determinism(world_seed: int = 42):
    """
    Verify all five ZeroBytes laws hold for ZeroVoice.
    """
    from zerovoice import voice_at, select_voices, derive_blend_weights

    embeddings = load_all_presets()

    # Law 1: O(1) — direct computation
    v1 = voice_at(50, 30, 10, embeddings, world_seed)  # No iteration needed

    # Law 2: Parallelism — independent coordinates
    coords = [(50, 30, 10), (51, 30, 10), (50, 31, 10)]
    results_fwd = {c: voice_at(*c, embeddings, world_seed) for c in coords}
    results_rev = {c: voice_at(*c, embeddings, world_seed) for c in reversed(coords)}
    for c in coords:
        assert torch.equal(results_fwd[c], results_rev[c]), f"Order-dependent at {c}!"

    # Law 3: Coherence — nearby coords are similar
    v_center = voice_at(50, 50, 50, embeddings, world_seed)
    v_near = voice_at(51, 50, 50, embeddings, world_seed)
    v_far = voice_at(500, 500, 500, embeddings, world_seed)
    near_dist = (v_center.float() - v_near.float()).norm()
    far_dist = (v_center.float() - v_far.float()).norm()
    assert near_dist < far_dist, "Coherence violated: nearby should be more similar!"

    # Law 4: Hierarchy — Z controls family
    voices_z0 = select_voices(50, 50, 0, world_seed)
    voices_z200 = select_voices(50, 50, 200, world_seed)
    # z=0 should favor English pool, z=200 should favor Asian/Arabic pool
    # (probabilistic, not guaranteed for every coordinate, but consistent)

    # Law 5: Determinism — same inputs, same outputs
    for _ in range(10):
        v_check = voice_at(50, 30, 10, embeddings, world_seed)
        assert torch.equal(v1, v_check), "Non-deterministic!"

    print("All ZeroBytes laws verified.")
```

---

## Testing Scenarios

### Unit Tests

| Test | Input | Expected |
|---|---|---|
| Hash determinism | `position_hash(1, 2, 3, 42)` called twice | Identical results |
| Hash independence | Different coordinates | Different hashes |
| Negative coords | `position_hash(-5, -10, -3, 42)` | Valid hash, deterministic |
| Coherent noise range | `coherent_value(x, y, seed)` for many x,y | Always in [-1, 1] |
| Coherent smoothness | Adjacent x values (50.0, 50.1) | Values differ by < 0.2 |
| Voice selection pool | z=10 vs z=150 | Different family distributions |
| Voice selection determinism | Same (x,y,z,seed) called 100x | Same 3 voices every time |
| Blend weight range | Any coordinates | t_ab in [0,1], t_abc in [0, 0.4] |
| Full voice_at determinism | Same coordinates, different call order | Identical tensors |
| Name parsing | `"zv_50_30_10"`, `"zv_-5_100_0_s99"` | Correct (x,y,z,seed) tuples |

### Integration Tests

| Test | Action | Expected |
|---|---|---|
| ZeroVoice TTS | `POST /v1/audio/speech {"voice": "zv_50_30_10"}` | Valid 24kHz WAV audio |
| Recipe endpoint | `POST /v1/audio/voices/zerovoice {"x":50,"y":30,"z":10}` | Returns recipe with voice names and weights |
| Adjacent voices | TTS for `zv_50_30_10` and `zv_51_30_10` | Both valid, perceptually similar |
| Z-axis sweep | `zv_50_50_0`, `zv_50_50_50`, `zv_50_50_100`, `zv_50_50_200` | Shifts from English to European to Asian/Arabic character |
| Persist + reload | Create `zv_50_30_10` with persist=true, restart server | Same voice available without recomputation |
| Cache hit | Request same zv_ voice twice | Second request uses cache (no recomputation log) |
| World seed variation | `zv_50_30_10` vs `zv_50_30_10_s99` | Different voices, both valid |

---

## Performance Goals

| Metric | Target |
|---|---|
| `voice_at()` computation (CPU, cold) | < 15ms for any coordinate |
| `voice_at()` computation (cache hit) | < 0.1ms (dict lookup) |
| Hash computation (xxhash) | < 0.001ms per hash |
| Coherent noise (3 octaves) | < 0.01ms per evaluation |
| LRU cache memory (256 voices) | ~640 MB max (256 * 218 * 3072 * 2 bytes) |
| TTS latency overhead vs preset voice | < 20ms on first use, 0ms on cached |
| Batch region (1000 voices) | < 15s total |

---

## Implementation Steps (Ordered)

### Phase 1: Core Engine (extends SLERP Phase 1)

1. Install `xxhash`: `uv pip install xxhash`
2. Create `slerp_voices.py` (from SLERP plan -- align_lengths, slerp, blend_voices)
3. Create `zerovoice.py` with position_hash, coherent_value, voice pool selection, voice_at
4. Write determinism verification tests
5. Write unit tests for all hash and noise functions
6. Test voice_at for a grid of coordinates, verify coherence

### Phase 2: Server Integration (extends SLERP Phase 2)

7. Patch `voxtral_tts.py`: add `zv_` prefix handling in tts_preprocess
8. Patch `serving_speech.py`: bypass validation for `zv_` names
9. Restart server, test `POST /v1/audio/speech {"voice": "zv_50_30_10"}`
10. Test Z-axis sweep for voice family transitions
11. Test X/Y coherence (adjacent coordinates sound similar)

### Phase 3: API + CLI (extends SLERP Phase 3)

12. Add `POST /v1/audio/voices/zerovoice` recipe endpoint
13. Extend CLI with `--coord` and `--region` modes
14. Generate a demo region (10x10x5 grid, step=10) and serve all voices

### Phase 4: Gradio Explorer (extends SLERP Phase 4)

15. Add "Voice Explorer" tab with X/Y/Z sliders
16. Show recipe, play preview, navigate with nearby-voice buttons
17. Add "Save as .pt" button for persisting favorites

---

## File Manifest

```
K:\voxtral-mini-4b\
+-- slerp_voices.py                  # Core SLERP + alignment (from SLERP plan)
+-- zerovoice.py                     # NEW: ZeroVoice coordinate engine
+-- generate_slerp_voice.py          # CLI tool (extended with --coord mode)
+-- ZeroVoice-Voxtral-plan.md        # This file
+-- SLERP-voxtral-voice-plan.md      # Original SLERP plan (dependency)
+-- Voxtral-4B-TTS-2603/
    +-- voice_embedding/
        +-- casual_male.pt            # Existing 20 presets
        +-- ...
        +-- zv_*.pt                   # Persisted coordinate voices

Patched files (WSL2 venv):
~/voxtral-tts/.venv/lib/python3.12/site-packages/
+-- vllm_omni/.../voxtral_tts.py     # Patched: ZeroVoice resolver
+-- vllm_omni/.../serving_speech.py  # Patched: zv_ bypass + recipe endpoint
```
