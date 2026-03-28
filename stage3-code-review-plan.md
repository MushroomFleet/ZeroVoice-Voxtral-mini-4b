<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:CRITICAL -->
<!-- ZS:PLATFORM:LINUX+WSL2 -->
<!-- ZS:LANGUAGE:PYTHON -->

# Stage 3: Code Review — Rollback Quadratic, Restore Pure ZeroVoice

## Assessment

### What Was Working (Stage 1 ZeroVoice, Pre-Quadratic)

The original ZeroVoice SLERP blending produced coherent speech in early tests:

| Coordinate | Text | Duration | Status |
|---|---|---|---|
| zv_50_30_10 | "Hello, I am a procedurally generated voice..." | 10.96s | Coherent speech |
| zv_51_30_10 | "Hello, I am a nearby voice..." | 13.76s | Coherent speech |
| zv_50_30_150 | "Bonjour, je suis une voix..." | 6.48s | Coherent speech |
| zv_100_100_0 | "This is another voice..." | 9.12s | Coherent speech |

### What Broke (Stage 2 Quadratic Layer)

The quadratic layer introduced three distortions that pushed embeddings out of distribution:
- **Curvature [-0.5, 0.5]** displaced the control point off-geodesic, reducing norms 15%
- **Tilt [0.8, 1.2]** warped row magnitudes by up to 40%
- **Harmonic bias** (0.02 on 256 dims) doubled the mean value

These compounded with resampling artifacts to produce embeddings the model couldn't interpret, resulting in mumbled/garbled output.

### Reference: KokoroTTS ZeroVoice (Working Implementation)

The `zerovoice-kokorotts-v2-tins.md` reference shows a proven approach:

1. **Pure LERP between single rows** — Kokoro selects one 256-dim row per voice at a given token count index, then lerps. No sequence-level blending.
2. **GenderFilter** — `Female`, `Male`, `Mixed` pools control which voices can pair. This prevents mixing dissimilar voice types.
3. **B ≠ A guaranteed** — `((h_b % (n - 1)) + a_pos + 1) % n` ensures different voices without retry loops.
4. **Separate generation params** — `pitch_scale` and `energy_scale` are generation-time parameters, not embedding modifications.
5. **Quadratic is applied to the interpolation parameter t** (power-curve warp), not to the embedding vectors themselves.

### Key Architectural Difference

| Aspect | KokoroTTS | Voxtral TTS |
|---|---|---|
| Voice format | `[511, 256]` — one row per token count | `[N, 3072]` — full reference audio frames |
| Blend target | Single 256-dim row at token index | All N rows as sequence prefix |
| Blend method | LERP one row | SLERP entire sequence |
| Why blend works | Each row independently represents "style at position X" | Rows encode temporal audio content |

Voxtral's voice embeddings carry temporal content (what was said), not just style. Blending entire sequences is inherently riskier, but **the Stage 1 results show it CAN produce coherent output** when the embeddings aren't further distorted.

---

## Plan: Rollback to Pure ZeroVoice + Safety Caps + Type Filtering

### Changes to Make

**1. Remove quadratic layer from `zerovoice.py`**

Revert `voice_at()` to use standard `slerp()` instead of `quadratic_slerp()`. Remove `derive_pair_params()` from the active path. Remove quadratic fields from `voice_recipe()`.

**2. Restore `align_lengths` (max strategy) in `voice_at()`**

The Stage 1 approach used `align_lengths(strategy="resample")` which resamples to `max(Na, Nb)`. This worked. Restore it.

**3. Keep `calibrate_norms()`**

Norm calibration is a pure safety measure. Apply after SLERP to keep norms within preset distribution.

**4. Keep `max_new_tokens=256` safety cap**

The max_tokens fix for Voxtral (applying it to all TTS models, not just Fish Speech) is essential.

**5. Add GenderFilter (inspired by Kokoro)**

Add `male`, `female`, `mixed` filtering to voice pool selection. This prevents blending voices of different genders which produces the worst artifacts.

```python
class GenderFilter:
    MIXED = "mixed"
    FEMALE = "female"
    MALE = "male"

VOICE_GENDER = {
    "casual_male": "male", "casual_female": "female",
    "cheerful_female": "female", "neutral_male": "male",
    "neutral_female": "female",
    "fr_male": "male", "fr_female": "female",
    "es_male": "male", "es_female": "female",
    "de_male": "male", "de_female": "female",
    "it_male": "male", "it_female": "female",
    "pt_male": "male", "pt_female": "female",
    "nl_male": "male", "nl_female": "female",
    "ar_male": "male",
    "hi_male": "male", "hi_female": "female",
}
```

**6. Guaranteed B ≠ A pair selection (Kokoro pattern)**

Replace the retry loop with the modular arithmetic approach:
```python
idx_a = h_a % len(pool)
idx_b = ((h_b % (len(pool) - 1)) + idx_a + 1) % len(pool)
```

**7. Restore tokenizer to use `max(Na, Nb, Nc)`**

Since we're back to `align_lengths(max)`, the tokenizer token count should match.

### Updated `voice_at()` (Post-Rollback)

```python
def voice_at(x, y, z, voice_embeddings, world_seed=42):
    va, vb, vc = select_voices(x, y, z, world_seed)
    t_ab, t_abc = derive_blend_weights(x, y, z, world_seed)

    emb_a = voice_embeddings[va]
    emb_b = voice_embeddings[vb]
    emb_c = voice_embeddings[vc]

    # Stage 1 SLERP: A + B (no quadratic)
    a_aligned, b_aligned = align_lengths(emb_a, emb_b, strategy="resample")
    blend_ab = slerp(a_aligned, b_aligned, t_ab)

    # Tertiary tint: AB + C
    if t_abc > 0.01:
        ab_aligned, c_aligned = align_lengths(blend_ab, emb_c, strategy="resample")
        result = slerp(ab_aligned, c_aligned, t_abc)
    else:
        result = blend_ab

    # Norm calibration (Stage 3 safety)
    result = calibrate_norms(result, target_mean_norm=4.48)

    return result
```

### Updated `voice_recipe()` (Post-Rollback)

```python
def voice_recipe(x, y, z, world_seed=42):
    va, vb, vc = select_voices(x, y, z, world_seed)
    t_ab, t_abc = derive_blend_weights(x, y, z, world_seed)
    return {
        "coordinate": (x, y, z),
        "world_seed": world_seed,
        "voice_a": va,
        "voice_b": vb,
        "voice_c": vc,
        "t_ab": round(t_ab, 4),
        "t_abc": round(t_abc, 4),
        "canonical_name": voice_name(x, y, z, world_seed),
    }
```

### Updated `select_voices()` with Gender Filter + Guaranteed B ≠ A

```python
def select_voices(x, y, z, world_seed, gender="mixed"):
    pool = select_voice_pool(z, world_seed)

    # Apply gender filter
    if gender != "mixed":
        pool = [v for v in pool if VOICE_GENDER.get(v) == gender]
        if len(pool) < 2:
            pool = select_voice_pool(z, world_seed)  # fallback to full pool

    h_a = position_hash(x, y, z, world_seed + 1)
    h_b = position_hash(x, y, z, world_seed + 2)
    h_c = position_hash(x, y, z, world_seed + 3)

    n = len(pool)
    idx_a = h_a % n
    idx_b = ((h_b % (n - 1)) + idx_a + 1) % n  # guaranteed B ≠ A
    idx_c = h_c % n

    return pool[idx_a], pool[idx_b], pool[idx_c]
```

### Tokenizer Patch (Restore max formula)

```python
if voice.startswith("zv_"):
    try:
        from zerovoice import parse_voice_name, select_voices
        _x, _y, _z, _seed = parse_voice_name(voice)
        _va, _vb, _vc = select_voices(_x, _y, _z, _seed)
        _vnat = self.audio_config.voice_num_audio_tokens
        _na = _vnat.get(_va, 147)
        _nb = _vnat.get(_vb, 147)
        _nc = _vnat.get(_vc, 147)
        num_audio_tokens = max(max(_na, _nb), _nc)
    except Exception:
        _vals = sorted(self.audio_config.voice_num_audio_tokens.values())
        num_audio_tokens = _vals[len(_vals) // 2]
```

---

## Implementation Steps

### Step 1: Update `zerovoice.py`

- Revert `voice_at()` to use `slerp()` + `align_lengths()` (not quadratic)
- Add `VOICE_GENDER` dict
- Update `select_voices()` with gender filter + guaranteed B ≠ A
- Update `voice_recipe()` to remove quadratic fields
- Keep `calibrate_norms` import and usage
- Remove `quadratic_slerp` import (keep in slerp_voices.py as library code)

### Step 2: Update tokenizer patch

- Restore `max(max(_na, _nb), _nc)` formula

### Step 3: Deploy to WSL2

- Copy updated modules to site-packages
- Copy to frontend directory
- Restart server

### Step 4: Test

- Verify preset voices still produce clear speech
- Verify ZeroVoice coordinates produce coherent speech (not mumbling)
- Verify gender filtering works
- Verify adjacent coordinate coherence
- Verify max_tokens cap still active

---

## Files Changed

```
zerovoice.py:
  MOD: voice_at() — pure SLERP + norm calibration (no quadratic)
  MOD: select_voices() — gender filter + guaranteed B ≠ A
  MOD: voice_recipe() — no quadratic fields
  ADD: VOICE_GENDER dict
  REMOVE: derive_pair_params() from active path (keep function for reference)
  REMOVE: quadratic_slerp import

slerp_voices.py:
  KEEP: everything (quadratic_slerp stays as library code)
  KEEP: calibrate_norms, align_to_target (utility functions)

Patches:
  MOD: mistral_common/audio.py — restore max(Na,Nb,Nc) formula
  KEEP: serving_speech.py — max_new_tokens=256 + Voxtral override + zv_ bypass
  KEEP: voxtral_tts.py — ZeroVoice resolver in tts_preprocess
```
