<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:MEDIUM -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:LINUX+WSL2 -->
<!-- ZS:LANGUAGE:PYTHON -->

# Stage 4: ZeroBytes Voice Selection Refactor

## Problems Identified

### Problem 1: Z=0 Maps to Asian/Arabic, Not English

The coherent noise function at `z * 0.01` with `world_seed + 7777` maps z=0-125 entirely to the `asian_arabic` pool (3 voices: ar_male, hi_male, hi_female). **English voices (casual, neutral, cheerful) are never selected at any Z value.**

| Z Range | Current Pool | Expected Pool |
|---|---|---|
| 0-125 | asian_arabic only (3 voices) | English (5 voices) |
| 150-200 | european + asian_arabic | European (12 voices) |
| Never reached | english | Asian/Arabic (3 voices) |

The family order `["english", "european", "asian_arabic"]` is correct, but the noise-to-index mapping inverts it.

### Problem 2: voice_c Adds No Value

voice_c is blended at t_abc = 0.03-0.06 (capped at 0.10). At these weights, the tertiary voice contributes < 5% to the output -- perceptually inaudible. It adds computation cost (a second align + SLERP), increases the output token count (via `max(Na, Nb, Nc)`), and complicates the tokenizer patch. Remove it.

### Problem 3: Same-Family Blending Produces Bad Results

The current system selects both A and B from the same weighted pool. This means:
- At z=0-125: A=hi_male, B=ar_male (both asian_arabic) -- garbled cross-language mix of two short-N voices
- At z=150+: A=nl_male, B=fr_male (both european) -- slightly better but still same-family

The user's insight: **always blend a European voice with a non-European voice** (English or Asian/Arabic). Cross-family pairing gives the model the best chance of rendering coherent output because the primary voice's reference structure is preserved while the secondary adds subtle accent/timbre.

### Problem 4: Only 3 Voices in the Asian/Arabic Pool

The asian_arabic family has only ar_male, hi_male, hi_female. With the guaranteed B!=A pattern, many coordinates collapse to the same few pairs, reducing variety.

---

## Design: Cross-Family Pairing with Fixed Axis Semantics

### New Voice Selection Strategy

**Remove the weighted pool entirely.** Instead, use fixed axis semantics:

- **Voice A (primary, dominant):** Selected from a pool based on Z-axis (family) and X-axis (index within family)
- **Voice B (secondary, tint at t <= 0.20):** Always selected from a DIFFERENT family than A
- **voice_c: removed from pipeline**

### Family Assignment (Z-axis)

Use direct thresholds instead of coherent noise (which produced wrong mappings):

| Z Range | Voice A Pool | Voice B Pool |
|---|---|---|
| z < 100 | english (5 voices) | european (12 voices) |
| 100 <= z < 200 | european (12 voices) | english (5 voices) |
| z >= 200 | asian_arabic (3 voices) | european (12 voices) |

This guarantees:
- A and B are always from different families
- Every Z range has access to its natural family as the primary
- English voices appear at low Z (the most common starting position)

### Voice Index Selection (X-axis)

Within the chosen pool, X-axis selects which specific voice via hash:

```python
idx_a = position_hash(x, y, z, world_seed + 1) % len(pool_a)
idx_b = position_hash(x, y, z, world_seed + 2) % len(pool_b)
```

Since A and B are from different pools, B!=A is guaranteed without any retry logic.

### Blend Weight (Y-axis coherent noise, capped)

Y-axis drives the blend weight via coherent noise, capped at 0.20:

```python
t_raw = coherent_value(x * 0.02, y * 0.02, world_seed + 5000, octaves=3)
t = (t_raw + 1.0) / 2.0 * 0.20  # [0.0, 0.20]
```

### Gender Filter

The gender filter from Stage 3 is preserved. When `gender="male"`, both pools are filtered to male-only voices. When `gender="female"`, female-only. `"mixed"` (default) allows any.

---

## Updated Code

### `select_voices()` — Cross-Family Pairing

```python
def select_voices(
    x: int, y: int, z: int, world_seed: int, gender: str = "mixed"
) -> tuple[str, str]:
    """Select voice pair with A and B from different families.

    Z-axis determines the primary family for A:
      z < 100:  A from english,       B from european
      100-199:  A from european,       B from english
      z >= 200: A from asian_arabic,   B from european

    Returns (voice_a, voice_b). No voice_c.
    """
    if z < 100:
        pool_a = VOICE_FAMILIES["english"]
        pool_b = VOICE_FAMILIES["european"]
    elif z < 200:
        pool_a = VOICE_FAMILIES["european"]
        pool_b = VOICE_FAMILIES["english"]
    else:
        pool_a = VOICE_FAMILIES["asian_arabic"]
        pool_b = VOICE_FAMILIES["european"]

    # Apply gender filter
    if gender != "mixed":
        filtered_a = [v for v in pool_a if VOICE_GENDER.get(v) == gender]
        filtered_b = [v for v in pool_b if VOICE_GENDER.get(v) == gender]
        if filtered_a:
            pool_a = filtered_a
        if filtered_b:
            pool_b = filtered_b

    h_a = position_hash(x, y, z, world_seed + 1)
    h_b = position_hash(x, y, z, world_seed + 2)

    voice_a = pool_a[h_a % len(pool_a)]
    voice_b = pool_b[h_b % len(pool_b)]

    return voice_a, voice_b
```

### `derive_blend_weights()` — Single Weight, No Tertiary

```python
def derive_blend_weights(x: int, y: int, z: int, world_seed: int) -> float:
    """Derive a single SLERP weight from coordinates.

    Returns t_ab in [0.0, 0.20], capped for reference coherence.
    """
    t_raw = coherent_value(x * 0.02, y * 0.02, world_seed + 5000, octaves=3)
    t = (t_raw + 1.0) / 2.0 * 0.20
    return max(0.0, min(0.20, t))
```

### `voice_at()` — Single SLERP, No Tertiary

```python
def voice_at(x, y, z, voice_embeddings, world_seed=42):
    va, vb = select_voices(x, y, z, world_seed)
    t = derive_blend_weights(x, y, z, world_seed)

    emb_a = voice_embeddings[va]
    emb_b = voice_embeddings[vb]

    a_aligned, b_aligned = align_lengths(emb_a, emb_b, strategy="resample")
    result = slerp(a_aligned, b_aligned, t)
    result = calibrate_norms(result, target_mean_norm=4.48)

    return result
```

### `voice_recipe()` — Simplified

```python
def voice_recipe(x, y, z, world_seed=42):
    va, vb = select_voices(x, y, z, world_seed)
    t = derive_blend_weights(x, y, z, world_seed)
    return {
        "coordinate": (x, y, z),
        "world_seed": world_seed,
        "voice_a": va,
        "voice_b": vb,
        "t": round(t, 4),
        "canonical_name": voice_name(x, y, z, world_seed),
    }
```

### Tokenizer Patch — Two Voices Only

```python
if voice.startswith("zv_"):
    try:
        from zerovoice import parse_voice_name, select_voices
        _x, _y, _z, _seed = parse_voice_name(voice)
        _va, _vb = select_voices(_x, _y, _z, _seed)
        _vnat = self.audio_config.voice_num_audio_tokens
        num_audio_tokens = max(_vnat.get(_va, 147), _vnat.get(_vb, 147))
    except Exception:
        _vals = sorted(self.audio_config.voice_num_audio_tokens.values())
        num_audio_tokens = _vals[len(_vals) // 2]
```

---

## Implementation Steps

### Step 1: Rewrite `zerovoice.py`

- `select_voices()` returns 2 voices (not 3), cross-family pairing via Z thresholds
- `derive_blend_weights()` returns single float (not tuple)
- `voice_at()` single SLERP + calibrate_norms (no tertiary blend)
- `voice_recipe()` simplified fields (no voice_c, no t_abc)
- Remove `select_voice_pool()` (replaced by direct threshold logic)
- Keep: `VOICE_GENDER`, `position_hash`, `coherent_value`, `voice_name`, `parse_voice_name`, `load_preset_embeddings`

### Step 2: Update tokenizer patch

- `max(_va, _vb)` instead of `max(_va, _vb, _vc)`
- `select_voices` returns 2 values

### Step 3: Deploy to WSL2

- Copy to site-packages + frontend directory
- Apply tokenizer patch
- Restart server

### Step 4: Test

- z=0-99 coordinates should produce English primary voices (casual, neutral, cheerful)
- z=100-199 should produce European primary voices
- z=200+ should produce Asian/Arabic primary voices
- B is always from a different family than A
- All coordinates produce < 15s audio for short text
- Gender filter works when specified

---

## Expected Voice Distribution After Refactor

### Z=0-99 (English primary + European tint)

| Voice A (primary) | Available as A |
|---|---|
| casual_male | Yes |
| casual_female | Yes |
| cheerful_female | Yes |
| neutral_male | Yes |
| neutral_female | Yes |

Voice B drawn from: fr_male, fr_female, es_male, es_female, de_male, de_female, it_male, it_female, pt_male, pt_female, nl_male, nl_female (12 choices)

### Z=100-199 (European primary + English tint)

| Voice A (primary) | Available as A |
|---|---|
| 12 European voices | All |

Voice B drawn from: 5 English voices

### Z=200+ (Asian/Arabic primary + European tint)

| Voice A (primary) | Available as A |
|---|---|
| ar_male, hi_male, hi_female | All 3 |

Voice B drawn from: 12 European voices

---

## Files Changed

```
zerovoice.py:
  REWRITE: select_voices() — cross-family, returns 2 voices
  REWRITE: derive_blend_weights() — returns single float
  REWRITE: voice_at() — single SLERP, no tertiary
  REWRITE: voice_recipe() — simplified fields
  REMOVE: select_voice_pool() (replaced by threshold logic)
  KEEP: everything else

Patches:
  MOD: mistral_common/audio.py — max(Na, Nb) instead of max(Na, Nb, Nc)
```
