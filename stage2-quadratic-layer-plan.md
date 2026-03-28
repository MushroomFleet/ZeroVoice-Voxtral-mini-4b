<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:LINUX+WSL2 -->
<!-- ZS:LANGUAGE:PYTHON -->

# Stage 2: ZeroQuadratic Voice Layer for ZeroVoice

## Assessment of Current System

### What Exists (Stage 1)

The current ZeroVoice system (`zerovoice.py` + `slerp_voices.py`) maps 3D coordinates to blended voices via:
1. **Z-axis** selects a weighted voice pool from 3 families (English, European, Asian/Arabic)
2. **Position hash** selects 3 voices (A, B, C) from the pool
3. **Coherent noise** on X/Y derives linear blend weights `t_ab` and `t_abc`
4. **Standard SLERP** blends A+B, then result+C

This works and produces coherent output, but has three fundamental limitations.

### Limitation 1: Linear Interpolation Path

SLERP traces a **great-circle arc** between two points on the hypersphere. The arc is always the same shape -- it's the shortest geodesic. This means:
- Every blend between the same two voices at t=0.5 hits the same midpoint
- The "character" of the interpolation path is identical for all voice pairs
- Moving from t=0.0 to t=1.0 always traces the same curve geometry

**ZeroQuadratic fix:** Each voice pair gets a **unique curvature parameter** derived from a pair-specific hash. The interpolation path bows away from the geodesic via a quadratic control point, so `casual_male + fr_female` traces a different path shape than `casual_male + de_male`, even at the same `t` value.

### Limitation 2: No Pair-Derived Character

Currently, the hash selects *which* voices to blend and *how much*, but nothing about *how* they blend. Two pairs blended at t=0.5 use identical SLERP math.

**ZeroQuadratic fix:** From the hash of each voice pair, derive:
- **Curvature** (`k`): How far the interpolation path bows away from the geodesic [-0.5, 0.5]
- **Spectral tilt** (`tilt`): A per-row magnitude scaling that shifts energy toward early or late sequence positions [0.8, 1.2]
- **Harmonic bias** (`hbias`): A subtle additive offset in a hash-selected subspace of the 3072-dim hidden space, giving each pair a unique "harmonic fingerprint"

These three parameters are derived from `position_hash(voice_a_idx, voice_b_idx, 0, world_seed)` -- deterministic per pair, zero bytes stored.

### Limitation 3: Blind Pair Selection

Voice selection currently picks from a weighted pool randomly. Two voices in the same pool might be very similar (cosine ~0.99) or moderately different (cosine ~0.97). There's no awareness of *how different* the selected voices are.

**Measured pairwise cosine similarity (mean-pooled signatures):**

| Pair Type | Cosine Range | Example |
|---|---|---|
| Same-family, same-gender | 0.985 - 0.993 | casual_female + neutral_female: 0.993 |
| Same-family, cross-gender | 0.986 - 0.989 | casual_male + neutral_female: 0.989 |
| Cross-family, same-gender | 0.975 - 0.987 | casual_male + fr_male: 0.983 |
| Cross-family, cross-gender | 0.980 - 0.992 | de_female + nl_female: 0.992 |
| Most distant pair | 0.974 | hi_male + casual_female: 0.974 |

**Key insight:** All voices are highly similar (cosine > 0.97). The differences are subtle but audible. This means:
- **Blending within the same family is safe** and produces smooth transitions
- **Cross-family blending works** because voices aren't far apart in embedding space
- **No pair needs to be forbidden** -- the quadratic layer should instead *modulate how aggressively* the blend curves based on pair distance

**ZeroQuadratic approach:** Rather than preventing cross-family blending, use pair distance to **scale the curvature**. Close voices get subtle curvature (they already blend smoothly). Distant voices get more pronounced curvature (the path needs more "character" to avoid bland averaging).

### Voice Characteristic Profile

| Voice | N | Mean Norm | Norm StdDev | Temporal Coherence | Mean Value |
|---|---|---|---|---|---|
| cheerful_female | 132 | 4.42 | 0.91 | 0.807 | 0.0013 |
| neutral_female | 218 | 4.40 | 0.71 | 0.800 | 0.0013 |
| neutral_male | 169 | 4.42 | 0.81 | 0.791 | 0.0012 |
| pt_male | 144 | 4.43 | 0.88 | 0.791 | 0.0018 |
| fr_male | 97 | 4.47 | 1.07 | 0.784 | 0.0020 |
| casual_male | 147 | 4.39 | 0.86 | 0.776 | 0.0014 |
| pt_female | 175 | 4.47 | 0.80 | 0.773 | 0.0020 |
| de_male | 163 | 4.45 | 0.83 | 0.770 | 0.0020 |
| casual_female | 214 | 4.42 | 0.72 | 0.768 | 0.0013 |
| de_female | 147 | 4.46 | 0.87 | 0.767 | 0.0020 |
| nl_female | 146 | 4.43 | 0.88 | 0.768 | 0.0022 |
| es_male | 208 | 4.47 | 0.74 | 0.769 | 0.0018 |
| it_male | 168 | 4.49 | 0.81 | 0.764 | 0.0017 |
| it_female | 172 | 4.52 | 0.80 | 0.759 | 0.0018 |
| nl_male | 138 | 4.48 | 0.90 | 0.762 | 0.0019 |
| hi_male | 94 | 4.57 | 1.08 | 0.751 | 0.0022 |
| fr_female | 97 | 4.51 | 1.07 | 0.750 | 0.0020 |
| es_female | 138 | 4.55 | 0.89 | 0.747 | 0.0018 |
| ar_male | 67 | 4.58 | 1.27 | 0.746 | 0.0013 |
| hi_female | 86 | 4.57 | 1.12 | 0.740 | 0.0019 |

**Observable patterns:**
- Shorter voices (ar_male:67, hi_female:86) have higher norm variance and lower temporal coherence
- English voices cluster with highest temporal coherence (0.77-0.81)
- Norm means are tightly clustered (4.39-4.58) -- the embedding space is well-normalized
- `mean_value` distinguishes English voices (~0.0013) from non-English (~0.0019) -- a subtle spectral bias

---

## Description

Stage 2 adds a **ZeroQuadratic interpolation layer** to the ZeroVoice system. Instead of standard SLERP (great-circle geodesic), voice blending follows a **quadratic Bezier curve** on the hypersphere, where the control point is derived from a pair-specific hash. This gives every voice pair a unique interpolation "arc" with distinct curvature, harmonic character, and spectral tilt -- all computed on demand, zero bytes stored.

The layer modifies `slerp_voices.py` to support quadratic SLERP and modifies `zerovoice.py` to derive per-pair parameters from hashes. The server patches, frontend, and API remain unchanged -- the quadratic layer is transparent to callers.

**Builds on:** `zerovoice.py`, `slerp_voices.py`, all Stage 1 patches

---

## Technical Implementation

### Algorithm: Quadratic SLERP

Standard SLERP traces the shortest arc from A to B on the hypersphere. Quadratic SLERP introduces a **control point M** that pulls the arc away from the geodesic, creating a unique curvature per voice pair.

#### Computing the Control Point

For a voice pair (A, B), the control point M is computed as:

```python
def compute_control_point(
    a: torch.Tensor,       # [N, D] aligned embedding A
    b: torch.Tensor,       # [N, D] aligned embedding B
    curvature: float,      # k in [-0.5, 0.5], derived from pair hash
    hbias_dim: int,        # which dimension subspace gets the harmonic bias
    hbias_strength: float, # harmonic bias magnitude [0.0, 0.02]
) -> torch.Tensor:
    """Compute the quadratic Bezier control point M for a voice pair.

    M is the SLERP midpoint (t=0.5) displaced by:
    1. A curvature offset perpendicular to the A-B geodesic
    2. A harmonic bias in a hash-selected dimension subspace

    Args:
        a, b: [N, D] aligned voice embeddings (float32)
        curvature: how far M bows from the geodesic. 0 = standard SLERP.
        hbias_dim: starting dimension for the harmonic bias band
        hbias_strength: magnitude of the harmonic bias

    Returns:
        M: [N, D] control point tensor
    """
    # Start with geodesic midpoint
    midpoint = slerp(a, b, 0.5)  # [N, D]

    # Compute perpendicular direction to the A-B plane
    # For each row: project out the A-B direction from a random vector
    ab_dir = F.normalize(b - a, dim=-1)  # [N, D]

    # Use the cross between ab_dir and a fixed reference to get perpendicular
    # Hash-based reference vector per row avoids coordinate dependence
    ref = F.normalize(a + b, dim=-1)  # Arbitrary but deterministic
    perp = ref - (ref * ab_dir).sum(dim=-1, keepdim=True) * ab_dir
    perp = F.normalize(perp, dim=-1)  # [N, D] perpendicular to A-B

    # Displace midpoint by curvature along perpendicular
    midpoint_mag = midpoint.norm(dim=-1, keepdim=True)
    M = midpoint + curvature * midpoint_mag * perp

    # Apply harmonic bias: boost a 256-dim band starting at hbias_dim
    band_start = hbias_dim % (a.shape[1] - 256)
    band_end = band_start + 256
    M[:, band_start:band_end] += hbias_strength * midpoint_mag

    return M
```

#### Quadratic SLERP (De Casteljau on the Sphere)

The quadratic Bezier on the hypersphere uses De Casteljau's algorithm with SLERP as the primitive:

```python
def quadratic_slerp(
    a: torch.Tensor,       # [N, D]
    b: torch.Tensor,       # [N, D]
    t: float,              # [0, 1]
    curvature: float,      # [-0.5, 0.5]
    hbias_dim: int,        # harmonic bias dimension
    hbias_strength: float, # harmonic bias magnitude
    tilt: float,           # spectral tilt [0.8, 1.2]
) -> torch.Tensor:
    """Quadratic Bezier SLERP on the hypersphere.

    De Casteljau: result = SLERP(SLERP(A, M, t), SLERP(M, B, t), t)
    where M is the pair-specific control point.

    Args:
        a, b: [N, D] aligned voice embeddings (bfloat16 in, bfloat16 out)
        t: interpolation weight [0, 1]
        curvature: arc curvature from pair hash
        hbias_dim: harmonic bias band start
        hbias_strength: harmonic bias magnitude
        tilt: spectral tilt applied to result

    Returns:
        [N, D] interpolated embedding with pair-specific character
    """
    orig_dtype = a.dtype
    a_f = a.float()
    b_f = b.float()

    # Compute control point
    M = compute_control_point(a_f, b_f, curvature, hbias_dim, hbias_strength)

    # De Casteljau on sphere
    am = slerp(a_f, M, t)    # SLERP(A, M, t)
    mb = slerp(M, b_f, t)    # SLERP(M, B, t)
    result = slerp(am, mb, t) # SLERP(AM, MB, t)

    # Apply spectral tilt: scale row magnitudes based on sequence position
    N = result.shape[0]
    if N > 1:
        # tilt > 1.0 = boost early rows (onset emphasis)
        # tilt < 1.0 = boost late rows (tail emphasis)
        positions = torch.linspace(0.0, 1.0, N, device=result.device)
        # Tilt curve: exponential from tilt at start to 1/tilt at end
        tilt_curve = tilt ** (1.0 - 2.0 * positions)  # [N]
        result = result * tilt_curve.unsqueeze(1)      # [N, D]

    return result.to(orig_dtype)
```

### Deriving Pair Parameters from Hash

Every unique voice pair (A, B) gets deterministic curvature, tilt, and harmonic bias parameters. These are derived from hashing the pair's indices -- zero bytes stored.

```python
# Voice name -> index mapping (stable, alphabetical)
VOICE_INDEX = {name: i for i, name in enumerate(sorted(ALL_VOICES))}

def derive_pair_params(
    voice_a: str,
    voice_b: str,
    world_seed: int,
) -> dict:
    """Derive quadratic interpolation parameters from a voice pair hash.

    Returns dict with:
        curvature: float in [-0.5, 0.5]
        tilt: float in [0.8, 1.2]
        hbias_dim: int in [0, 2816] (start of 256-dim harmonic band)
        hbias_strength: float in [0.0, 0.02]
    """
    # Order-independent pair hash: always hash (min_idx, max_idx)
    idx_a = VOICE_INDEX.get(voice_a, 0)
    idx_b = VOICE_INDEX.get(voice_b, 0)
    pair_min, pair_max = min(idx_a, idx_b), max(idx_a, idx_b)

    pair_hash = position_hash(pair_min, pair_max, 0, world_seed + 9999)

    # Extract 4 independent floats from the hash
    h0 = hash_to_float(pair_hash)
    h1 = hash_to_float(position_hash(pair_min, pair_max, 1, world_seed + 9999))
    h2 = hash_to_float(position_hash(pair_min, pair_max, 2, world_seed + 9999))
    h3 = hash_to_float(position_hash(pair_min, pair_max, 3, world_seed + 9999))

    # Scale pair distance to modulate curvature strength
    # (distant voices get more curvature, close voices get subtle curvature)
    # This uses a precomputed cosine similarity, but can also be approximated
    # from the voice index distance as a proxy
    idx_distance = abs(idx_a - idx_b) / max(len(ALL_VOICES) - 1, 1)
    distance_scale = 0.5 + idx_distance  # [0.5, 1.5]

    curvature = (h0 * 2.0 - 1.0) * 0.5 * distance_scale   # [-0.5, 0.5] scaled
    tilt = 0.8 + h1 * 0.4                                   # [0.8, 1.2]
    hbias_dim = int(h2 * 2816)                               # [0, 2816]
    hbias_strength = h3 * 0.02 * distance_scale              # [0.0, 0.02] scaled

    return {
        "curvature": curvature,
        "tilt": tilt,
        "hbias_dim": hbias_dim,
        "hbias_strength": hbias_strength,
    }
```

### Modified `voice_at()` — Quadratic Path

The existing `voice_at()` function in `zerovoice.py` is updated to use `quadratic_slerp` instead of plain `slerp`:

```python
def voice_at(x, y, z, voice_embeddings, world_seed=42):
    va, vb, vc = select_voices(x, y, z, world_seed)
    t_ab, t_abc = derive_blend_weights(x, y, z, world_seed)

    emb_a = voice_embeddings[va]
    emb_b = voice_embeddings[vb]
    emb_c = voice_embeddings[vc]

    # Derive pair-specific quadratic parameters
    params_ab = derive_pair_params(va, vb, world_seed)

    # First blend: A + B via quadratic SLERP
    a_aligned, b_aligned = align_lengths(emb_a, emb_b, strategy="resample")
    blend_ab = quadratic_slerp(
        a_aligned, b_aligned, t_ab,
        curvature=params_ab["curvature"],
        hbias_dim=params_ab["hbias_dim"],
        hbias_strength=params_ab["hbias_strength"],
        tilt=params_ab["tilt"],
    )

    # Second blend: AB + C (still standard SLERP -- tertiary tint is subtle)
    if t_abc > 0.01:
        ab_aligned, c_aligned = align_lengths(blend_ab, emb_c, strategy="resample")
        result = slerp(ab_aligned, c_aligned, t_abc)
    else:
        result = blend_ab

    return result
```

### Updated `voice_recipe()` — Include Pair Parameters

```python
def voice_recipe(x, y, z, world_seed=42):
    va, vb, vc = select_voices(x, y, z, world_seed)
    t_ab, t_abc = derive_blend_weights(x, y, z, world_seed)
    params_ab = derive_pair_params(va, vb, world_seed)
    return {
        "coordinate": (x, y, z),
        "world_seed": world_seed,
        "voice_a": va,
        "voice_b": vb,
        "voice_c": vc,
        "t_ab": round(t_ab, 4),
        "t_abc": round(t_abc, 4),
        "curvature": round(params_ab["curvature"], 4),
        "tilt": round(params_ab["tilt"], 4),
        "hbias_dim": params_ab["hbias_dim"],
        "hbias_strength": round(params_ab["hbias_strength"], 5),
        "canonical_name": voice_name(x, y, z, world_seed),
    }
```

---

## Similarity-Aware Voice Selection

### Assessment: Should We Prevent Cross-Family Blending?

**No.** The cosine similarity data shows:
- All 20 voices have pairwise cosine > 0.974 (very close in embedding space)
- Cross-family blends (e.g., hi_male + nl_male: 0.978) already produce coherent audio
- Preventing cross-family blending would dramatically reduce voice diversity

### What We Should Do Instead: Similarity-Modulated Curvature

Use measured similarity to **scale** the quadratic parameters:
- **High similarity pairs** (cos > 0.99): Subtle curvature, gentle tilt. The voices are already close -- heavy distortion would overwhelm the difference.
- **Moderate similarity pairs** (cos 0.98-0.99): Medium curvature. The path needs some character to distinguish the blend from a simple average.
- **Lower similarity pairs** (cos < 0.98): Higher curvature, more pronounced harmonic bias. The wider embedding distance gives room for creative path shaping.

This is already approximated in `derive_pair_params()` via the `idx_distance * distance_scale` factor. For a more precise version, we can precompute a 20x20 cosine similarity matrix at module load time (one-time cost, ~0.5ms) and use it directly.

### Optional: Similarity Affinity Matrix

```python
# Precomputed at module load time (one-time, from embeddings)
def compute_affinity_matrix(embeddings: dict[str, torch.Tensor]) -> dict[tuple[str, str], float]:
    """Compute pairwise cosine similarity between all voice mean-pool signatures."""
    import torch.nn.functional as F
    sigs = {name: emb.float().mean(dim=0) for name, emb in embeddings.items()}
    affinity = {}
    names = sorted(sigs.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                cos = F.cosine_similarity(sigs[a].unsqueeze(0), sigs[b].unsqueeze(0)).item()
                affinity[(a, b)] = cos
                affinity[(b, a)] = cos
    return affinity
```

This can optionally replace `idx_distance` in `derive_pair_params()` with the actual measured cosine distance for more precise curvature scaling.

---

## Implementation Steps

### Phase 1: Add Quadratic SLERP to `slerp_voices.py`

1. Add `compute_control_point()` function
2. Add `quadratic_slerp()` function
3. Keep existing `slerp()` unchanged (used for standard blends and the tertiary tint)
4. Unit test: verify `quadratic_slerp` with `curvature=0` equals standard `slerp`
5. Unit test: verify `quadratic_slerp` at t=0 returns A, t=1 returns B
6. Unit test: verify curvature > 0 and curvature < 0 produce different results
7. Unit test: verify determinism (same inputs -> same outputs)

### Phase 2: Add Pair Parameter Derivation to `zerovoice.py`

8. Add `VOICE_INDEX` mapping
9. Add `derive_pair_params()` function
10. Update `voice_at()` to call `quadratic_slerp` for the A-B blend
11. Update `voice_recipe()` to include curvature, tilt, hbias_dim, hbias_strength
12. Unit test: verify pair params are deterministic
13. Unit test: verify pair params are order-independent (A,B same as B,A)
14. Unit test: verify different pairs produce different params

### Phase 3: Deploy and Test

15. Copy updated `slerp_voices.py` and `zerovoice.py` to WSL2 site-packages
16. Restart vLLM server
17. Test TTS with existing `zv_` coordinates -- should produce subtly different audio than Stage 1
18. Compare adjacent coordinates for coherence (should still hold)
19. Run determinism verification across all 5 ZeroBytes laws

### Phase 4: Frontend Update

20. Update `zerovoice_frontend.py` to display new recipe fields (curvature, tilt, hbias_dim) in Voice Explorer
21. No new UI controls needed -- parameters are derived automatically from pair hashes

---

## Testing Scenarios

### Unit Tests

| Test | Input | Expected |
|---|---|---|
| Quadratic SLERP k=0 matches standard | `quadratic_slerp(a, b, 0.5, k=0, ...)` | Same as `slerp(a, b, 0.5)` within tolerance |
| Quadratic identity t=0 | `quadratic_slerp(a, b, 0.0, ...)` | Returns A |
| Quadratic identity t=1 | `quadratic_slerp(a, b, 1.0, ...)` | Returns B |
| Positive curvature differs from negative | `k=0.3` vs `k=-0.3` | Different results, same shape |
| Tilt > 1.0 boosts early rows | Compare row norms | First half has higher norms |
| Tilt < 1.0 boosts late rows | Compare row norms | Second half has higher norms |
| Pair params deterministic | Same pair, same seed, 10x | Identical params |
| Pair params order-independent | (A, B) vs (B, A) | Identical params |
| Different pairs, different params | (A, B) vs (A, C) | Different params |
| Pair distance scaling | Close pair vs distant pair | Distant has higher abs(curvature) |

### Integration Tests

| Test | Expected |
|---|---|
| TTS with `zv_50_30_10` | Valid 24kHz audio (may sound different from Stage 1) |
| Adjacent coord coherence | `zv_50_30_10` and `zv_51_30_10` still sound similar |
| Determinism | Same coordinate produces identical audio bytes |
| Recipe shows new fields | `curvature`, `tilt`, `hbias_dim` in recipe output |

---

## Performance Goals

| Metric | Target | Notes |
|---|---|---|
| `derive_pair_params()` | < 0.1ms | 4 hash calls |
| `compute_control_point()` | < 2ms | Vector math on [N, 3072] |
| `quadratic_slerp()` | < 15ms | 3x SLERP + tilt |
| Total `voice_at()` overhead vs Stage 1 | +5-10ms | Acceptable, still < 25ms total |
| Affinity matrix precompute | < 5ms | One-time at module load |

---

## File Changes

```
Modified files:
+-- slerp_voices.py     # ADD: compute_control_point(), quadratic_slerp()
+-- zerovoice.py         # ADD: VOICE_INDEX, derive_pair_params()
                         # MOD: voice_at() uses quadratic_slerp for A-B blend
                         # MOD: voice_recipe() includes pair params

No changes needed:
+-- voxtral_tts.py       # Server patch unchanged (calls voice_at transparently)
+-- serving_speech.py    # API unchanged
+-- zerovoice_frontend.py # Frontend shows new recipe fields automatically (JSON display)
```
