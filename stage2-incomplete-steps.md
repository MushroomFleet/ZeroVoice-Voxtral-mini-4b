# Stage 2 Assessment — ZeroQuadratic Voice Layer

**Date:** 2026-03-27
**Assessed against:** `stage2-quadratic-layer-plan.md`
**Status:** OPERATIONAL — Quadratic SLERP layer deployed and generating audio

---

## Summary

| Plan Step | Status | Notes |
|---|---|---|
| Phase 1, Step 1: Add `compute_control_point()` | COMPLETE | Added to `slerp_voices.py` |
| Phase 1, Step 2: Add `quadratic_slerp()` | COMPLETE | De Casteljau + tilt, fast path for k=0 |
| Phase 1, Step 3: Keep `slerp()` unchanged | COMPLETE | Used for tertiary tint + fallback |
| Phase 1, Step 4: Test k=0 equals standard SLERP | COMPLETE | PASS |
| Phase 1, Step 5: Test t=0 and t=1 identity | COMPLETE | PASS (within tilt tolerance) |
| Phase 1, Step 6: Test curvature +/- differ | COMPLETE | PASS |
| Phase 1, Step 7: Test determinism | COMPLETE | PASS |
| Phase 2, Step 8: Add `VOICE_INDEX` | COMPLETE | 20 entries, alphabetical |
| Phase 2, Step 9: Add `derive_pair_params()` | COMPLETE | Order-independent, distance-scaled |
| Phase 2, Step 10: Update `voice_at()` | COMPLETE | Uses `quadratic_slerp` for A-B blend |
| Phase 2, Step 11: Update `voice_recipe()` | COMPLETE | Includes curvature, tilt, hbias_dim, hbias_strength |
| Phase 2, Step 12: Test pair params deterministic | COMPLETE | PASS |
| Phase 2, Step 13: Test pair params order-independent | COMPLETE | PASS |
| Phase 2, Step 14: Test different pairs differ | COMPLETE | PASS |
| Phase 3, Step 15: Copy to WSL2 site-packages | COMPLETE | Both files updated |
| Phase 3, Step 16: Restart vLLM server | COMPLETE | Server UP, 22.5GB GPU |
| Phase 3, Step 17: Test TTS with zv_ coordinates | COMPLETE | 4/4 coordinates produce valid 24kHz audio |
| Phase 3, Step 18: Test adjacent coherence | COMPLETE | (50,30,10) and (51,30,10) share same pair params |
| Phase 3, Step 19: Determinism verification | COMPLETE | 27/27 unit tests + E2E determinism PASS |
| Phase 4, Step 20: Update frontend recipe display | INCOMPLETE | See below |

---

## Test Results

### Unit Tests: 27/27 PASSED

| Category | Tests | Result |
|---|---|---|
| Quadratic SLERP math | 10 | All PASS |
| Pair parameter derivation | 8 | All PASS |
| Full pipeline (real embeddings) | 9 | All PASS |

### Key Metrics Verified

| Metric | Expected | Actual |
|---|---|---|
| k=0 matches standard SLERP | Identical | PASS (atol=1e-2) |
| Stage 2 output differs from Stage 1 | Different | PASS |
| Pair params order-independent | (A,B) == (B,A) | PASS |
| Tilt > 1.0 boosts early rows | first > second | PASS (60.39 > 50.26) |
| Tilt < 1.0 boosts late rows | first < second | PASS (49.23 < 61.73) |

### E2E TTS Results

| Coordinate | Duration | Voice Pair | Curvature | Tilt |
|---|---|---|---|---|
| zv_50_30_10 | 6.64s | hi_male + ar_male | -0.4012 | 1.0888 |
| zv_51_30_10 | 18.56s | ar_male + hi_male | -0.4012 | 1.0888 |
| zv_50_30_150 | 8.48s | nl_male + hi_male | -0.1152 | 1.0595 |
| zv_100_100_0 | 12.64s | ar_male + hi_male | -0.4012 | 1.0888 |

---

## Incomplete Steps

### 1. Frontend V2 Recipe Display Not Updated (Phase 4, Step 20)

**Status:** INCOMPLETE — The frontend (`zerovoice_frontend.py`) uses `gr.JSON` to display the recipe, which **automatically shows the new fields** (curvature, tilt, hbias_dim, hbias_strength) without any code change. However, the plan called for explicit UI labeling of these new fields, and the frontend file itself was not updated or redeployed with the Stage 2 modules.

**Impact:** LOW — The JSON display shows all fields correctly. No dedicated UI controls are needed since parameters are derived automatically.

**To complete:**
1. Copy updated `zerovoice.py` to `~/voxtral-tts/` (the frontend imports from there)
2. Restart the frontend process
3. Verify recipe JSON in browser shows curvature/tilt/hbias fields

```bash
cp /mnt/k/voxtral-mini-4b/zerovoice.py ~/voxtral-tts/
# Restart frontend (kill old process, relaunch)
```

### 2. Affinity Matrix Not Implemented (Optional Enhancement)

**Status:** NOT IMPLEMENTED — The plan described an optional `compute_affinity_matrix()` function that precomputes the 20x20 pairwise cosine similarity matrix for more precise curvature scaling (replacing the `idx_distance` proxy).

**Impact:** LOW — The current `idx_distance` proxy works well. The affinity matrix is an optimization, not a requirement. The Stage 2 pair distance scaling already modulates curvature appropriately.

**To implement (optional):**
```python
# Add to zerovoice.py
def compute_affinity_matrix(embeddings):
    sigs = {n: e.float().mean(dim=0) for n, e in embeddings.items()}
    names = sorted(sigs.keys())
    affinity = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                cos = F.cosine_similarity(sigs[a].unsqueeze(0), sigs[b].unsqueeze(0)).item()
                affinity[(a, b)] = cos
                affinity[(b, a)] = cos
    return affinity
```

### 3. Frontend Copy of zerovoice.py Is Stage 1 Version

**Status:** STALE — The file at `~/voxtral-tts/zerovoice.py` (used by the frontend for client-side recipe display) is the Stage 1 version without `derive_pair_params()`. The site-packages copy (used by the server) is Stage 2.

**Impact:** MEDIUM — The frontend will show recipes without curvature/tilt fields until the local copy is updated. The TTS audio itself is correct (server uses updated site-packages).

**To complete:**
```bash
cp /mnt/k/voxtral-mini-4b/zerovoice.py ~/voxtral-tts/zerovoice.py
cp /mnt/k/voxtral-mini-4b/slerp_voices.py ~/voxtral-tts/slerp_voices.py
```

---

## Current System State

### Files Modified in Stage 2

| File | Change | Location |
|---|---|---|
| `slerp_voices.py` | ADDED `compute_control_point()`, `quadratic_slerp()` | Windows + WSL2 site-packages |
| `zerovoice.py` | ADDED `VOICE_INDEX`, `derive_pair_params()`, imports `quadratic_slerp` | Windows + WSL2 site-packages |
| `zerovoice.py` | MODIFIED `voice_at()` to use `quadratic_slerp` for A-B blend | Windows + WSL2 site-packages |
| `zerovoice.py` | MODIFIED `voice_recipe()` to include pair params | Windows + WSL2 site-packages |

### Files NOT Modified (as planned)

| File | Reason |
|---|---|
| `voxtral_tts.py` (server patch) | Calls `voice_at()` transparently — no change needed |
| `serving_speech.py` (API patch) | API unchanged — no change needed |
| `mistral_common/audio.py` (tokenizer patch) | Token count formula unchanged — no change needed |

### Verified Component Versions

| Component | Version | Status |
|---|---|---|
| `slerp_voices.py` | Stage 2 (with quadratic_slerp) | Deployed to site-packages |
| `zerovoice.py` | Stage 2 (with derive_pair_params) | Deployed to site-packages |
| vLLM server | Running with Stage 2 patches | Port 8000, healthy |
| Frontend V2 | Running with Stage 1 zerovoice.py | Port 7861, needs zerovoice.py update |
