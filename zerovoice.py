"""ZeroVoice: Procedural voice space for Voxtral TTS.

Maps 3D integer coordinates (x, y, z) to deterministic voice embeddings
via position hashing, coherent noise, and cross-family SLERP blending.

Five ZeroBytes Laws:
  1. O(1) access — no iteration over other coordinates
  2. Parallelism — each coordinate independent
  3. Coherence — nearby coordinates sound similar
  4. Hierarchy — Z selects family, X/Y explore within
  5. Determinism — same inputs -> same outputs everywhere

Voice pairing strategy:
  A (primary) and B (secondary tint) are always from DIFFERENT families.
  Z < 100:  A = english,       B = european
  Z 100-199: A = european,     B = english
  Z >= 200:  A = asian_arabic, B = european

Dependencies: torch, xxhash, slerp_voices
"""

import struct

import torch
import xxhash

from slerp_voices import align_lengths, slerp, calibrate_norms

# ---------------------------------------------------------------------------
# Voice registry
# ---------------------------------------------------------------------------

ALL_VOICES = [
    "casual_female", "casual_male", "cheerful_female", "neutral_female", "neutral_male",
    "fr_male", "fr_female", "es_male", "es_female", "de_male", "de_female",
    "it_male", "it_female", "pt_male", "pt_female", "nl_male", "nl_female",
    "ar_male", "hi_male", "hi_female",
]

VOICE_FAMILIES = {
    "english": ["casual_male", "casual_female", "cheerful_female", "neutral_male", "neutral_female"],
    "european": [
        "fr_male", "fr_female", "es_male", "es_female", "de_male", "de_female",
        "it_male", "it_female", "pt_male", "pt_female", "nl_male", "nl_female",
    ],
    "asian_arabic": ["ar_male", "hi_male", "hi_female"],
}

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

# ---------------------------------------------------------------------------
# Core hashing (ZeroBytes primitives)
# ---------------------------------------------------------------------------

def position_hash(x: int, y: int, z: int, salt: int = 0) -> int:
    """Deterministic 64-bit hash from 3D integer coordinates."""
    h = xxhash.xxh64(seed=salt)
    h.update(struct.pack('<qqq', x, y, z))
    return h.intdigest()


def hash_to_float(h: int) -> float:
    """Convert a 64-bit hash to a float in [0.0, 1.0)."""
    return (h & 0xFFFFFFFF) / 0x100000000


# ---------------------------------------------------------------------------
# Coherent noise
# ---------------------------------------------------------------------------

def coherent_value(x: float, y: float, seed: int, octaves: int = 3) -> float:
    """Multi-octave coherent value noise in [-1, 1]."""
    value = 0.0
    amp = 1.0
    freq = 1.0
    max_amp = 0.0
    for i in range(octaves):
        xf = x * freq
        yf = y * freq
        x0 = int(xf) if xf >= 0 else int(xf) - 1
        y0 = int(yf) if yf >= 0 else int(yf) - 1
        sx = xf - x0
        sx = sx * sx * (3 - 2 * sx)
        sy = yf - y0
        sy = sy * sy * (3 - 2 * sy)
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


# ---------------------------------------------------------------------------
# Voice selection — cross-family pairing
# ---------------------------------------------------------------------------

def select_voices(
    x: int, y: int, z: int, world_seed: int, gender: str = "mixed"
) -> tuple[str, str]:
    """Select voice pair with A and B from different families.

    Z-axis determines the primary family for A:
      z < 100:   A from english (5),       B from european (12)
      100-199:   A from european (12),      B from english (5)
      z >= 200:  A from asian_arabic (3),   B from european (12)

    Cross-family pairing is guaranteed (A and B never from same family).

    Args:
        gender: "mixed" (any), "male", "female"

    Returns:
        (voice_a, voice_b)
    """
    if z < 100:
        pool_a = list(VOICE_FAMILIES["english"])
        pool_b = list(VOICE_FAMILIES["european"])
    elif z < 200:
        pool_a = list(VOICE_FAMILIES["european"])
        pool_b = list(VOICE_FAMILIES["english"])
    else:
        pool_a = list(VOICE_FAMILIES["asian_arabic"])
        pool_b = list(VOICE_FAMILIES["european"])

    # Apply gender filter
    if gender != "mixed":
        fa = [v for v in pool_a if VOICE_GENDER.get(v) == gender]
        fb = [v for v in pool_b if VOICE_GENDER.get(v) == gender]
        if fa:
            pool_a = fa
        if fb:
            pool_b = fb

    h_a = position_hash(x, y, z, world_seed + 1)
    h_b = position_hash(x, y, z, world_seed + 2)

    voice_a = pool_a[h_a % len(pool_a)]
    voice_b = pool_b[h_b % len(pool_b)]

    return voice_a, voice_b


def derive_blend_weight(x: int, y: int, z: int, world_seed: int) -> float:
    """Derive a single SLERP weight from coordinates using coherent noise.

    Returns t in [0.0, 0.20], capped to preserve voice_a reference coherence.
    """
    t_raw = coherent_value(x * 0.02, y * 0.02, world_seed + 5000, octaves=3)
    t = (t_raw + 1.0) / 2.0 * 0.20
    return max(0.0, min(0.20, t))


# ---------------------------------------------------------------------------
# Voice recipe (metadata without computing embedding)
# ---------------------------------------------------------------------------

def voice_recipe(x: int, y: int, z: int, world_seed: int = 42) -> dict:
    """Return the recipe for a coordinate without computing the embedding."""
    va, vb = select_voices(x, y, z, world_seed)
    t = derive_blend_weight(x, y, z, world_seed)
    return {
        "coordinate": (x, y, z),
        "world_seed": world_seed,
        "voice_a": va,
        "voice_b": vb,
        "t": round(t, 4),
        "canonical_name": voice_name(x, y, z, world_seed),
    }


# ---------------------------------------------------------------------------
# Core: coordinate -> embedding
# ---------------------------------------------------------------------------

def voice_at(
    x: int, y: int, z: int,
    voice_embeddings: dict[str, torch.Tensor],
    world_seed: int = 42,
) -> torch.Tensor:
    """Compute the voice embedding for a 3D coordinate.

    Single cross-family SLERP blend + norm calibration.
    O(1), deterministic, coherent with adjacent coordinates.
    """
    va, vb = select_voices(x, y, z, world_seed)
    t = derive_blend_weight(x, y, z, world_seed)

    emb_a = voice_embeddings[va]
    emb_b = voice_embeddings[vb]

    a_aligned, b_aligned = align_lengths(emb_a, emb_b, strategy="resample")
    result = slerp(a_aligned, b_aligned, t)
    result = calibrate_norms(result, target_mean_norm=4.48)

    return result


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def voice_name(x: int, y: int, z: int, world_seed: int = 42) -> str:
    """Canonical voice name from coordinates."""
    if world_seed == 42:
        return f"zv_{x}_{y}_{z}"
    return f"zv_{x}_{y}_{z}_s{world_seed}"


def parse_voice_name(name: str) -> tuple[int, int, int, int]:
    """Parse 'zv_X_Y_Z' or 'zv_X_Y_Z_sSEED' into (x, y, z, world_seed)."""
    rest = name[3:]  # strip "zv_"
    segments = rest.split("_")
    nums = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.startswith("s") and len(nums) == 3:
            break
        nums.append(int(seg))
        i += 1

    x, y, z = nums[0], nums[1], nums[2]
    seed = 42
    if i < len(segments) and segments[i].startswith("s"):
        seed = int(segments[i][1:])
    return x, y, z, seed


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_preset_embeddings(voice_dir: str) -> dict[str, torch.Tensor]:
    """Load all 20 preset voice embeddings from a directory."""
    from pathlib import Path
    embeddings = {}
    for name in ALL_VOICES:
        path = Path(voice_dir) / f"{name}.pt"
        if path.exists():
            embeddings[name] = torch.load(path, map_location="cpu", weights_only=True)
    return embeddings
