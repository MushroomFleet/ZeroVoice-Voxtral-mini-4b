"""Core SLERP math and length-alignment utilities for Voxtral TTS voice embeddings.

Voice embeddings are 2D tensors of shape [N, 3072] where N varies per voice (67-218).
SLERP is applied row-wise across the 3072-dim hidden space after length alignment.

Dependencies: torch only. No vLLM or vllm-omni imports.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def align_lengths(
    a: torch.Tensor, b: torch.Tensor, strategy: str = "resample"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two voice embeddings to the same sequence length.

    Args:
        a: tensor of shape [Na, D]
        b: tensor of shape [Nb, D]
        strategy: "resample" (interpolate shorter to longer),
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
            a = F.pad(a, (0, 0, 0, N - Na))
        if Nb < N:
            b = F.pad(b, (0, 0, 0, N - Nb))
        return a, b

    elif strategy == "resample":
        N = max(Na, Nb)
        if Na < N:
            a = _resample_seq(a, N)
        if Nb < N:
            b = _resample_seq(b, N)
        return a, b

    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")


def _resample_seq(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample a [N, D] tensor to [target_len, D] via linear interpolation."""
    orig_dtype = t.dtype
    # interpolate needs [batch, channels, length]
    t = t.float().unsqueeze(0).permute(0, 2, 1)  # [1, D, N]
    t = F.interpolate(t, size=target_len, mode="linear", align_corners=True)
    return t.permute(0, 2, 1).squeeze(0).to(orig_dtype)  # [target_len, D]


def align_to_target(
    a: torch.Tensor, b: torch.Tensor, target_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align both tensors to a specific target length.

    Unlike align_lengths (which uses max), this resamples both to target_n.
    A tensor already at target_n is returned unchanged.
    """
    if a.shape[0] != target_n:
        a = _resample_seq(a, target_n)
    if b.shape[0] != target_n:
        b = _resample_seq(b, target_n)
    return a, b


def calibrate_norms(result: torch.Tensor, target_mean_norm: float = 4.48) -> torch.Tensor:
    """Rescale rows so the mean row norm matches the preset average.

    Preset voices have mean row norms of 4.39-4.58 (mean ~4.48).
    Blended embeddings drift from this, causing generation issues.
    """
    orig_dtype = result.dtype
    f = result.float()
    current_norms = f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    current_mean = current_norms.mean()
    if current_mean > 1e-8:
        scale = target_mean_norm / current_mean
        f = f * scale
    return f.to(orig_dtype)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical Linear Interpolation between two tensors, row-wise.

    For tensors of shape [N, D], each of the N rows is independently
    interpolated on the D-dimensional unit hypersphere, then rescaled
    to preserve interpolated magnitude.

    Args:
        a: tensor of shape [N, D] (already length-aligned)
        b: tensor of shape [N, D] (already length-aligned)
        t: interpolation weight in [0.0, 1.0]. t=0 -> a, t=1 -> b.

    Returns:
        Interpolated tensor of shape [N, D], same dtype as input a.
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    orig_dtype = a.dtype

    a_f = a.float()
    b_f = b.float()

    # Normalize each row to unit length
    a_norm = F.normalize(a_f, dim=-1)
    b_norm = F.normalize(b_f, dim=-1)

    # Per-row cosine similarity
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True)  # [N, 1]
    dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    omega = torch.acos(dot)  # [N, 1]
    sin_omega = torch.sin(omega)  # [N, 1]

    # Detect near-parallel rows for LERP fallback
    use_lerp = (sin_omega.abs() < 1e-6).squeeze(-1)  # [N]

    # SLERP weights
    weight_a = torch.sin((1.0 - t) * omega) / sin_omega  # [N, 1]
    weight_b = torch.sin(t * omega) / sin_omega  # [N, 1]
    result = weight_a * a_f + weight_b * b_f  # [N, D]

    # LERP fallback for near-parallel rows
    if use_lerp.any():
        lerp_result = (1.0 - t) * a_f + t * b_f
        result[use_lerp] = lerp_result[use_lerp]

    # Restore interpolated magnitude
    a_mag = a_f.norm(dim=-1, keepdim=True)
    b_mag = b_f.norm(dim=-1, keepdim=True)
    target_mag = (1.0 - t) * a_mag + t * b_mag
    result_mag = result.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result * (target_mag / result_mag)

    return result.to(orig_dtype)


def blend_voices(
    voice_a_path: str,
    voice_b_path: str,
    t: float,
    alignment: str = "resample",
) -> torch.Tensor:
    """Load two voice embeddings and produce a SLERP-blended result.

    Args:
        voice_a_path: path to first .pt file
        voice_b_path: path to second .pt file
        t: interpolation weight [0.0, 1.0]
        alignment: length alignment strategy

    Returns:
        Blended tensor of shape [N_aligned, 3072], dtype bfloat16
    """
    t = max(0.0, min(1.0, t))

    a = torch.load(voice_a_path, map_location="cpu", weights_only=True)
    b = torch.load(voice_b_path, map_location="cpu", weights_only=True)

    a_aligned, b_aligned = align_lengths(a, b, strategy=alignment)
    return slerp(a_aligned, b_aligned, t)


def list_voices(voice_dir: str) -> dict[str, tuple[int, ...]]:
    """List available voice .pt files and their shapes.

    Returns:
        Dict mapping voice name (stem) to tensor shape tuple.
    """
    result = {}
    for pt_file in sorted(Path(voice_dir).glob("*.pt")):
        t = torch.load(pt_file, map_location="cpu", weights_only=True)
        result[pt_file.stem] = tuple(t.shape)
    return result


def voice_name_from_blend(voice_a: str, voice_b: str, t: float) -> str:
    """Generate canonical name for a SLERP blend."""
    return f"slerp_{voice_a}_{voice_b}_{t:.2f}"


# ---------------------------------------------------------------------------
# Quadratic SLERP (ZeroQuadratic layer)
# ---------------------------------------------------------------------------

def compute_control_point(
    a: torch.Tensor,
    b: torch.Tensor,
    curvature: float,
    hbias_dim: int,
    hbias_strength: float,
) -> torch.Tensor:
    """Compute the quadratic Bezier control point M for a voice pair.

    M is the geodesic midpoint displaced by curvature perpendicular to the A-B
    direction, plus a harmonic bias in a hash-selected 256-dim subspace.

    Args:
        a, b: [N, D] aligned voice embeddings (float32)
        curvature: how far M bows from the geodesic [-0.5, 0.5]. 0 = standard SLERP.
        hbias_dim: starting dimension for the 256-dim harmonic bias band
        hbias_strength: magnitude of the harmonic bias [0.0, 0.02]

    Returns:
        M: [N, D] control point tensor (float32)
    """
    midpoint = slerp(a, b, 0.5)

    if abs(curvature) < 1e-6 and hbias_strength < 1e-6:
        return midpoint

    D = a.shape[1]

    # Perpendicular direction to the A-B arc at each row
    ab_dir = F.normalize(b - a, dim=-1)
    ref = F.normalize(a + b, dim=-1)
    # Gram-Schmidt: remove the ab_dir component from ref
    perp = ref - (ref * ab_dir).sum(dim=-1, keepdim=True) * ab_dir
    perp_norm = perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    perp = perp / perp_norm

    # Displace midpoint along perpendicular by curvature * midpoint magnitude
    midpoint_mag = midpoint.norm(dim=-1, keepdim=True)
    M = midpoint + curvature * midpoint_mag * perp

    # Harmonic bias: boost a 256-dim band
    if hbias_strength > 1e-6:
        band_start = hbias_dim % max(D - 256, 1)
        band_end = band_start + 256
        M[:, band_start:band_end] = M[:, band_start:band_end] + hbias_strength * midpoint_mag

    return M


def quadratic_slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
    curvature: float = 0.0,
    hbias_dim: int = 0,
    hbias_strength: float = 0.0,
    tilt: float = 1.0,
) -> torch.Tensor:
    """Quadratic Bezier SLERP on the hypersphere via De Casteljau's algorithm.

    result = SLERP(SLERP(A, M, t), SLERP(M, B, t), t)
    where M is the pair-specific control point.

    When curvature=0, hbias_strength=0, tilt=1.0, this reduces to standard SLERP.

    Args:
        a, b: [N, D] aligned voice embeddings (bfloat16 or float)
        t: interpolation weight [0, 1]
        curvature: arc curvature from pair hash [-0.5, 0.5]
        hbias_dim: harmonic bias band start dimension
        hbias_strength: harmonic bias magnitude [0.0, 0.02]
        tilt: spectral tilt [0.8, 1.2]. >1 boosts early rows, <1 boosts late rows.

    Returns:
        [N, D] interpolated embedding, same dtype as input a.
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    orig_dtype = a.dtype
    a_f = a.float()
    b_f = b.float()

    # Fast path: no quadratic modification -> standard SLERP
    if abs(curvature) < 1e-6 and hbias_strength < 1e-6 and abs(tilt - 1.0) < 1e-6:
        return slerp(a, b, t)

    # Compute control point
    M = compute_control_point(a_f, b_f, curvature, hbias_dim, hbias_strength)

    # De Casteljau on sphere: two-level SLERP
    am = slerp(a_f, M, t)
    mb = slerp(M, b_f, t)
    result = slerp(am, mb, t)

    # Apply spectral tilt: scale row magnitudes by sequence position
    N = result.shape[0]
    if N > 1 and abs(tilt - 1.0) > 1e-6:
        positions = torch.linspace(0.0, 1.0, N, device=result.device)
        tilt_curve = tilt ** (1.0 - 2.0 * positions)  # [N]
        result = result * tilt_curve.unsqueeze(1)

    return result.to(orig_dtype)
