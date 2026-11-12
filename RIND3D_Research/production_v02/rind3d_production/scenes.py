"""Scene construction: certified convex geometry, legal sources, profile identities.

Geometry: rind3d.lite_generators.generate (nine certified convex-union families, unchanged recipes).
Sources: independent uniform proposals in the open world with the certified FREE test and the certified
clearance proof (radius DRIVER_RADIUS = 2^-26, the v0.1 rule); multi-source profiles additionally require
pairwise separation >= MIN_SEPARATION and draw binary64 strengths uniformly in STRENGTH_RANGE.
Identities: scene_id = SHA256 over (profile, geometry_id, source float64 bytes, strength float64 bytes).
"""
import hashlib

import numpy as np

from rind3d.geometry import keyed_rng
from rind3d.lite_acceptance import DRIVER_RADIUS, geometry_acceptance, verify_clearance
from rind3d.lite_generators import FAMILIES, generate
from rind3d.rendering.context import GeometryContext

from .env import POLICY

MIN_SEPARATION = 1/32
STRENGTH_RANGE = (0.1, 1.0)


def geometry(seed, slot, family, attempt=0):
    bundle, recipe = generate(seed, slot, family, attempt)
    if recipe["proposal_failures"]:
        raise ValueError("geometry world-fitting proposal failed")
    return bundle, recipe


def context_for(bundle):
    return GeometryContext(bundle, POLICY)


def legal_source(context, p):
    """Certified FREE + clearance GT (the v0.1 driver rule). Returns the clearance proof or None."""
    p = np.asarray(p, dtype="<f8").reshape(3)
    if not np.all((p > 0) & (p < 1)):
        return None
    if context.contains(p[None])[0]:
        return None
    proof = context.clearance(p, DRIVER_RADIUS)
    if proof["status"] != "GT":
        return None
    verify_clearance(context, p, proof)
    return proof


def sample_sources(context, seed, slot, K, attempt=0, limit=512):
    """K legal sources (stream distinct from geometry), pairwise separation, binary64 strengths."""
    rng = keyed_rng(seed, slot, 20011 + attempt*37)
    sources, proofs, rejections = [], [], {"outside": 0, "solid_or_clearance": 0, "separation": 0}
    proposals = 0
    while len(sources) < K:
        if proposals >= limit:
            raise RuntimeError("source proposal budget exhausted (incomplete work, not a physical rejection)")
        proposals += 1
        p = rng.random(3)
        if not np.all((p > 0) & (p < 1)):
            rejections["outside"] += 1
            continue
        if any(np.linalg.norm(p - q) < MIN_SEPARATION for q in sources):
            rejections["separation"] += 1
            continue
        proof = legal_source(context, p)
        if proof is None:
            rejections["solid_or_clearance"] += 1
            continue
        sources.append(p)
        proofs.append(proof)
    strengths = np.ones(K) if K == 1 else rng.uniform(*STRENGTH_RANGE, K)
    return np.array(sources, dtype="<f8"), np.asarray(strengths, dtype="<f8"), {
        "proposals": proposals, "rejections": rejections, "clearance": proofs}


def scene_id(profile, geometry_id, sources, strengths):
    h = hashlib.sha256()
    h.update(profile.encode())
    h.update(bytes.fromhex(geometry_id))
    h.update(np.asarray(sources, dtype="<f8").tobytes())
    h.update(np.asarray(strengths, dtype="<f8").tobytes())
    return h.hexdigest()


def hypothetical_sources(context, rng, n, limit=None):
    """Teacher-style candidate sources: uniform world proposals that pass the same legality tests."""
    out, tried = [], 0
    limit = limit or 64*n
    while len(out) < n and tried < limit:
        tried += 1
        p = rng.random(3)
        if legal_source(context, p) is not None:
            out.append(p)
    return np.array(out, dtype="<f8"), tried


__all__ = ["FAMILIES", "geometry", "context_for", "sample_sources", "scene_id", "hypothetical_sources",
           "geometry_acceptance", "legal_source", "DRIVER_RADIUS"]
