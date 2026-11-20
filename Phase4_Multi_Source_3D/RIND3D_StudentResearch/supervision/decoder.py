"""Reader for authenticated CBC/2 supervision. ZYX observations, XYZ posterior addresses."""
import base64
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import zstandard

GRID = 128
CELLS = GRID ** 3
TICKS = 1 << 52
CAP = Fraction(1, 20)
BUILD = "acc21857771fb64b0c012b0236d0d4b979d2751f"
BUILD_JSON_SHA256 = "0f4b04a10e674964a9b8713a44cd2a992aa31c2be2e5420de7af6b457bf19ed6"
FORMAT = "sparse-128-xyz/little-u32-binary64/dyadic52/v1"
WIRE = np.dtype([("id", "<u4"), ("weight", "<f8")])


def _verified_bytes(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"unexpected file digest: {path}")
    return raw


def _ticks(weights):
    if weights.ndim != 1 or weights.dtype not in (np.dtype("<f4"), np.dtype("<f8")):
        raise ValueError("expected a binary32 or binary64 vector")
    if not np.all((weights > 0) & (weights <= 1)):
        raise ValueError("nonpositive or nonfinite mass, or mass exceeds one")
    scaled = np.ldexp(weights.astype(np.float64, copy=False), 52)
    integer = scaled.astype(np.int64)
    if not np.array_equal(scaled, integer):
        raise ValueError("weight outside the certified dyadic lattice")
    return integer


def _immutable(arr):
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype)


@dataclass(frozen=True)
class SparseTarget:
    """Positive additive terms. Repeated IDs in float32_terms have the same class label."""
    cells: np.ndarray
    weights: np.ndarray
    tv_bound: Fraction


def _decode(wire, bound, representation):
    if not wire or len(wire) % WIRE.itemsize or len(wire) > CELLS * WIRE.itemsize:
        raise ValueError("invalid wire length")
    entries = np.frombuffer(wire, WIRE)
    ids = entries["id"].astype(np.int64)
    if ids[-1] >= CELLS or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("invalid cell addresses")
    weights = entries["weight"]
    ticks = _ticks(weights)
    # Up to 1024 positive terms per subtotal, each <=2^52. Sum subtotals as Python ints.
    if sum(int(ticks[i:i+1024].sum()) for i in range(0, len(ticks), 1024)) != TICKS:
        raise ValueError("wire is not exactly normalized")

    if representation == "binary64":
        return SparseTarget(_immutable(ids), _immutable(weights), bound)
    if representation != "float32_terms":
        raise ValueError("select binary64 or float32_terms")

    # A directly representable mass costs only one record. Otherwise split its
    # 52-bit tick count into at most three positive base-2^24 digits.
    nearest = weights.astype(np.float32)
    exact = nearest.astype(np.float64) == weights
    terms = np.zeros((len(ids), 3), np.float32)
    terms[exact, 0] = nearest[exact]
    split = ticks[~exact]
    for column, shift in enumerate((48, 24, 0)):
        digit = ((split >> shift) & 0xffffff).astype(np.float32)
        terms[~exact, column] = np.ldexp(digit, shift - 52)
    row, col = np.nonzero(terms)
    actual = terms[row, col]
    # Recheck the emitted numeric representation, including ownership of every digit.
    starts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
    if (not np.array_equal(row[starts], np.arange(len(ids))) or
            not np.array_equal(np.add.reduceat(_ticks(actual), starts), ticks)):
        raise ArithmeticError("float32 terms changed source-cell mass")
    return SparseTarget(_immutable(ids[row]), _immutable(actual), bound)


class IssuedExamples:
    """Strict complete-shard reader. Pins must come from an independently trusted handoff."""
    def __init__(self, campaign, scenes, index, *, manifest_sha256, index_sha256,
                 representation="binary64"):
        self._root = Path(campaign)
        self._scenes = Path(scenes)
        self._plan = json.loads(_verified_bytes(self._root / "manifest.json", manifest_sha256))
        self._issued = json.loads(_verified_bytes(index, index_sha256))
        self._manifest_sha = manifest_sha256
        self.representation = representation
        plan = self._plan
        build_bytes = (json.dumps(plan["build"], sort_keys=True, indent=2) + "\n").encode()
        if (hashlib.sha256(build_bytes).hexdigest() != BUILD_JSON_SHA256 or not self._issued or
                plan["schema"] != "cbc-campaign/1" or plan["build"]["commit"] != BUILD or
                plan["build"]["kernel"] != "astra-xlf-cbc/2" or
                plan["build"]["backend"] != plan["backend"] or plan["backend"] != "python-integer-fraction" or
                plan["tolerance"] != "1/20" or set(plan["tasks"]) != set(self._issued) or
                representation not in ("binary64", "float32_terms")):
            raise ValueError("unsupported build, representation or incomplete issuance")
        self.keys = tuple(sorted(self._issued))
        self._cache = (None, None)

    def __len__(self):
        return len(self.keys)

    def _artifact(self, key, ident):
        if key != f"{ident['slot']:06d}_w{ident['window']:02d}_s{ident['channel']}":
            raise ValueError("task key disagrees with physical identity")
        expected = self._issued[key]
        starts = sorted((self._root / "tasks" / key).glob("*/start.json"))
        if not starts:
            raise ValueError("missing issuance")
        for i, path in enumerate(starts):
            start = json.loads(path.read_bytes())
            rec = json.loads((path.parent / "receipt.json").read_bytes())
            if start["key"] != key or start["manifest_sha"] != self._manifest_sha:
                raise ValueError("attempt belongs to a different task")
            if i+1 == len(starts):
                if rec["status"] != "CERTIFIED" or rec["sha256"] != expected:
                    raise ValueError("missing authorized label")
            elif rec["status"] not in ("INCOMPLETE", "RESOURCE_INCOMPLETE", "INTERRUPTED"):
                raise ValueError("invalid or duplicated issuance")
        return json.loads(_verified_bytes(starts[-1].parent / "label.json", expected))

    def _observation(self, ident):
        slot, window, channel = (ident[k] for k in ("slot", "window", "channel"))
        cached_slot, cached = self._cache
        if cached_slot != slot:
            info = self._plan["files"][str(slot)]
            raw = _verified_bytes(self._scenes / info["path"], info["sha"])
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                cached = z["windows"], z["obs_index"], z["obs"].tobytes()
            self._cache = (slot, cached)
        windows, index, blob = cached
        row = windows[window]
        side = int(row["L"])
        origin = tuple(int(row[k]) for k in "xyz")
        if (side not in (64, 128, 256) or side != ident["side"] or list(origin) != ident["origin"]
                or ident["resolution"] != 1024 or not int(row["mask"]) & (1 << channel)):
            raise ValueError("window metadata mismatch")

        def crop(j):
            ((offset, size),) = [(int(o), int(n)) for w,s,o,n in index if w == window and s == j]
            if offset < 0 or size <= 0 or offset+size > len(blob):
                raise ValueError("invalid crop extent")
            packed = zstandard.ZstdDecompressor().decompress(blob[offset:offset+size],
                    max_output_size=side**3//8, allow_extra_data=False)
            if len(packed) != side**3//8:
                raise ValueError("invalid crop length")
            return np.unpackbits(np.frombuffer(packed,np.uint8)).view(np.bool_).reshape((side,)*3)

        solid, response = crop(-1), crop(channel)
        if np.any(solid & response):
            raise ValueError("response-active SOLID")
        # Hash X-major classes in slabs; returned model inputs retain storage order ZYX.
        digest = hashlib.sha256()
        for x in range(0, side, 8):
            classes = np.where(solid[:,:,x:x+8], np.uint8(2), response[:,:,x:x+8].astype(np.uint8))
            digest.update(classes.T.tobytes())
        if digest.hexdigest() != ident["observation_xyz_sha256"]:
            raise ValueError("local observation differs from issued task")
        return dict(solid=solid, response=response, L=side, origin=origin)

    def __getitem__(self, item):
        key = item if isinstance(item, str) else self.keys[item]
        ident = self._plan["tasks"][key]
        data = self._artifact(key, ident)
        statement = data["certificate"]["statement"]
        if (data["format"] != "astra-xlf-certified-interface/1" or data["status"] != "CERTIFIED"
                or data["solver"] != "astra-xlf-cbc/2" or data["problem"] != ident
                or statement["build"] != self._plan["build"] or statement["wire_format"] != FORMAT
                or statement["problem"] != {k:ident[k] for k in
                    ("halfspaces_sha256","origin","side","resolution","observation_xyz_sha256")}
                or statement["tolerance"] != data["tolerance"] or statement["tv"] != data["tv"]):
            raise ValueError("unsupported or mismatched scientific statement")
        def canonical_digest(value):
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            return hashlib.sha256(raw).hexdigest()

        seal = canonical_digest(statement)
        if (seal != data["certificate"]["statement_sha256"] or
                canonical_digest([ident, seal]) != data["artifact_sha256"] or
                data["tolerance"] != self._plan["tolerance"]):
            raise ValueError("changed scientific envelope")
        bound = Fraction(data["tv"])
        if not 0 <= bound <= Fraction(data["tolerance"]) <= CAP:
            raise ValueError("unapproved error bound")
        wire = base64.b64decode(data["wire"], validate=True)
        if (hashlib.sha256(wire).hexdigest() != data["wire_sha256"] or
                statement["wire_sha256"] != data["wire_sha256"] or len(wire) != WIRE.itemsize*data["cells"]):
            raise ValueError("changed sparse posterior")
        return self._observation(ident), _decode(wire, bound, self.representation)


def torch_batch(targets):
    """Return (batch index, flat XYZ cell ID, additive weight) for a sparse CE consumer."""
    import torch
    if not targets or any(t.weights.dtype != targets[0].weights.dtype for t in targets):
        raise ValueError("expected nonempty targets of a common precision")
    batch = np.repeat(np.arange(len(targets), dtype=np.int64), [len(t.cells) for t in targets])
    return (torch.from_numpy(batch), torch.from_numpy(np.concatenate([t.cells for t in targets])),
            torch.from_numpy(np.concatenate([t.weights for t in targets])))


def sparse_cross_entropy(log_prob_xyz, targets):
    """Per-example reference loss on CPU. TV certification covers target coefficients."""
    import torch
    if (log_prob_xyz.device.type != "cpu" or not log_prob_xyz.is_floating_point() or
            tuple(log_prob_xyz.shape) not in ((len(targets), CELLS), (len(targets), GRID, GRID, GRID))):
        raise ValueError("expected floating CPU log probabilities on the full XYZ grid")
    batch, cell, weight = torch_batch(targets)
    selected = log_prob_xyz.reshape(len(targets), CELLS)[batch, cell].double()
    return torch.zeros(len(targets), dtype=torch.float64).index_add_(0, batch, -weight.double()*selected)
