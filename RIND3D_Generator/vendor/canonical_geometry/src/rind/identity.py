"""Layout-independent, versioned binary framing for semantic content IDs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from struct import pack
from typing import Iterable

import numpy as np


HASH_FAMILY = "rind-content-sha256-v2"
FRAMING_ID = "rind-frame-le-1"
MAGIC = b"RIND-CONTENT\x00"
MAX_RANK = 8


class DTypeTag(IntEnum):
    FLOAT64 = 1
    UINT8 = 2
    UINT16 = 3
    UINT32 = 4
    UINT64 = 5
    INT64 = 6
    UTF8 = 7
    OPAQUE_BYTES = 8


DTYPES = {
    DTypeTag.FLOAT64: np.dtype("<f8"),
    DTypeTag.UINT8: np.dtype("u1"),
    DTypeTag.UINT16: np.dtype("<u2"),
    DTypeTag.UINT32: np.dtype("<u4"),
    DTypeTag.UINT64: np.dtype("<u8"),
    DTypeTag.INT64: np.dtype("<i8"),
}


def _ascii(value: str, *, what: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{what} must be ASCII") from error
    if not encoded or len(encoded) >= 2**32:
        raise ValueError(f"{what} has invalid byte length")
    return encoded


def _prefixed(value: bytes) -> bytes:
    if len(value) >= 2**32:
        raise ValueError("string too large for framing")
    return pack("<I", len(value)) + value


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    tag: DTypeTag
    shape: tuple[int, ...]
    payload: bytes

    def __post_init__(self) -> None:
        _ascii(self.name, what="field name")
        if len(self.shape) > MAX_RANK:
            raise ValueError("field rank exceeds framing limit")
        if any(not isinstance(size, int) or size < 0 or size >= 2**64 for size in self.shape):
            raise ValueError("invalid field shape")
        if self.tag in DTYPES:
            item_count = 1
            for size in self.shape:
                item_count *= size
            if item_count * DTYPES[self.tag].itemsize != len(self.payload):
                raise ValueError(f"payload size does not match {self.name} shape")
            if self.tag == DTypeTag.FLOAT64:
                values = np.frombuffer(self.payload, dtype="<f8")
                if not np.isfinite(values).all():
                    raise ValueError(f"{self.name} contains NaN or infinity")
                bits = np.frombuffer(self.payload, dtype="<u8")
                if np.any(bits == 0x8000000000000000):
                    raise ValueError(f"{self.name} contains noncanonical negative zero")
        elif self.tag in (DTypeTag.UTF8, DTypeTag.OPAQUE_BYTES):
            if self.shape != (len(self.payload),):
                raise ValueError("byte/string fields must be rank one with byte-count shape")
            if self.tag == DTypeTag.UTF8:
                self.payload.decode("utf-8", errors="strict")
        else:
            raise ValueError(f"unknown dtype tag {self.tag}")

    @classmethod
    def numeric(cls, name: str, tag: DTypeTag, values: object) -> Field:
        if tag not in DTYPES:
            raise ValueError("numeric constructor requires a numeric dtype")
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain numeric values")
        if tag == DTypeTag.FLOAT64:
            if raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
                raise TypeError("official real parameters cannot be narrowed from extended precision")
            array = np.asarray(raw, dtype="<f8").copy(order="C")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinity")
            array[array == 0] = 0.0
        else:
            target = DTYPES[tag]
            if raw.dtype.kind == "f":
                raise TypeError(f"integer field {name} cannot be made from floats")
            # Casting in NumPy can wrap integers, so compare every value before it.
            info = np.iinfo(target)
            if any(int(value) < info.min or int(value) > info.max for value in raw.flat):
                raise OverflowError(f"integer field {name} exceeds {target}")
            array = np.asarray(raw, dtype=target).copy(order="C")
        return cls(name, tag, tuple(int(n) for n in array.shape), array.tobytes(order="C"))

    @classmethod
    def text(cls, name: str, value: str) -> Field:
        data = value.encode("utf-8")
        return cls(name, DTypeTag.UTF8, (len(data),), data)

    @classmethod
    def opaque(cls, name: str, value: bytes) -> Field:
        data = bytes(value)
        return cls(name, DTypeTag.OPAQUE_BYTES, (len(data),), data)

    def as_array(self) -> np.ndarray:
        if self.tag not in DTYPES:
            raise TypeError(f"{self.name} is not a numeric field")
        result = np.frombuffer(self.payload, dtype=DTYPES[self.tag]).reshape(self.shape)
        result.flags.writeable = False
        return result


def frame(record_type: str, fields: Iterable[Field]) -> bytes:
    record_bytes = _ascii(record_type, what="record type")
    ordered = sorted(tuple(fields), key=lambda field: field.name.encode("ascii"))
    if len(ordered) >= 2**32:
        raise ValueError("too many semantic fields")
    if len({field.name for field in ordered}) != len(ordered):
        raise ValueError("duplicate semantic field name")

    output = bytearray(MAGIC)
    for value in (HASH_FAMILY.encode("ascii"), FRAMING_ID.encode("ascii"), record_bytes):
        output.extend(_prefixed(value))
    output.extend(pack("<I", len(ordered)))
    for field in ordered:
        output.extend(_prefixed(field.name.encode("ascii")))
        output.extend(pack("<BB", int(field.tag), len(field.shape)))
        for size in field.shape:
            output.extend(pack("<Q", size))
        output.extend(pack("<Q", len(field.payload)))
        output.extend(field.payload)
    return bytes(output)


def digest(record_type: str, fields: Iterable[Field]) -> bytes:
    return sha256(frame(record_type, fields)).digest()


def digest_hex(record_type: str, fields: Iterable[Field]) -> str:
    return digest(record_type, fields).hex()


def hash_bytes_field(name: str, digest_value: bytes) -> Field:
    if len(digest_value) != 32:
        raise ValueError(f"{name} must be a 32-byte SHA-256 digest")
    return Field.numeric(name, DTypeTag.UINT8, list(digest_value))

