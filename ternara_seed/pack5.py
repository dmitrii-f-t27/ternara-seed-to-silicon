"""Strict clean-room radix-243 pack5 codec (5 trits -> 1 byte, 13 reserved fault bytes).

Encodes balanced ternary weights {-1, 0, +1} at 5 trits per 8-bit byte (3^5 = 243 < 256).
Bytes 243..255 are strictly reserved fault codes and ALWAYS raise Pack5FaultError on decode.
Also provides CRC32 + SHA-256 framed cartridge (.tseed.bin) and sparse federated delta (.tdelta.bin).
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

RADIX_5 = 243
SEED_MAGIC = b"TSD1"
DELTA_MAGIC = b"TDL1"


class Pack5FaultError(ValueError):
    """Raised when encountering reserved bytes (243..255), invalid trits, or CRC/format corruption."""


def pack5_encode(trits: np.ndarray) -> bytes:
    """Pack a 1D array of trits in {-1, 0, +1} into ceil(N / 5) bytes (LSB-first radix-3)."""
    flat = np.asarray(trits, dtype=np.int16).reshape(-1)
    if flat.size == 0:
        return b""
    if np.any((flat < -1) | (flat > 1)):
        raise Pack5FaultError("pack5_encode only accepts balanced trits in {-1, 0, +1}")
    digits = (flat + 1).astype(np.int32)
    n = digits.size
    pad = (5 - (n % 5)) % 5
    if pad:
        digits = np.concatenate([digits, np.ones(pad, dtype=np.int32)])  # pad with trit 0 -> digit 1
    groups = digits.reshape(-1, 5)
    vals = (
        groups[:, 0]
        + groups[:, 1] * 3
        + groups[:, 2] * 9
        + groups[:, 3] * 27
        + groups[:, 4] * 81
    )
    return bytes(int(v) for v in vals)


def pack5_decode(data: bytes, count: int) -> np.ndarray:
    """Decode `count` balanced trits from radix-243 bytes; rejects any byte >= 243."""
    if count < 0:
        raise Pack5FaultError("count must be non-negative")
    expected_bytes = (count + 4) // 5
    if len(data) != expected_bytes:
        raise Pack5FaultError(f"Expected {expected_bytes} bytes for {count} trits, got {len(data)}")
    out = np.empty(expected_bytes * 5, dtype=np.int8)
    for i, b in enumerate(data):
        if b >= RADIX_5:
            raise Pack5FaultError(f"Reserved pack5 fault byte 0x{b:02x} ({b}) at offset {i}")
        v = b
        base = i * 5
        for k in range(5):
            out[base + k] = (v % 3) - 1
            v //= 3
    return out[:count]


@dataclass(frozen=True)
class SeedCartridge:
    """Self-contained Q27 Seed ternary cartridge (2-layer 0-DSP backbone + plastic head)."""

    organ: str
    board_id: str
    in_dim: int
    hidden_dim: int
    out_dim: int
    w1: np.ndarray        # (hidden_dim, in_dim) int8 in {-1, 0, +1}
    b1: np.ndarray        # (hidden_dim,) int16
    thresh1: int          # native Q27 `измерить` threshold
    w2: np.ndarray        # (out_dim, hidden_dim) int8 in {-1, 0, +1} (plastic on-chip layer!)
    b2: np.ndarray        # (out_dim,) int16
    metadata: Dict[str, object]

    def __post_init__(self) -> None:
        if self.w1.shape != (self.hidden_dim, self.in_dim):
            raise Pack5FaultError(f"w1 shape mismatch: {self.w1.shape}")
        if self.b1.shape != (self.hidden_dim,):
            raise Pack5FaultError(f"b1 shape mismatch: {self.b1.shape}")
        if self.w2.shape != (self.out_dim, self.hidden_dim):
            raise Pack5FaultError(f"w2 shape mismatch: {self.w2.shape}")
        if self.b2.shape != (self.out_dim,):
            raise Pack5FaultError(f"b2 shape mismatch: {self.b2.shape}")

    @property
    def total_trits(self) -> int:
        return int(self.w1.size + self.w2.size)

    @property
    def zero_fraction(self) -> float:
        z = int(np.sum(self.w1 == 0) + np.sum(self.w2 == 0))
        return z / max(1, self.total_trits)

    def to_bytes(self) -> bytes:
        meta_bytes = json.dumps(
            {
                "organ": self.organ,
                "board_id": self.board_id,
                "in_dim": self.in_dim,
                "hidden_dim": self.hidden_dim,
                "out_dim": self.out_dim,
                "thresh1": int(self.thresh1),
                "metadata": self.metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        w1_p5 = pack5_encode(self.w1)
        w2_p5 = pack5_encode(self.w2)
        b1_raw = np.asarray(self.b1, dtype="<i2").tobytes()
        b2_raw = np.asarray(self.b2, dtype="<i2").tobytes()

        header = struct.pack(
            "<4sIIIII",
            SEED_MAGIC,
            len(meta_bytes),
            len(w1_p5),
            len(w2_p5),
            len(b1_raw),
            len(b2_raw),
        )
        body = header + meta_bytes + w1_p5 + w2_p5 + b1_raw + b2_raw
        crc = zlib.crc32(body) & 0xFFFFFFFF
        return body + struct.pack("<I", crc)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.to_bytes())
        return p

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SeedCartridge":
        if len(raw) < 28:
            raise Pack5FaultError("SeedCartridge too short")
        body, crc_raw = raw[:-4], raw[-4:]
        expected_crc = struct.unpack("<I", crc_raw)[0]
        actual_crc = zlib.crc32(body) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise Pack5FaultError(f"CRC32 mismatch: expected 0x{expected_crc:08x}, got 0x{actual_crc:08x}")
        magic, l_meta, l_w1, l_w2, l_b1, l_b2 = struct.unpack("<4sIIIII", body[:24])
        if magic != SEED_MAGIC:
            raise Pack5FaultError(f"Bad SeedCartridge magic: {magic!r}")
        off = 24
        meta = json.loads(body[off : off + l_meta].decode("utf-8"))
        off += l_meta
        w1_p5 = body[off : off + l_w1]
        off += l_w1
        w2_p5 = body[off : off + l_w2]
        off += l_w2
        b1_raw = body[off : off + l_b1]
        off += l_b1
        b2_raw = body[off : off + l_b2]

        in_dim = int(meta["in_dim"])
        hidden_dim = int(meta["hidden_dim"])
        out_dim = int(meta["out_dim"])
        w1 = pack5_decode(w1_p5, hidden_dim * in_dim).reshape(hidden_dim, in_dim)
        w2 = pack5_decode(w2_p5, out_dim * hidden_dim).reshape(out_dim, hidden_dim)
        b1 = np.frombuffer(b1_raw, dtype="<i2").astype(np.int16).copy()
        b2 = np.frombuffer(b2_raw, dtype="<i2").astype(np.int16).copy()
        return cls(
            organ=str(meta["organ"]),
            board_id=str(meta["board_id"]),
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            w1=w1,
            b1=b1,
            thresh1=int(meta["thresh1"]),
            w2=w2,
            b2=b2,
            metadata=dict(meta.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SeedCartridge":
        return cls.from_bytes(Path(path).read_bytes())


def verify_all_reserved_fault_bytes() -> Tuple[int, int]:
    """Verify all 13 reserved pack5 bytes (243..255) are strictly rejected."""
    rejected = 0
    for b in range(RADIX_5, 256):
        try:
            pack5_decode(bytes([b]), 5)
        except Pack5FaultError:
            rejected += 1
    return rejected, 256 - RADIX_5
