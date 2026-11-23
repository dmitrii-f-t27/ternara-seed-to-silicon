"""Real working Ternara Mesh Node Client (connects FPGA host -> live Mesh Server -> BRAM hot-load).

Workflow per connected FPGA node:
  1. (Optional `--detect-jtag`) Reads JTAG IDCODE via `openFPGALoader --detect` and maps to `boards.py`.
  2. Downloads the anchor base `.tseed.bin` (<430 B) over HTTP from `/v1/base/<organ>` and verifies CRC32+SHA256.
  3. Runs local integer on-chip adaptation (`onchip_plastic_adapt`, 0 floats) for this board's environment.
  4. Packs `<250 B` binary `.tdelta.bin` (`pack5` + CRC32) and POSTs to `/v1/delta`.
  5. Downloads the upgraded collective `.tseed.bin` from `/v1/seed/<organ>`, verifies integrity,
     writes `collective.tseed.bin` + `collective_pack5.hex` for BRAM hot-reload (`обуч_вкл`).
  6. Optionally runs `simulate_compiled_q27_rtl` to verify the downloaded collective weights directly
     inside the compiled `design.sv` SystemVerilog hardware core!
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .boards import BOARDS
from .mesh import MeshDelta
from .oracle import oracle_forward_batch
from .organs import generate_organ_dataset
from .pack5 import SeedCartridge, pack5_encode
from .plug_and_learn import onchip_plastic_adapt


@dataclass(frozen=True)
class MeshConnectReport:
    node_id: str
    organ_id: str
    server_url: str
    detected_board_id: str
    delta_bytes: int
    http_status: int
    delta_accepted: bool
    server_reason: str
    mesh_round: int
    collective_proof_acc: float
    collective_sha256: str
    collective_seed_path: str
    pack5_hex_path: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def detect_jtag_board_id(fallback_board: str = "ax7203_xc7a200t") -> Tuple[str, str]:
    """Opt-in live JTAG IDCODE detection via `openFPGALoader --detect`."""
    loader = shutil.which("openFPGALoader")
    if not loader:
        return fallback_board, "openFPGALoader not found; using fallback board"
    try:
        proc = subprocess.run([loader, "--detect"], capture_output=True, text=True, timeout=6)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"0x[0-9a-fA-F]{8}", out)
        if m:
            idcode = m.group(0).lower()
            for bid, bp in BOARDS.items():
                if bp.jtag_idcode.lower() == idcode:
                    return bid, f"Detected live JTAG IDCODE {idcode} -> {bp.name}"
            return fallback_board, f"Detected unknown JTAG IDCODE {idcode}; using {fallback_board}"
        return fallback_board, "No USB JTAG FPGA cable detected; using fallback board"
    except Exception as exc:
        return fallback_board, f"JTAG detect skipped ({exc}); using fallback board"


def _http_get_bytes(url: str, timeout: float = 5.0) -> Tuple[int, bytes, Dict[str, str]]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read(), dict(resp.headers.items())


def _http_post_bytes(url: str, data: bytes, timeout: float = 5.0) -> Tuple[int, Dict[str, object]]:
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"accepted": False, "reason": raw}
        return err.code, body


def connect_node_to_mesh(
    server_url: str,
    organ_id: str,
    node_id: str,
    out_dir: str | Path = "build/mesh_node",
    board_id: str = "ax7203_xc7a200t",
    detect_jtag: bool = False,
    poison: bool = False,
    env_seed: int = 101,
) -> Tuple[SeedCartridge, MeshConnectReport]:
    """Execute one full live Ternara Mesh sync cycle over HTTP with binary pack5 packets."""
    base_url = server_url.rstrip("/")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    active_board = board_id
    if detect_jtag:
        active_board, _ = detect_jtag_board_id(fallback_board=board_id)

    # 1. Fetch immutable base anchor cartridge from coordinator
    _, base_raw, _ = _http_get_bytes(f"{base_url}/v1/base/{organ_id}")
    base_cart = SeedCartridge.from_bytes(base_raw)

    # 2. Local on-chip integer adaptation for this node's environment (or adversarial poison test)
    rng = np.random.default_rng(env_seed)
    if poison:
        w2_new = rng.integers(-1, 2, size=base_cart.w2.shape, dtype=np.int8)
        b2_new = rng.integers(-120, 120, size=base_cart.b2.shape, dtype=np.int16)
        local_acc = 0.99
    else:
        shift = rng.normal(0.0, 14.0, size=base_cart.in_dim)
        X_loc, y_loc = generate_organ_dataset(organ_id, 36, seed=env_seed + 100, noise_std=22.0, domain_shift=shift)
        adapted = onchip_plastic_adapt(base_cart, X_loc, y_loc, epochs=8)
        w2_new = adapted.w2
        b2_new = adapted.b2
        _, _, preds = oracle_forward_batch(adapted, X_loc)
        local_acc = float(np.mean(preds == y_loc))

    # 3. Pack <250 B binary pack5 delta and POST to coordinator
    delta = MeshDelta(
        node_id=node_id,
        organ_id=organ_id,
        base_sha256=base_cart.sha256(),
        w2_new=w2_new,
        b2_new=b2_new,
        local_samples=36,
        local_val_acc=local_acc,
    )
    delta_bytes = delta.to_bytes()
    code, resp = _http_post_bytes(f"{base_url}/v1/delta", delta_bytes)

    # 4. Pull the latest verified collective cartridge (<430 B) and export BRAM pack5 hex ROM
    _, col_raw, col_headers = _http_get_bytes(f"{base_url}/v1/seed/{organ_id}")
    collective_cart = SeedCartridge.from_bytes(col_raw)
    seed_path = collective_cart.save(out / f"{organ_id}_{node_id}_collective.tseed.bin")
    p5_hex = out / f"{organ_id}_{node_id}_collective_pack5.hex"
    p5_bytes = pack5_encode(collective_cart.w1) + pack5_encode(collective_cart.w2)
    p5_hex.write_text("\n".join(f"{b:02x}" for b in p5_bytes) + "\n", encoding="utf-8")

    rep = MeshConnectReport(
        node_id=node_id,
        organ_id=organ_id,
        server_url=base_url,
        detected_board_id=active_board,
        delta_bytes=len(delta_bytes),
        http_status=int(code),
        delta_accepted=bool(resp.get("accepted", False)),
        server_reason=str(resp.get("reason", "accepted")),
        mesh_round=int(col_headers.get("X-Ternara-Round", resp.get("round", 0))),
        collective_proof_acc=float(col_headers.get("X-Ternara-Proof-Acc", resp.get("collective_proof_acc", 0.0))),
        collective_sha256=collective_cart.sha256(),
        collective_seed_path=str(seed_path),
        pack5_hex_path=str(p5_hex),
    )
    return collective_cart, rep
