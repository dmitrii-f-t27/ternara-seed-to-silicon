"""Real working Ternara Mesh Coordinator Server (HTTP + binary pack5 `.tdelta.bin` / `.tseed.bin`).

Endpoints:
  GET  /v1/health            -> RTT/liveness check + summary counts
  GET  /v1/status            -> Live per-organ collective accuracy, accepted/rejected counts, SHA-256
  GET  /v1/base/<organ>      -> Immutable anchor base `.tseed.bin` (`application/octet-stream`)
  GET  /v1/seed/<organ>      -> Latest verified collective `.tseed.bin` (`application/octet-stream`)
  POST /v1/delta             -> Submit binary `.tdelta.bin` (<250 B); CRC32 + pack5 + Proof Gate verified
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mesh import MeshDelta, aggregate_verified_mesh
from .oracle import oracle_forward_batch
from .organs import ORGAN_ORDER, generate_organ_dataset, train_organ_seed
from .pack5 import Pack5FaultError, SeedCartridge


@dataclass
class OrganMeshState:
    organ_id: str
    base_cart: SeedCartridge
    collective_cart: SeedCartridge
    X_proof: np.ndarray
    y_proof: np.ndarray
    base_proof_acc: float
    collective_proof_acc: float
    min_proof_acc: float = 0.65
    round_num: int = 0
    accepted_deltas: List[MeshDelta] = field(default_factory=list)
    deltas_rejected: int = 0
    corrupt_packets: int = 0
    total_delta_bytes: int = 0


class MeshCoordinator:
    """Thread-safe, disk-persisting Ternara Mesh state machine with deterministic Proof Gate."""

    def __init__(
        self,
        state_dir: str | Path = "build/mesh_state",
        organs: Tuple[str, ...] = tuple(ORGAN_ORDER),
        board_id: str = "ax7203_xc7a200t",
        start_from_blank_head: bool = True,
        seed: int = 20261122,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.states: Dict[str, OrganMeshState] = {}

        for oid in organs:
            trained, _, _ = train_organ_seed(oid, board_id=board_id)
            if start_from_blank_head:
                base = SeedCartridge(
                    organ=trained.organ,
                    board_id=trained.board_id,
                    in_dim=trained.in_dim,
                    hidden_dim=trained.hidden_dim,
                    out_dim=trained.out_dim,
                    w1=trained.w1.copy(),
                    b1=trained.b1.copy(),
                    thresh1=trained.thresh1,
                    w2=np.zeros_like(trained.w2),
                    b2=np.zeros_like(trained.b2),
                    metadata={**trained.metadata, "mesh_anchor": True},
                )
            else:
                base = trained

            X_proof, y_proof = generate_organ_dataset(oid, 160, seed=seed + 999, noise_std=20.0)
            _, _, base_preds = oracle_forward_batch(base, X_proof)
            base_acc = float(np.mean(base_preds == y_proof))

            st = OrganMeshState(
                organ_id=oid,
                base_cart=base,
                collective_cart=base,
                X_proof=X_proof,
                y_proof=y_proof,
                base_proof_acc=round(base_acc, 4),
                collective_proof_acc=round(base_acc, 4),
            )
            self.states[oid] = st
            self._persist_organ(st)

    def _persist_organ(self, st: OrganMeshState) -> None:
        odir = self.state_dir / st.organ_id
        odir.mkdir(parents=True, exist_ok=True)
        st.base_cart.save(odir / "base.tseed.bin")
        st.collective_cart.save(odir / "collective.tseed.bin")
        meta = {
            "organ_id": st.organ_id,
            "round": st.round_num,
            "base_sha256": st.base_cart.sha256(),
            "collective_sha256": st.collective_cart.sha256(),
            "base_proof_acc": st.base_proof_acc,
            "collective_proof_acc": st.collective_proof_acc,
            "deltas_accepted": len(st.accepted_deltas),
            "deltas_rejected": st.deltas_rejected,
            "corrupt_packets": st.corrupt_packets,
        }
        (odir / "status.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def get_seed(self, organ_id: str, anchor: bool = False) -> Optional[Tuple[bytes, Dict[str, str]]]:
        with self._lock:
            st = self.states.get(organ_id)
            if st is None:
                return None
            cart = st.base_cart if anchor else st.collective_cart
            raw = cart.to_bytes()
            headers = {
                "X-Ternara-Organ": st.organ_id,
                "X-Ternara-SHA256": cart.sha256(),
                "X-Ternara-Base-SHA256": st.base_cart.sha256(),
                "X-Ternara-Round": str(st.round_num),
                "X-Ternara-Proof-Acc": f"{st.collective_proof_acc:.4f}",
            }
            return raw, headers

    def submit_delta_bytes(self, raw_bytes: bytes) -> Tuple[int, Dict[str, object]]:
        if not raw_bytes or len(raw_bytes) > 65536:
            return HTTPStatus.BAD_REQUEST, {"accepted": False, "reason": "empty_or_oversized_payload"}
        try:
            delta = MeshDelta.from_bytes(raw_bytes)
        except (Pack5FaultError, ValueError) as exc:
            with self._lock:
                for st in self.states.values():
                    st.corrupt_packets += 1
                    break
            return HTTPStatus.BAD_REQUEST, {"accepted": False, "reason": f"pack5_or_crc_fault: {exc}"}

        with self._lock:
            st = self.states.get(delta.organ_id)
            if st is None:
                return HTTPStatus.NOT_FOUND, {"accepted": False, "reason": f"unknown_organ: {delta.organ_id}"}

            if delta.base_sha256 != st.base_cart.sha256():
                st.deltas_rejected += 1
                self._persist_organ(st)
                return HTTPStatus.UNPROCESSABLE_ENTITY, {
                    "accepted": False,
                    "reason": "base_sha256_mismatch",
                    "expected_base_sha256": st.base_cart.sha256(),
                }

            # Deterministic Proof Gate evaluation of this submitted node head
            cand = SeedCartridge(
                organ=st.base_cart.organ,
                board_id=st.base_cart.board_id,
                in_dim=st.base_cart.in_dim,
                hidden_dim=st.base_cart.hidden_dim,
                out_dim=st.base_cart.out_dim,
                w1=st.base_cart.w1,
                b1=st.base_cart.b1,
                thresh1=st.base_cart.thresh1,
                w2=delta.w2_new,
                b2=delta.b2_new,
                metadata=st.base_cart.metadata,
            )
            _, _, preds = oracle_forward_batch(cand, st.X_proof)
            node_proof_acc = float(np.mean(preds == st.y_proof))
            if node_proof_acc < st.min_proof_acc:
                st.deltas_rejected += 1
                self._persist_organ(st)
                return HTTPStatus.UNPROCESSABLE_ENTITY, {
                    "accepted": False,
                    "reason": "proof_gate_rejected_poison_or_low_accuracy",
                    "node_id": delta.node_id,
                    "node_proof_acc": round(node_proof_acc, 4),
                    "min_required_acc": st.min_proof_acc,
                    "deltas_rejected": st.deltas_rejected,
                }

            # Replace previous delta from same node_id or append new verified node
            st.accepted_deltas = [d for d in st.accepted_deltas if d.node_id != delta.node_id]
            st.accepted_deltas.append(delta)
            st.total_delta_bytes += len(raw_bytes)

            merged, n_acc, _ = aggregate_verified_mesh(
                st.base_cart,
                st.accepted_deltas,
                st.X_proof,
                st.y_proof,
                min_proof_acc=st.min_proof_acc,
            )
            _, _, col_preds = oracle_forward_batch(merged, st.X_proof)
            st.collective_cart = merged
            st.collective_proof_acc = round(float(np.mean(col_preds == st.y_proof)), 4)
            st.round_num += 1
            self._persist_organ(st)

            return HTTPStatus.OK, {
                "accepted": True,
                "organ_id": st.organ_id,
                "node_id": delta.node_id,
                "delta_bytes": len(raw_bytes),
                "node_proof_acc": round(node_proof_acc, 4),
                "round": st.round_num,
                "deltas_accepted": n_acc,
                "deltas_rejected": st.deltas_rejected,
                "base_proof_acc": st.base_proof_acc,
                "collective_proof_acc": st.collective_proof_acc,
                "collective_sha256": st.collective_cart.sha256(),
            }

    def status_dict(self) -> Dict[str, object]:
        with self._lock:
            organs_out = {}
            tot_acc = 0
            tot_rej = 0
            for oid, st in self.states.items():
                n_acc = len(st.accepted_deltas)
                tot_acc += n_acc
                tot_rej += st.deltas_rejected
                avg_b = round(st.total_delta_bytes / n_acc, 1) if n_acc else 0.0
                organs_out[oid] = {
                    "round": st.round_num,
                    "deltas_accepted": n_acc,
                    "deltas_rejected": st.deltas_rejected,
                    "avg_delta_bytes": avg_b,
                    "base_proof_acc": st.base_proof_acc,
                    "collective_proof_acc": st.collective_proof_acc,
                    "base_sha256": st.base_cart.sha256(),
                    "collective_sha256": st.collective_cart.sha256(),
                    "collective_bytes": len(st.collective_cart.to_bytes()),
                }
            return {
                "status": "ok",
                "uptime_s": round(time.time() - self.started_at, 2),
                "total_deltas_accepted": tot_acc,
                "total_deltas_rejected": tot_rej,
                "organs": organs_out,
            }


def create_mesh_server(
    host: str = "127.0.0.1",
    port: int = 8795,
    coordinator: Optional[MeshCoordinator] = None,
    state_dir: str | Path = "build/mesh_state",
) -> Tuple[ThreadingHTTPServer, MeshCoordinator]:
    """Create a real multi-threaded HTTP Ternara Mesh coordinator server."""
    coord = coordinator if coordinator is not None else MeshCoordinator(state_dir=state_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, code: int, obj: Dict[str, object]) -> None:
            body = (json.dumps(obj, indent=2) + "\n").encode("utf-8")
            self.send_response(int(code))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/v1/health":
                st = coord.status_dict()
                self._send_json(HTTPStatus.OK, {
                    "status": "ok",
                    "organs": len(st["organs"]),
                    "total_deltas_accepted": st["total_deltas_accepted"],
                    "total_deltas_rejected": st["total_deltas_rejected"],
                })
                return
            if path == "/v1/status":
                self._send_json(HTTPStatus.OK, coord.status_dict())
                return
            if path.startswith("/v1/seed/") or path.startswith("/v1/base/"):
                anchor = path.startswith("/v1/base/")
                organ_id = path.split("/", 3)[3].strip("/")
                res = coord.get_seed(organ_id, anchor=anchor)
                if res is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown organ {organ_id}"})
                    return
                raw, headers = res
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/v1/delta":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            code, payload = coord.submit_delta_bytes(raw)
            self._send_json(code, payload)

    server = ThreadingHTTPServer((host, port), Handler)
    return server, coord
