"""End-to-end hardware RTL simulation (`iverilog`/`vvp`) and live TCP/HTTP Ternara Mesh tests."""

from __future__ import annotations

import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from ternara_seed.mesh_client import connect_node_to_mesh
from ternara_seed.mesh_server import create_mesh_server
from ternara_seed.organs import ORGAN_ORDER, generate_organ_dataset, train_organ_seed
from ternara_seed.plug_and_learn import MODULES, attach_and_adapt_module
from ternara_seed.q27_emit import DEFAULT_Q27_WORKSPACE, export_and_compile_q27
from ternara_seed.rtl_sim import simulate_compiled_q27_rtl

REPO_ROOT = Path(__file__).resolve().parent.parent
HAS_COMPILER = (DEFAULT_Q27_WORKSPACE / ".venv/bin/python").exists()
HAS_IVERILOG = shutil.which("iverilog") is not None and shutil.which("vvp") is not None


@pytest.mark.skipif(not (HAS_COMPILER and HAS_IVERILOG), reason="requires q27rtl2 and iverilog/vvp")
@pytest.mark.parametrize("organ_id", ORGAN_ORDER)
def test_compiled_q27_rtl_bit_exact_in_iverilog(organ_id: str) -> None:
    out_dir = REPO_ROOT / "build" / "pytest_rtl" / organ_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        cart, plan, _ = train_organ_seed(organ_id, board_id="ax7203_xc7a200t")
        exp = export_and_compile_q27(cart, plan, out_dir, run_compiler=True)
        assert exp.q27rtl2_compiled and exp.verilog_path is not None
        X_val, y_val = generate_organ_dataset(organ_id, n_samples=80, seed=9999)
        sim = simulate_compiled_q27_rtl(Path(exp.verilog_path).parent, X_val, y_val, cart)
        assert sim.bit_exact_mismatches == 0
        assert sim.hardware_errors == 0
        assert sim.observed_latency_ticks == 3
        assert sim.rtl_accuracy >= 0.85
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)


@pytest.mark.skipif(not (HAS_COMPILER and HAS_IVERILOG), reason="requires q27rtl2 and iverilog/vvp")
def test_live_onchip_plug_and_learn_inside_same_compiled_rtl() -> None:
    out_dir = REPO_ROOT / "build" / "pytest_rtl" / "plug_and_learn"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        cart, plan, _ = train_organ_seed("reflex", board_id="ax7203_xc7a200t")
        exp = export_and_compile_q27(cart, plan, out_dir, run_compiler=True)
        assert exp.q27rtl2_compiled and exp.verilog_path is not None
        rdir = Path(exp.verilog_path).parent

        mod = MODULES["bosch_bmi270_imu"]
        seed = 9001
        adapted, _ = attach_and_adapt_module(cart, "bosch_bmi270_imu", seed=seed)
        rng = np.random.default_rng(seed)
        shift = np.full(cart.in_dim, mod.offset_bias, dtype=np.float64) + rng.normal(0.0, 4.0, size=cart.in_dim)
        gain = np.full(cart.in_dim, mod.gain_scale, dtype=np.float64)
        perm = dict(mod.swap_classes) if mod.swap_classes else None
        X_test, y_test = generate_organ_dataset(
            "reflex", 200, seed=seed + 2, domain_shift=shift, channel_gain=gain, permute_classes=perm
        )

        before = simulate_compiled_q27_rtl(rdir, X_test, y_test, cart, hot_load_cart=None, work_dir=rdir / "before")
        after = simulate_compiled_q27_rtl(rdir, X_test, y_test, cart, hot_load_cart=adapted, work_dir=rdir / "after")
        assert before.bit_exact_mismatches == 0 and before.hardware_errors == 0
        assert after.bit_exact_mismatches == 0 and after.hardware_errors == 0
        assert after.hot_loaded_plastic_head is True
        assert after.rtl_accuracy - before.rtl_accuracy >= 0.45
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)


def test_live_tcp_mesh_server_client_poison_rejection_and_rtl_hotload(tmp_path: Path) -> None:
    """Full closed loop: real TCP/HTTP mesh-server + 6 honest + 2 poison + 1 corrupt packet -> RTL hot-load!"""
    server, coord = create_mesh_server(host="127.0.0.1", port=0, state_dir=tmp_path / "state")
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        # 1. Connect 6 honest FPGA nodes over real TCP/HTTP
        collective = None
        for i in range(6):
            collective, rep = connect_node_to_mesh(
                server_url=base_url,
                organ_id="guard",
                node_id=f"fpga-honest-{i+1:02d}",
                out_dir=tmp_path / f"node_{i}",
                env_seed=300 + i,
            )
            assert rep.http_status == 200
            assert rep.delta_accepted is True
            assert rep.delta_bytes < 300
            assert Path(rep.collective_seed_path).exists()
            assert Path(rep.pack5_hex_path).exists()

        # 2. Connect 2 poisoned nodes -> MUST be rejected with HTTP 422!
        for j in range(2):
            _, bad_rep = connect_node_to_mesh(
                server_url=base_url,
                organ_id="guard",
                node_id=f"fpga-poison-{j+1:02d}",
                out_dir=tmp_path / f"poison_{j}",
                poison=True,
                env_seed=900 + j,
            )
            assert bad_rep.http_status == 422
            assert bad_rep.delta_accepted is False

        # 3. Send corrupted raw bytes -> MUST be rejected with HTTP 400!
        req = urllib.request.Request(
            f"{base_url}/v1/delta",
            data=b"CORRUPT_PACK5_PAYLOAD",
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=3.0)
        assert exc_info.value.code == 400

        # 4. Verify coordinator live status
        st = coord.status_dict()["organs"]["guard"]
        assert st["deltas_accepted"] == 6
        assert st["deltas_rejected"] == 2
        assert st["collective_proof_acc"] >= 0.85
        assert st["collective_proof_acc"] > st["base_proof_acc"] + 0.45

        # 5. Closed loop Silicon-to-Mesh-to-Silicon: load downloaded collective .tseed.bin into real design.sv!
        if HAS_COMPILER and HAS_IVERILOG and collective is not None:
            rtl_out = REPO_ROOT / "build" / "pytest_rtl" / "mesh_closed_loop"
            if rtl_out.exists():
                shutil.rmtree(rtl_out)
            try:
                base_anchor = coord.states["guard"].base_cart
                _, plan, _ = train_organ_seed("guard", board_id="ax7203_xc7a200t")
                exp = export_and_compile_q27(base_anchor, plan, rtl_out, run_compiler=True)
                assert exp.q27rtl2_compiled and exp.verilog_path is not None
                rdir = Path(exp.verilog_path).parent
                X_p = coord.states["guard"].X_proof
                y_p = coord.states["guard"].y_proof
                sim_col = simulate_compiled_q27_rtl(rdir, X_p, y_p, base_anchor, hot_load_cart=collective)
                assert sim_col.bit_exact_mismatches == 0
                assert sim_col.hardware_errors == 0
                assert sim_col.rtl_accuracy == pytest.approx(st["collective_proof_acc"], abs=1e-4)
            finally:
                if rtl_out.exists():
                    shutil.rmtree(rtl_out)
    finally:
        server.shutdown()
        server.server_close()
