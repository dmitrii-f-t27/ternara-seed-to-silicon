"""Tests for full ALINX AX7203 top Verilog + 8N1 UART bit-banged live INFER & LEARN simulation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from ternara_seed.ax7203_hw import DEFAULT_XDC_SRC, prepare_ax7203_project, simulate_ax7203_uart_top
from ternara_seed.organs import generate_organ_dataset, train_organ_seed
from ternara_seed.plug_and_learn import MODULES, attach_and_adapt_module
from ternara_seed.q27_emit import DEFAULT_Q27_WORKSPACE

REPO_ROOT = Path(__file__).resolve().parent.parent
HAS_ALL = (
    (DEFAULT_Q27_WORKSPACE / ".venv/bin/python").exists()
    and shutil.which("iverilog") is not None
    and shutil.which("vvp") is not None
    and DEFAULT_XDC_SRC.exists()
)


@pytest.mark.skipif(not HAS_ALL, reason="requires q27rtl2, iverilog/vvp, and ax7203.xdc")
def test_ax7203_uart_top_ping_infer_and_live_bram_learn_reflex() -> None:
    out_dir = REPO_ROOT / "build" / "pytest_ax7203_reflex"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        meta = prepare_ax7203_project("reflex", out_dir)
        assert Path(meta["top_verilog"]).exists()
        assert Path(meta["seed_sv"]).exists()
        assert Path(meta["xdc"]).exists()
        assert Path(meta["build_script"]).exists()

        cart, _, _ = train_organ_seed("reflex", board_id="ax7203_xc7a200t")
        mod = MODULES["bosch_bmi270_imu"]
        seed = 9001
        adapted, _ = attach_and_adapt_module(cart, "bosch_bmi270_imu", seed=seed)
        rng = np.random.default_rng(seed)
        shift = np.full(cart.in_dim, mod.offset_bias, dtype=np.float64) + rng.normal(0.0, 4.0, size=cart.in_dim)
        gain = np.full(cart.in_dim, mod.gain_scale, dtype=np.float64)
        X_t, y_t = generate_organ_dataset(
            "reflex", 32, seed=seed + 2, domain_shift=shift, channel_gain=gain, permute_classes=dict(mod.swap_classes)
        )

        sim = simulate_ax7203_uart_top(out_dir, "reflex", cart, X_t, y_t, hot_load_cart=adapted)
        assert sim.ping_ok is True
        assert sim.before_mismatches == 0
        assert sim.after_mismatches == 0
        assert sim.hardware_errors == 0
        assert sim.observed_latency_ticks == 3
        assert sim.after_learn_rtl_acc - sim.before_rtl_acc >= 0.45
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)
