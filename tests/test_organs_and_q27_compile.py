"""Tests for all 5 Q27 organs, bit-exact integer oracle, 6-board Auto-Grower, and real q27rtl2 compilation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ternara_seed.boards import BOARD_ORDER
from ternara_seed.grower import grow_seed_for_board
from ternara_seed.organs import ORGAN_ORDER, ORGANS, train_organ_seed
from ternara_seed.q27_emit import DEFAULT_Q27_WORKSPACE, export_and_compile_q27

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_auto_grower_scales_across_all_six_boards_zero_dsp_zero_dram() -> None:
    spec = ORGANS["guard"]
    prev_trits = 0
    for bid in BOARD_ORDER:
        plan = grow_seed_for_board(spec, bid)
        assert plan.dsp_used == 0
        assert plan.dram_bytes_used == 0
        assert plan.parallel_cores >= 1
        assert plan.hot_cartridge_slots >= 1
        assert plan.total_trits_on_chip >= prev_trits
        prev_trits = plan.total_trits_on_chip

    hive_plan = grow_seed_for_board(spec, "ecp5_85f", hive_count=4)
    single_plan = grow_seed_for_board(spec, "ecp5_85f", hive_count=1)
    assert hive_plan.parallel_cores >= single_plan.parallel_cores * 4
    assert hive_plan.dsp_used == 0


@pytest.mark.parametrize("organ_id", ORGAN_ORDER)
def test_organ_train_bit_exact_and_real_q27rtl2_compile(organ_id: str) -> None:
    cart, plan, rep = train_organ_seed(organ_id, board_id="ax7203_xc7a200t")
    assert rep.val_acc >= 0.85, f"{organ_id} val_acc too low: {rep.val_acc}"
    assert rep.bit_exact_diff == 0
    assert rep.cartridge_bytes < 512

    out_dir = REPO_ROOT / "build" / "pytest_q27" / organ_id
    if out_dir.exists():
        shutil.rmtree(out_dir)

    has_compiler = (DEFAULT_Q27_WORKSPACE / ".venv/bin/python").exists()
    try:
        exp = export_and_compile_q27(cart, plan, out_dir, run_compiler=has_compiler)
        assert Path(exp.q27_path).exists()
        assert Path(exp.pack5_hex_path).exists()
        assert Path(exp.cartridge_path).exists()

        if has_compiler:
            assert exp.q27rtl2_compiled is True, exp.compiler_note
            assert exp.verilog_path is not None and Path(exp.verilog_path).exists()
            assert exp.verilog_lines >= 1000
            assert exp.overflow_checks == 0
            assert exp.double_writes == 0
            sv_text = Path(exp.verilog_path).read_text(encoding="utf-8")
            assert "DSP48" not in sv_text
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)
