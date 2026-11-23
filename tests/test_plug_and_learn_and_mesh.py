"""Tests for Layer-2 Plug-and-Learn on-chip BRAM adaptation and Ternara Mesh proof-gated scaling."""

from __future__ import annotations

import numpy as np
import pytest

from ternara_seed.mesh import MeshDelta, run_mesh_simulation
from ternara_seed.organs import ORGAN_ORDER, train_organ_seed
from ternara_seed.pack5 import Pack5FaultError
from ternara_seed.plug_and_learn import MODULES, attach_and_adapt_module


@pytest.mark.parametrize("module_id", list(MODULES.keys()))
def test_plug_and_learn_adapts_onchip_zero_resynthesis(module_id: str) -> None:
    mod = MODULES[module_id]
    cart, _, _ = train_organ_seed(mod.target_organ, board_id="ax7203_xc7a200t")
    adapted, rep = attach_and_adapt_module(cart, module_id)
    assert rep.bit_exact_diff == 0
    assert rep.est_onchip_time_ms < 5.0
    assert rep.acc_after_adapt >= rep.acc_before_adapt
    assert rep.acc_after_adapt >= 0.90
    if module_id in ("bent_rotor_esc", "bosch_bmi270_imu"):
        assert rep.accuracy_gain_pct >= 40.0
    assert adapted.sha256() == rep.adapted_sha256


def test_mesh_delta_wire_roundtrip_and_crc_fault() -> None:
    delta = MeshDelta(
        node_id="fpga_01",
        organ_id="guard",
        base_sha256="deadbeef01234567" * 4,
        w2_new=np.ones((3, 16), dtype=np.int8),
        b2_new=np.array([10, -5, 3], dtype=np.int32),
        local_samples=320,
        local_val_acc=0.95,
    )
    raw = delta.to_bytes()
    assert len(raw) < 300
    parsed = MeshDelta.from_bytes(raw)
    assert parsed.sha256() == delta.sha256()
    np.testing.assert_array_equal(parsed.w2_new, delta.w2_new)
    np.testing.assert_array_equal(parsed.b2_new, delta.b2_new)

    corrupted = bytearray(raw)
    corrupted[-2] ^= 0x55
    with pytest.raises(Pack5FaultError):
        MeshDelta.from_bytes(bytes(corrupted))


@pytest.mark.parametrize("organ_id", ORGAN_ORDER)
def test_ternara_mesh_collective_gain_and_poison_rejection(organ_id: str) -> None:
    cart, _, _ = train_organ_seed(organ_id, board_id="ax7203_xc7a200t")
    collective, rep = run_mesh_simulation(cart, honest_nodes=6, poison_nodes=2)
    assert rep.deltas_accepted == 6
    assert rep.deltas_rejected == 2
    assert rep.avg_delta_bytes < 512.0
    assert rep.collective_mesh_global_acc >= 0.85
    assert rep.collective_mesh_global_acc > rep.base_global_acc + 0.45
    assert collective.sha256() == rep.collective_sha256
