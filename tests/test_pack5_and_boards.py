"""Clean-room pack5 codec, SeedCartridge integrity, and 6-board + Hive passport tests."""

from __future__ import annotations

import numpy as np
import pytest

from ternara_seed.boards import BOARD_ORDER, BOARDS, get_board
from ternara_seed.pack5 import (
    Pack5FaultError,
    SeedCartridge,
    pack5_decode,
    pack5_encode,
    verify_all_reserved_fault_bytes,
)


def test_pack5_roundtrip_arbitrary_lengths() -> None:
    rng = np.random.default_rng(12345)
    for n in (1, 2, 4, 5, 6, 17, 64, 256, 313):
        trits = rng.integers(-1, 2, size=n, dtype=np.int8)
        raw = pack5_encode(trits)
        assert len(raw) == (n + 4) // 5
        assert all(0 <= b <= 242 for b in raw)
        decoded = pack5_decode(raw, n)
        np.testing.assert_array_equal(decoded, trits)


def test_pack5_rejects_all_13_reserved_fault_bytes() -> None:
    rejected, total = verify_all_reserved_fault_bytes()
    assert (rejected, total) == (13, 13)
    for bad in range(243, 256):
        with pytest.raises(Pack5FaultError):
            pack5_decode(bytes([bad]), 5)


def test_seed_cartridge_roundtrip_and_crc_fault(tmp_path) -> None:
    rng = np.random.default_rng(77)
    cart = SeedCartridge(
        organ="guard",
        board_id="ax7203_xc7a200t",
        in_dim=16,
        hidden_dim=16,
        out_dim=3,
        thresh1=64,
        w1=rng.integers(-1, 2, size=(16, 16), dtype=np.int8),
        b1=rng.integers(-100, 100, size=(16,), dtype=np.int32),
        w2=rng.integers(-1, 2, size=(3, 16), dtype=np.int8),
        b2=rng.integers(-100, 100, size=(3,), dtype=np.int32),
        metadata={"unit_test": True},
    )
    path = cart.save(tmp_path / "test.tseed.bin")
    loaded = SeedCartridge.load(path)
    assert loaded.sha256() == cart.sha256()
    np.testing.assert_array_equal(loaded.w1, cart.w1)
    np.testing.assert_array_equal(loaded.w2, cart.w2)
    np.testing.assert_array_equal(loaded.b1, cart.b1)
    np.testing.assert_array_equal(loaded.b2, cart.b2)

    corrupted = bytearray(cart.to_bytes())
    corrupted[-1] ^= 0xFF
    with pytest.raises(Pack5FaultError):
        SeedCartridge.from_bytes(bytes(corrupted))


def test_six_boards_and_hive_composition() -> None:
    assert len(BOARD_ORDER) == 6
    assert set(BOARD_ORDER) == set(BOARDS.keys())
    eff_cap = [get_board(bid).lut / get_board(bid).lut_cost_factor for bid in BOARD_ORDER]
    assert eff_cap == sorted(eff_cap)

    single = get_board("ecp5_85f", hive_count=1)
    hive4 = get_board("ecp5_85f", hive_count=4)
    assert hive4.lut == single.lut * 4
    assert hive4.bram_blocks == single.bram_blocks * 4
    assert hive4.bram_kibits == single.bram_kibits
    assert hive4.ff == single.ff * 4
