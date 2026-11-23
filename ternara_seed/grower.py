"""Ternara Seed-to-Silicon Auto-Grower: grows a compact <4 KB Q27 Seed to fit any FPGA board or Hive.

Given a SeedSpec (organ task) and a target BoardPassport (from iCE40-UP5K up to AX7203 or Hive),
the Auto-Grower computes:
  1. The single-organ Q27 core dimensions (hidden_dim, input/output widths) that fit comfortably.
  2. How many parallel 0-DSP pack5 organ replicas (`parallel_cores`) + extra BRAM cartridge slots
     (`hot_cartridge_slots`) fit inside ~80% of the chip's LUT & BRAM envelope.
  3. Exact deterministic cycle count, microsecond latency, 0 DSP, 0 DRAM, and estimated watts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

from .boards import BoardPassport, get_board


@dataclass(frozen=True)
class OrganSpec:
    organ_id: str
    title: str
    tagline: str
    domain: str
    in_dim: int
    out_dim: int
    class_names: tuple[str, ...]
    cpu_p99_us: float         # conservative host OS/CPU+driver p99 latency for honest comparison
    target_fpga_us: float     # latency budget
    default_bus: str          # primary hardware interface
    plastic_head: bool = True # supports runtime on-chip BRAM adaptation


@dataclass(frozen=True)
class GrownPlan:
    organ_id: str
    board_id: str
    board_name: str
    family: str
    hive_count: int
    in_dim: int
    hidden_dim: int
    out_dim: int
    parallel_cores: int           # number of parallel 0-DSP Q27 cores grown onto the chip
    total_trits_per_core: int
    total_trits_on_chip: int
    pack5_bytes_per_core: int
    hot_cartridge_slots: int      # how many distinct pack5 organs fit simultaneously in chip BRAM
    bram_blocks_used: int
    bram_blocks_total: int
    bram_util_pct: float
    lut_est_per_core: int
    lut_est_total: int
    lut_total: int
    lut_util_pct: float
    dsp_used: int
    dram_bytes_used: int
    clock_mhz: float
    latency_cycles: int
    latency_us: float
    throughput_kfps: float        # aggregate kilo-inferences/sec across all grown parallel cores
    cpu_p99_us: float
    speedup_vs_cpu_p99: float
    est_power_w: float
    qualification: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def grow_seed_for_board(
    organ: OrganSpec,
    board_id: str,
    hive_count: int = 1,
    target_fill_ratio: float = 0.80,
) -> GrownPlan:
    """Automatically morph a Q27 Seed to match the exact LUT and BRAM budget of any target FPGA."""
    board: BoardPassport = get_board(board_id, hive_count=hive_count)

    # Choose clean Q27 vector hidden width matched to silicon tier
    if board.lut < 10_000:
        hidden_dim = 8
    elif board.lut < 40_000:
        hidden_dim = 16
    else:
        hidden_dim = 16  # fast 1-tick/row Q27 core; scale width & throughput via parallel_cores!

    trits_per_core = hidden_dim * organ.in_dim + organ.out_dim * hidden_dim
    pack5_bytes_core = (trits_per_core + 4) // 5 + (hidden_dim + organ.out_dim) * 2 + 64

    # LUT model anchored to Ternara AX7203 measured route anchors (serial_dot ~3k LUT, game_play ~10.6k LUT)
    base_shell_luts = int(round(1_250 * board.lut_cost_factor))
    per_core_luts = int(round((1_400 + hidden_dim * organ.in_dim * 4.5 + organ.out_dim * hidden_dim * 3.5) * board.lut_cost_factor))

    usable_luts = max(per_core_luts, int(board.lut * target_fill_ratio) - base_shell_luts)
    max_cores_by_lut = max(1, usable_luts // max(1, per_core_luts))

    bram_per_core = max(1, math.ceil(pack5_bytes_core / board.payload_bytes_per_block))
    usable_bram = max(1, int(board.bram_blocks * target_fill_ratio))
    max_cores_by_bram = max(1, usable_bram // bram_per_core)

    # Cap parallel_cores cleanly for routing headroom while truly filling big chips
    parallel_cores = max(1, min(max_cores_by_lut, max_cores_by_bram, 32 * hive_count))

    lut_total_est = min(board.lut, base_shell_luts + parallel_cores * per_core_luts)
    bram_used = min(board.bram_blocks, parallel_cores * bram_per_core)

    # Remaining BRAM holds hot-swappable pack5 cartridge organs / shadow banks!
    hot_slots = max(1, (usable_bram * board.payload_bytes_per_block) // max(1, pack5_bytes_core))

    # Deterministic Q27 hardware latency:
    # 1 input latch + hidden_dim cycles (1 row MAC/tick) + out_dim cycles (plastic head) + 2 argmax/emit
    latency_cycles = 1 + hidden_dim + organ.out_dim + 2
    latency_us = round(latency_cycles / board.default_mhz, 2)
    throughput_kfps = round((parallel_cores * board.default_mhz * 1000.0) / latency_cycles, 1)

    # Dynamic power scales gently with active LUT utilization
    lut_util = lut_total_est / max(1, board.lut)
    est_w = round(max(0.06, board.est_power_w * (0.35 + 0.65 * lut_util)), 2)

    return GrownPlan(
        organ_id=organ.organ_id,
        board_id=board.board_id,
        board_name=board.name,
        family=board.family,
        hive_count=hive_count,
        in_dim=organ.in_dim,
        hidden_dim=hidden_dim,
        out_dim=organ.out_dim,
        parallel_cores=parallel_cores,
        total_trits_per_core=trits_per_core,
        total_trits_on_chip=trits_per_core * parallel_cores,
        pack5_bytes_per_core=pack5_bytes_core,
        hot_cartridge_slots=hot_slots,
        bram_blocks_used=bram_used,
        bram_blocks_total=board.bram_blocks,
        bram_util_pct=round(100.0 * bram_used / max(1, board.bram_blocks), 2),
        lut_est_per_core=per_core_luts,
        lut_est_total=lut_total_est,
        lut_total=board.lut,
        lut_util_pct=round(100.0 * lut_total_est / max(1, board.lut), 2),
        dsp_used=0,
        dram_bytes_used=0,
        clock_mhz=board.default_mhz,
        latency_cycles=latency_cycles,
        latency_us=latency_us,
        throughput_kfps=throughput_kfps,
        cpu_p99_us=organ.cpu_p99_us,
        speedup_vs_cpu_p99=round(organ.cpu_p99_us / max(0.01, latency_us), 1),
        est_power_w=est_w,
        qualification=board.qualification,
    )
