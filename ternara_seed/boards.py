"""Verified FPGA silicon passports (6 boards across 3 families) + Multi-Board Ternara Hive composer.

All resource figures are anchored to manufacturer datasheets and Ternara's measured AX7203 openXC7
calibration (see ternara-self-adapt boards.json). Binary FPGA silicon emulates balanced ternary
arithmetic via pack5 (5 trits/byte) with 0 DSP slices and 0 external DRAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BoardPassport:
    board_id: str
    name: str
    family: str
    part: str
    jtag_idcode: str
    lut: int
    ff: int
    lut_cost_factor: float
    bram_blocks: int
    bram_kibits: int
    payload_bytes_per_block: int
    default_mhz: float
    est_power_w: float
    retail_usd: int
    toolchain: str
    qualification: str  # "measured_live_silicon" | "open_toolchain_projection" | "reference_ceiling"

    @property
    def total_bram_bytes(self) -> int:
        return self.bram_blocks * self.payload_bytes_per_block

    @property
    def max_pack5_trits(self) -> int:
        return self.total_bram_bytes * 5


BOARDS: Dict[str, BoardPassport] = {
    "icebreaker_up5k": BoardPassport(
        board_id="icebreaker_up5k",
        name="iCEBreaker (iCE40-UP5K)",
        family="Lattice iCE40",
        part="iCE40UP5K-SG48",
        jtag_idcode="0x00005000",
        lut=5_280,
        ff=5_280,
        lut_cost_factor=1.6,
        bram_blocks=30,
        bram_kibits=4,
        payload_bytes_per_block=512,
        default_mhz=12.0,
        est_power_w=0.09,
        retail_usd=69,
        toolchain="Yosys + nextpnr-ice40 + IceStorm (0% proprietary)",
        qualification="open_toolchain_projection",
    ),
    "xc7a35t": BoardPassport(
        board_id="xc7a35t",
        name="Digilent Arty A7-35T",
        family="Xilinx Artix-7",
        part="XC7A35TICSG324-1L",
        jtag_idcode="0x0362D093",
        lut=20_800,
        ff=41_600,
        lut_cost_factor=1.0,
        bram_blocks=50,
        bram_kibits=36,
        payload_bytes_per_block=4_096,
        default_mhz=20.0,
        est_power_w=0.55,
        retail_usd=159,
        toolchain="openXC7 (Yosys + nextpnr-xilinx, 0% Vivado)",
        qualification="open_toolchain_projection",
    ),
    "ecp5_85f": BoardPassport(
        board_id="ecp5_85f",
        name="Ternara Edge / ULX3S (ECP5-85F)",
        family="Lattice ECP5",
        part="LFE5U-85F-6BG381C",
        jtag_idcode="0x41113043",
        lut=83_640,
        ff=83_640,
        lut_cost_factor=1.6,
        bram_blocks=208,
        bram_kibits=18,
        payload_bytes_per_block=2_048,
        default_mhz=25.0,
        est_power_w=0.85,
        retail_usd=179,
        toolchain="Yosys + nextpnr-ecp5 + Project Trellis (0% proprietary)",
        qualification="open_toolchain_projection",
    ),
    "xc7a100t": BoardPassport(
        board_id="xc7a100t",
        name="Digilent Arty A7-100T",
        family="Xilinx Artix-7",
        part="XC7A100TCSG324-1",
        jtag_idcode="0x03631093",
        lut=63_400,
        ff=126_800,
        lut_cost_factor=1.0,
        bram_blocks=135,
        bram_kibits=36,
        payload_bytes_per_block=4_096,
        default_mhz=25.0,
        est_power_w=1.10,
        retail_usd=249,
        toolchain="openXC7 (Yosys + nextpnr-xilinx, 0% Vivado)",
        qualification="open_toolchain_projection",
    ),
    "ax7203_xc7a200t": BoardPassport(
        board_id="ax7203_xc7a200t",
        name="ALINX AX7203 / Ternara Pro (XC7A200T)",
        family="Xilinx Artix-7",
        part="XC7A200T-FBG484-2",
        jtag_idcode="0x03636093",
        lut=134_600,
        ff=269_200,
        lut_cost_factor=1.0,
        bram_blocks=365,
        bram_kibits=36,
        payload_bytes_per_block=4_096,
        default_mhz=25.0,
        est_power_w=1.65,
        retail_usd=349,
        toolchain="openXC7 (Yosys + nextpnr-xilinx, 0% Vivado, measured live)",
        qualification="measured_live_silicon",
    ),
    "xcu50_alveo": BoardPassport(
        board_id="xcu50_alveo",
        name="AMD Alveo U50 (Reference Ceiling)",
        family="AMD UltraScale+",
        part="XCU50",
        jtag_idcode="0x14B31093",
        lut=872_000,
        ff=1_743_000,
        lut_cost_factor=1.0,
        bram_blocks=1_344,
        bram_kibits=36,
        payload_bytes_per_block=4_096,
        default_mhz=50.0,
        est_power_w=12.0,
        retail_usd=2_995,
        toolchain="Reference capacity ceiling (UltraScale+)",
        qualification="reference_ceiling",
    ),
}

BOARD_ORDER: List[str] = [
    "icebreaker_up5k",
    "xc7a35t",
    "ecp5_85f",
    "xc7a100t",
    "ax7203_xc7a200t",
    "xcu50_alveo",
]


def get_board(board_id: str, hive_count: int = 1) -> BoardPassport:
    """Lookup board passport by ID or JTAG IDCODE; optionally compose an N-board Ternara Hive."""
    if hive_count < 1 or hive_count > 16:
        raise ValueError("hive_count must be between 1 and 16")
    key = board_id.strip().lower()
    if key in BOARDS:
        base = BOARDS[key]
    else:
        matched = [b for b in BOARDS.values() if b.jtag_idcode.lower() == key]
        if not matched:
            raise KeyError(f"Unknown board_id or JTAG IDCODE: {board_id!r}. Available: {', '.join(BOARD_ORDER)}")
        base = matched[0]
    if hive_count == 1:
        return base
    # Local daisy-chained Ternara Hive: pools BRAM & LUTs over sub-microsecond LVDS/SPI links
    return BoardPassport(
        board_id=f"{base.board_id}_hive{hive_count}",
        name=f"{base.name} x{hive_count} Ternara Hive",
        family=base.family,
        part=f"{hive_count}x {base.part}",
        jtag_idcode=base.jtag_idcode,
        lut=base.lut * hive_count,
        ff=base.ff * hive_count,
        lut_cost_factor=base.lut_cost_factor,
        bram_blocks=base.bram_blocks * hive_count,
        bram_kibits=base.bram_kibits,
        payload_bytes_per_block=base.payload_bytes_per_block,
        default_mhz=base.default_mhz,
        est_power_w=round(base.est_power_w * hive_count, 2),
        retail_usd=base.retail_usd * hive_count,
        toolchain=f"{base.toolchain} + Ternara Hive LVDS mesh",
        qualification=base.qualification,
    )
