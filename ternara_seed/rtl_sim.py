"""Real SystemVerilog RTL simulation harness (`iverilog` + `vvp`) for compiled Q27 v2 Seed organs.

Drives the actual `design.sv` emitted by `q27rtl2 --profile q27-rtl-machine-v2` clock-by-clock:
  1. Verifies 100% bit-exact inference agreement (`0` mismatches) vs the integer oracle.
  2. Verifies zero hardware fault codes (`error_code == 0`) and exact cycle latency (`латентность`).
  3. Optionally pulses the live hardware write strobe (`обуч_вкл = +1`, `обуч_адр`, `обуч_дан`, `обуч_смещ`)
     to hot-load adapted `Plug-and-Learn` or `Ternara Mesh` weights directly into `В2_ПЛАСТ`/`С2_ПЛАСТ`
     inside the running RTL with ZERO resynthesis!
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .oracle import oracle_forward_batch
from .pack5 import SeedCartridge

TRIT_CODE = {-1: 3, 0: 0, 1: 1}


@dataclass(frozen=True)
class RtlSimReport:
    module_name: str
    samples_tested: int
    rtl_accuracy: float
    oracle_accuracy: float
    bit_exact_mismatches: int
    hardware_errors: int
    observed_latency_ticks: int
    hot_loaded_plastic_head: bool
    simulator: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _pack_trit_vec(vec: np.ndarray) -> int:
    out = 0
    for i, v in enumerate(vec.reshape(-1)):
        code = TRIT_CODE[int(v)]
        out |= (code & 0x3) << (2 * i)
    return out


def _pack_int8_vec(vec: np.ndarray) -> int:
    out = 0
    for i, v in enumerate(vec.reshape(-1)):
        out |= (int(v) & 0xFF) << (8 * i)
    return out


def _hexw(val: int, bits: int) -> str:
    mask = (1 << bits) - 1
    nibbles = (bits + 3) // 4
    return f"{val & mask:0{nibbles}x}"


def simulate_compiled_q27_rtl(
    rtl_dir: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    expected_cart: SeedCartridge,
    hot_load_cart: Optional[SeedCartridge] = None,
    work_dir: Optional[str | Path] = None,
) -> RtlSimReport:
    """Compile and run `design.sv` in `iverilog`/`vvp` and prove bit-exact match vs oracle."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required for RTL simulation")

    rdir = Path(rtl_dir)
    sv_path = rdir / "design.sv"
    abi_path = rdir / "abi.json"
    if not sv_path.exists() or not abi_path.exists():
        raise FileNotFoundError(f"Compiled design.sv / abi.json not found in {rdir}")

    abi = json.loads(abi_path.read_text(encoding="utf-8"))
    mod_name = abi["module"]
    ports_by_name = {p["name"]: p for p in abi["ports"]}

    p_start = ports_by_name["запуск"]
    p_x = ports_by_name["вход"]
    p_wen = ports_by_name["обуч_вкл"]
    p_waddr = ports_by_name["обуч_адр"]
    p_wdata = ports_by_name["обуч_дан"]
    p_wbias = ports_by_name["обуч_смещ"]
    p_pred = ports_by_name["решение"]
    p_done = ports_by_name["готов"]
    p_lat = ports_by_name["латентность"]

    n = int(X.shape[0])
    # Oracle ground-truth predictions for the active cartridge
    active_cart = hot_load_cart if hot_load_cart is not None else expected_cart
    _, _, oracle_preds = oracle_forward_batch(active_cart, X)

    sim_dir = Path(work_dir) if work_dir is not None else (rdir / "_rtlsim")
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Write stimulus vectors for testbench
    vec_path = sim_dir / "vectors.hex"
    vec_lines = [_hexw(_pack_int8_vec(X[i]), p_x["width"]) for i in range(n)]
    vec_path.write_text("\n".join(vec_lines) + "\n", encoding="utf-8")

    # Optional hot-load plastic head vectors (one clock cycle per output class!)
    load_stmts: List[str] = []
    if hot_load_cart is not None:
        for c in range(hot_load_cart.out_dim):
            w_hex = _hexw(_pack_trit_vec(hot_load_cart.w2[c]), p_wdata["width"])
            b_hex = _hexw(int(hot_load_cart.b2[c]), p_wbias["width"])
            load_stmts.extend([
                f"    {p_wen['port']} = 2'b01;",
                f"    {p_waddr['port']} = {p_waddr['width']}'d{c};",
                f"    {p_wdata['port']} = {p_wdata['width']}'h{w_hex};",
                f"    {p_wbias['port']} = {p_wbias['width']}'h{b_hex};",
                "    @(posedge clk); #1;",
                "    if (error_code != 0) err_cnt = err_cnt + 1;",
            ])
        load_stmts.extend([
            f"    {p_wen['port']} = 2'b00;",
            f"    {p_waddr['port']} = 0;",
            f"    {p_wdata['port']} = 0;",
            f"    {p_wbias['port']} = 0;",
        ])

    tb_sv = f"""`timescale 1ns/1ps
module tb_ternara_seed;
  reg clk = 1'b0;
  reg rst = 1'b1;
  reg ce  = 1'b1;
  wire ready;
  reg [{p_start['width']-1}:0] {p_start['port']} = 2'b00;
  reg [{p_x['width']-1}:0] {p_x['port']} = {p_x['width']}'h0;
  reg [{p_wen['width']-1}:0] {p_wen['port']} = 2'b00;
  reg [{p_waddr['width']-1}:0] {p_waddr['port']} = 0;
  reg [{p_wdata['width']-1}:0] {p_wdata['port']} = 0;
  reg [{p_wbias['width']-1}:0] {p_wbias['port']} = 0;
  wire [{p_pred['width']-1}:0] {p_pred['port']};
  wire [{p_done['width']-1}:0] {p_done['port']};
  wire [{p_lat['width']-1}:0] {p_lat['port']};
  wire [7:0] error_code;
  wire [31:0] error_line, error_column, error_tick, tick_count;
  reg [6:0] dbg_addr = 7'h0;
  wire [31:0] dbg_data;

  {mod_name} dut (
    .clk(clk), .rst(rst), .ce(ce), .ready(ready),
    .{p_start['port']}({p_start['port']}),
    .{p_x['port']}({p_x['port']}),
    .{p_wen['port']}({p_wen['port']}),
    .{p_waddr['port']}({p_waddr['port']}),
    .{p_wdata['port']}({p_wdata['port']}),
    .{p_wbias['port']}({p_wbias['port']}),
    .{p_pred['port']}({p_pred['port']}),
    .{p_done['port']}({p_done['port']}),
    .{p_lat['port']}({p_lat['port']}),
    .error_code(error_code),
    .error_line(error_line),
    .error_column(error_column),
    .error_tick(error_tick),
    .tick_count(tick_count),
    .dbg_addr(dbg_addr),
    .dbg_data(dbg_data)
  );

  always #5 clk = ~clk;

  reg [{p_x['width']-1}:0] mem [0:{n-1}];
  integer i, step, err_cnt;

  initial begin
    err_cnt = 0;
    $readmemh("{vec_path}", mem);
    rst = 1'b1;
    @(posedge clk); #1;
    @(posedge clk); #1;
    rst = 1'b0;
{chr(10).join(load_stmts)}
    for (i = 0; i < {n}; i = i + 1) begin
      {p_x['port']} = mem[i];
      {p_start['port']} = 2'b01;
      @(posedge clk); #1;
      if (error_code != 0) err_cnt = err_cnt + 1;
      {p_start['port']} = 2'b00;
      for (step = 0; step < 2; step = step + 1) begin
        @(posedge clk); #1;
        if (error_code != 0) err_cnt = err_cnt + 1;
      end
      $display("R %0d %0d %0d %0d %0d", i, {p_pred['port']}, {p_done['port']}, {p_lat['port']}, error_code);
    end
    $display("ERR %0d", err_cnt);
    $finish;
  end
endmodule
"""
    tb_path = sim_dir / "tb_ternara_seed.sv"
    out_vvp = sim_dir / "sim.vvp"
    tb_path.write_text(tb_sv, encoding="utf-8")

    comp = subprocess.run(
        [iverilog, "-g2012", "-o", str(out_vvp), str(sv_path), str(tb_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if comp.returncode != 0:
        raise RuntimeError(f"iverilog failed: {comp.stderr or comp.stdout}")

    run = subprocess.run(
        [vvp, str(out_vvp)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if run.returncode != 0:
        raise RuntimeError(f"vvp failed: {run.stderr or run.stdout}")

    rtl_preds = np.full(n, -1, dtype=np.int32)
    observed_lat = 0
    hw_errors = 0
    for line in run.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "R" and len(parts) >= 6:
            idx = int(parts[1])
            pred = int(parts[2])
            done = int(parts[3])
            lat = int(parts[4])
            err = int(parts[5])
            if done != 1 or err != 0:
                hw_errors += 1
            rtl_preds[idx] = pred
            observed_lat = lat
        elif parts[0] == "ERR" and len(parts) >= 2:
            hw_errors += int(parts[1])

    mismatches = int(np.sum(rtl_preds != oracle_preds))
    rtl_acc = float(np.mean(rtl_preds == y))
    oracle_acc = float(np.mean(oracle_preds == y))
    return RtlSimReport(
        module_name=mod_name,
        samples_tested=n,
        rtl_accuracy=round(rtl_acc, 4),
        oracle_accuracy=round(oracle_acc, 4),
        bit_exact_mismatches=mismatches,
        hardware_errors=hw_errors,
        observed_latency_ticks=observed_lat,
        hot_loaded_plastic_head=hot_load_cart is not None,
        simulator="iverilog/vvp",
    )
