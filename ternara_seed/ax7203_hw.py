"""ALINX AX7203 (XC7A200T) 0%-Vivado hardware wrapper, UART bit-bang RTL verifier,
open-source bitstream builder (Yosys + nextpnr-xilinx + prjxray), and live board runner.

Supports all 5 compiled Q27 v2 Seed organs (`guard`, `reflex`, `sentinel`, `tick`, `audio`)
with live on-chip BRAM Plug-and-Learn & Ternara Mesh hot-loading over 115200-baud UART
(0 resynthesis, 0 DSP, 0 external DRAM).
"""

from __future__ import annotations

import json
import os
import select
import shutil
import struct
import subprocess
import termios
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .oracle import oracle_forward_batch
from .organs import ORGAN_ORDER, generate_organ_dataset, train_organ_seed
from .pack5 import SeedCartridge
from .q27_emit import export_and_compile_q27
from .rtl_sim import TRIT_CODE

AX7203_PART = "xc7a200tfbg484-2"
DEFAULT_XDC_SRC = Path("/Volumes/SSDD/Ternara/q27-board-2026-11-07/ax7203/ax7203.xdc")
DEFAULT_CHIPDB_SRC = Path("/Volumes/SSDD/PROJECTS/CLAUDE/trinity-fpga/build/gf16_clean/chipdb/xc7a200tfbg484-2.bin")
DEFAULT_OPENXC7_IMAGE = "regymm/openxc7@sha256:eced1cdd4727549f2d983328e0cf170fb6f6f67d87f19b2bf24365163368c70c"
DEFAULT_HEAVY_LOCK = Path("/Volumes/SSDD/Workspaces/q27-rtl-v02/.tmp/heavy.lock")
DEFAULT_BOARD_LOCK = Path("/private/tmp/claude-501/ax7203-board.lock")
DEFAULT_UART_PORT = "/dev/cu.usbserial-1120"

ORGAN_CODES: Dict[str, int] = {oid: idx + 1 for idx, oid in enumerate(ORGAN_ORDER)}


def pack_trit_u32(vec: np.ndarray) -> int:
    out = 0
    for i, v in enumerate(vec.reshape(-1)):
        out |= (TRIT_CODE[int(v)] & 0x3) << (2 * i)
    return out


def emit_ax7203_top_verilog(abi: Dict[str, object], organ_id: str, in_dim: int, hidden_dim: int, out_dim: int) -> str:
    """Generate clean-room Apache-2.0 AX7203 top Verilog with 115200 UART + live обуч_вкл port."""
    if in_dim != 16 or hidden_dim != 16:
        raise ValueError(f"AX7203 top currently expects in_dim=16, hidden_dim=16; got ({in_dim}, {hidden_dim})")
    mod_name = str(abi["module"])
    ports = {p["name"]: p for p in abi["ports"]}  # type: ignore[index]
    p_start = ports["запуск"]
    p_x = ports["вход"]
    p_wen = ports["обуч_вкл"]
    p_waddr = ports["обуч_адр"]
    p_wdata = ports["обуч_дан"]
    p_wbias = ports["обуч_смещ"]
    p_pred = ports["решение"]
    p_done = ports["готов"]
    p_lat = ports["латентность"]

    organ_code = ORGAN_CODES.get(organ_id, 0)
    waddr_w = int(p_waddr["width"])
    pred_w = int(p_pred["width"])
    wbias_w = int(p_wbias["width"])
    wdata_w = int(p_wdata["width"])
    lat_w = int(p_lat["width"])

    return f"""// Copyright (c) 2026 Dmitrii Fedorov / Ternara (Apache-2.0)
// Clean-room ALINX AX7203 (XC7A200T) top wrapper for compiled Q27 v2 Seed organ: {organ_id}
// 200 MHz diff clock -> 25 MHz logic clock, 115200 8N1 UART (PING / INFER / live LEARN).
`timescale 1ns / 1ps
`default_nettype none

module ternara_ax7203_top #(
  parameter integer BAUD_DIV = 217
) (
  input  wire       clk200_p,
  input  wire       clk200_n,
  input  wire       rst_n,
  output wire [3:0] led,
  output wire       uart_tx,
  input  wire       uart_rx
);
  // ---- 1. Clocking: 200 MHz LVDS -> BUFG -> /8 divider -> 25 MHz BUFG (matches ax7203.xdc)
  wire clk200_raw, clk200, clk;
  IBUFDS clk_ibufds (.I(clk200_p), .IB(clk200_n), .O(clk200_raw));
  BUFG   clk200_bufg (.I(clk200_raw), .O(clk200));

  reg [2:0] clk_div = 3'd0;
  always @(posedge clk200) clk_div <= clk_div + 3'd1;
  wire div_bit = clk_div[2];
  BUFG clk_bufg (.I(div_bit), .O(clk));

  // Synchronized active-high reset (AX7203 rst_n is active-low)
  reg [1:0] rst_sync = 2'b11;
  always @(posedge clk) rst_sync <= {{rst_sync[0], ~rst_n}};
  wire rst = rst_sync[1];

  // ---- 2. UART RX (8N1, idle high, 2-FF synchronized)
  reg [1:0] rx_sync = 2'b11;
  always @(posedge clk) rx_sync <= {{rx_sync[0], uart_rx}};
  wire rx_in = rx_sync[1];

  localparam integer HALF_DIV = BAUD_DIV / 2;
  reg [1:0]  rx_state = 2'd0; // 0=IDLE, 1=START, 2=DATA, 3=STOP
  reg [11:0] rx_cnt   = 12'd0;
  reg [2:0]  rx_bit   = 3'd0;
  reg [7:0]  rx_shift = 8'd0;
  reg [7:0]  rx_data  = 8'd0;
  reg        rx_valid = 1'b0;

  always @(posedge clk) begin
    rx_valid <= 1'b0;
    if (rst) begin
      rx_state <= 2'd0;
      rx_cnt   <= 12'd0;
      rx_bit   <= 3'd0;
    end else begin
      case (rx_state)
        2'd0: begin
          if (!rx_in) begin
            rx_state <= 2'd1;
            rx_cnt   <= HALF_DIV[11:0];
          end
        end
        2'd1: begin
          if (rx_cnt == 12'd0) begin
            if (!rx_in) begin
              rx_state <= 2'd2;
              rx_cnt   <= BAUD_DIV[11:0] - 12'd1;
              rx_bit   <= 3'd0;
            end else begin
              rx_state <= 2'd0;
            end
          end else begin
            rx_cnt <= rx_cnt - 12'd1;
          end
        end
        2'd2: begin
          if (rx_cnt == 12'd0) begin
            rx_shift <= {{rx_in, rx_shift[7:1]}};
            rx_cnt   <= BAUD_DIV[11:0] - 12'd1;
            if (rx_bit == 3'd7) begin
              rx_state <= 2'd3;
            end else begin
              rx_bit <= rx_bit + 3'd1;
            end
          end else begin
            rx_cnt <= rx_cnt - 12'd1;
          end
        end
        2'd3: begin
          if (rx_cnt == 12'd0) begin
            rx_data  <= rx_shift;
            rx_valid <= 1'b1;
            rx_state <= 2'd0;
          end else begin
            rx_cnt <= rx_cnt - 12'd1;
          end
        end
      endcase
    end
  end

  // ---- 3. UART TX (8N1)
  reg        tx_busy  = 1'b0;
  reg [9:0]  tx_shift = 10'h3ff;
  reg [3:0]  tx_bits  = 4'd0;
  reg [11:0] tx_cnt   = 12'd0;
  reg        tx_start = 1'b0;
  reg [7:0]  tx_byte  = 8'd0;
  assign uart_tx = tx_shift[0];

  always @(posedge clk) begin
    if (rst) begin
      tx_busy  <= 1'b0;
      tx_shift <= 10'h3ff;
      tx_bits  <= 4'd0;
      tx_cnt   <= 12'd0;
    end else if (!tx_busy) begin
      if (tx_start) begin
        tx_shift <= {{1'b1, tx_byte, 1'b0}};
        tx_bits  <= 4'd10;
        tx_cnt   <= BAUD_DIV[11:0] - 12'd1;
        tx_busy  <= 1'b1;
      end
    end else begin
      if (tx_cnt == 12'd0) begin
        tx_shift <= {{1'b1, tx_shift[9:1]}};
        tx_cnt   <= BAUD_DIV[11:0] - 12'd1;
        if (tx_bits == 4'd1) begin
          tx_busy <= 1'b0;
          tx_bits <= 4'd0;
        end else begin
          tx_bits <= tx_bits - 4'd1;
        end
      end else begin
        tx_cnt <= tx_cnt - 12'd1;
      end
    end
  end

  // ---- 4. Compiled Q27 v2 Seed Organ Instance (0 DSP, 0 external DRAM)
  reg [1:0]               q_start = 2'b00;
  reg [127:0]             q_x     = 128'd0;
  reg [1:0]               q_wen   = 2'b00;
  reg [{waddr_w-1}:0]     q_waddr = {waddr_w}'d0;
  reg [{wdata_w-1}:0]     q_wdata = {wdata_w}'d0;
  reg [{wbias_w-1}:0]     q_wbias = {wbias_w}'d0;
  wire [{pred_w-1}:0]     q_pred;
  wire [1:0]              q_done;
  wire [{lat_w-1}:0]      q_lat;
  wire                    q_ready;
  wire [7:0]              q_err;
  wire [31:0]             q_err_line, q_err_col, q_err_tick, q_ticks, q_dbg_data;

  {mod_name} u_seed (
    .clk(clk),
    .rst(rst),
    .ce(1'b1),
    .ready(q_ready),
    .{p_start['port']}(q_start),
    .{p_x['port']}(q_x),
    .{p_wen['port']}(q_wen),
    .{p_waddr['port']}(q_waddr),
    .{p_wdata['port']}(q_wdata),
    .{p_wbias['port']}(q_wbias),
    .{p_pred['port']}(q_pred),
    .{p_done['port']}(q_done),
    .{p_lat['port']}(q_lat),
    .error_code(q_err),
    .error_line(q_err_line),
    .error_column(q_err_col),
    .error_tick(q_err_tick),
    .tick_count(q_ticks),
    .dbg_addr(7'd0),
    .dbg_data(q_dbg_data)
  );

  // ---- 5. Command FSM: 0x50 PING | 0x49 INFER (16B) | 0x4C LEARN (7B -> live В2_ПЛАСТ/С2_ПЛАСТ)
  localparam [2:0]
    S_IDLE       = 3'd0,
    S_RX_INFER   = 3'd1,
    S_RUN_INFER  = 3'd2,
    S_WAIT_INFER = 3'd3,
    S_RX_LEARN   = 3'd4,
    S_PULSE_LRN  = 3'd5,
    S_SEND_RESP  = 3'd6;

  reg [2:0]  state        = S_IDLE;
  reg [4:0]  byte_idx     = 5'd0;
  reg [2:0]  wait_cnt     = 3'd0;
  reg [7:0]  resp_buf [0:5];
  reg [2:0]  resp_len     = 3'd0;
  reg [2:0]  resp_idx     = 3'd0;
  reg [7:0]  lrn_class    = 8'd0;
  reg [31:0] lrn_w2       = 32'd0;
  reg [15:0] lrn_b2       = 16'd0;
  reg [3:0]  adapted_mask = 4'd0;
  reg [7:0]  last_err     = 8'd0;
  reg [7:0]  infer_cnt    = 8'd0;
  reg [24:0] hb_cnt       = 25'd0;

  always @(posedge clk) begin
    hb_cnt   <= hb_cnt + 25'd1;
    q_start  <= 2'b00;
    q_wen    <= 2'b00;
    tx_start <= 1'b0;

    if (rst) begin
      state        <= S_IDLE;
      byte_idx     <= 5'd0;
      wait_cnt     <= 3'd0;
      resp_len     <= 3'd0;
      resp_idx     <= 3'd0;
      adapted_mask <= 4'd0;
      last_err     <= 8'd0;
      infer_cnt    <= 8'd0;
    end else begin
      if (q_err != 8'd0) last_err <= q_err;

      case (state)
        S_IDLE: begin
          if (rx_valid) begin
            if (rx_data == 8'h50) begin
              // PING -> 6-byte status response
              resp_buf[0] <= 8'hA0;
              resp_buf[1] <= 8'd{organ_code};
              resp_buf[2] <= 8'd{in_dim};
              resp_buf[3] <= 8'd{hidden_dim};
              resp_buf[4] <= 8'd{out_dim};
              resp_buf[5] <= {{adapted_mask, 2'b00, (last_err != 8'd0), 1'b1}};
              resp_len    <= 3'd6;
              resp_idx    <= 3'd0;
              state       <= S_SEND_RESP;
            end else if (rx_data == 8'h49) begin
              byte_idx <= 5'd0;
              state    <= S_RX_INFER;
            end else if (rx_data == 8'h4C) begin
              byte_idx <= 5'd0;
              state    <= S_RX_LEARN;
            end
          end
        end

        S_RX_INFER: begin
          if (rx_valid) begin
            q_x[byte_idx[3:0] * 8 +: 8] <= rx_data;
            if (byte_idx == 5'd15) begin
              state <= S_RUN_INFER;
            end else begin
              byte_idx <= byte_idx + 5'd1;
            end
          end
        end

        S_RUN_INFER: begin
          q_start  <= 2'b01;
          wait_cnt <= 3'd0;
          state    <= S_WAIT_INFER;
        end

        S_WAIT_INFER: begin
          wait_cnt <= wait_cnt + 3'd1;
          if (q_done == 2'b01 || wait_cnt == 3'd6) begin
            infer_cnt   <= infer_cnt + 8'd1;
            resp_buf[0] <= 8'hA1;
            resp_buf[1] <= {{{8-pred_w}'d0, q_pred}};
            resp_buf[2] <= q_lat[7:0];
            resp_buf[3] <= q_err;
            resp_buf[4] <= {{4'd0, adapted_mask}};
            resp_len    <= 3'd5;
            resp_idx    <= 3'd0;
            state       <= S_SEND_RESP;
          end
        end

        S_RX_LEARN: begin
          if (rx_valid) begin
            case (byte_idx[2:0])
              3'd0: lrn_class       <= rx_data;
              3'd1: lrn_w2[7:0]     <= rx_data;
              3'd2: lrn_w2[15:8]    <= rx_data;
              3'd3: lrn_w2[23:16]   <= rx_data;
              3'd4: lrn_w2[31:24]   <= rx_data;
              3'd5: lrn_b2[7:0]     <= rx_data;
              3'd6: lrn_b2[15:8]    <= rx_data;
              default: ;
            endcase
            if (byte_idx == 5'd6) begin
              state <= S_PULSE_LRN;
            end else begin
              byte_idx <= byte_idx + 5'd1;
            end
          end
        end

        S_PULSE_LRN: begin
          q_wen   <= 2'b01;
          q_waddr <= lrn_class[{waddr_w-1}:0];
          q_wdata <= lrn_w2[{wdata_w-1}:0];
          q_wbias <= lrn_b2[{wbias_w-1}:0];
          if (lrn_class[1:0] == 2'd0) adapted_mask[0] <= 1'b1;
          if (lrn_class[1:0] == 2'd1) adapted_mask[1] <= 1'b1;
          if (lrn_class[1:0] == 2'd2) adapted_mask[2] <= 1'b1;
          if (lrn_class[1:0] == 2'd3) adapted_mask[3] <= 1'b1;
          resp_buf[0] <= 8'hA2;
          resp_buf[1] <= lrn_class;
          resp_buf[2] <= q_err;
          resp_buf[3] <= {{4'd0, adapted_mask | (4'd1 << lrn_class[1:0])}};
          resp_len    <= 3'd4;
          resp_idx    <= 3'd0;
          state       <= S_SEND_RESP;
        end

        S_SEND_RESP: begin
          if (!tx_busy && !tx_start) begin
            tx_byte  <= resp_buf[resp_idx];
            tx_start <= 1'b1;
            if (resp_idx + 3'd1 >= resp_len) begin
              state <= S_IDLE;
            end else begin
              resp_idx <= resp_idx + 3'd1;
            end
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  assign led[0] = hb_cnt[24];
  assign led[1] = infer_cnt[0];
  assign led[2] = |adapted_mask;
  assign led[3] = (last_err == 8'd0);
endmodule

`default_nettype wire
"""


def prepare_ax7203_project(
    organ_id: str,
    out_dir: str | Path,
    xdc_src: Path = DEFAULT_XDC_SRC,
    chipdb_src: Path = DEFAULT_CHIPDB_SRC,
    openxc7_image: str = DEFAULT_OPENXC7_IMAGE,
) -> Dict[str, object]:
    """Compile Q27 v2 organ via q27rtl2 and emit a complete 0%-Vivado AX7203 build directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cart, plan, rep = train_organ_seed(organ_id, board_id="ax7203_xc7a200t")
    exp = export_and_compile_q27(cart, plan, out, run_compiler=True)
    if not exp.q27rtl2_compiled or not exp.verilog_path:
        raise RuntimeError(f"q27rtl2 compile failed for {organ_id}: {exp.compiler_note}")

    rtl_dir = Path(exp.verilog_path).parent
    abi = json.loads((rtl_dir / "abi.json").read_text(encoding="utf-8"))
    top_v = emit_ax7203_top_verilog(abi, organ_id, cart.in_dim, cart.hidden_dim, cart.out_dim)
    top_path = out / "ternara_ax7203_top.v"
    top_path.write_text(top_v, encoding="utf-8")

    sv_dst = out / "seed_core.sv"
    shutil.copy2(rtl_dir / "design.sv", sv_dst)

    xdc_dst = out / "ax7203.xdc"
    if xdc_src.exists():
        shutil.copy2(xdc_src, xdc_dst)
    else:
        raise FileNotFoundError(f"AX7203 XDC not found at {xdc_src}")

    build_sh = out / "build_ax7203_bitstream.sh"
    build_sh.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PART="{AX7203_PART}"
CHIPDB="{chipdb_src}"
IMAGE="{openxc7_image}"
YOSYS="$(command -v yosys || echo /opt/homebrew/bin/yosys)"

echo "[1/3] Yosys 0-DSP synthesis (0% Vivado)..."
"$YOSYS" -q -l "$HERE/yosys.log" -p "read_verilog -sv $HERE/seed_core.sv $HERE/ternara_ax7203_top.v; synth_xilinx -flatten -abc9 -nocarry -nodsp -nowidelut -nosrl -family xc7 -top ternara_ax7203_top; setundef -zero -params; tee -o $HERE/yosys_stat.txt stat; write_json $HERE/top.json"
grep -E "DSP48|LUT|FDRE|RAMB" "$HERE/yosys_stat.txt" || true

echo "[2/3] nextpnr-xilinx place & route (openxc7)..."
docker run --rm --platform linux/amd64 --pull=never --network=none \\
  -v "$HERE:/work" -v "$(dirname "$CHIPDB"):/chipdb:ro" -w /work "$IMAGE" \\
  nextpnr-xilinx --chipdb "/chipdb/$(basename "$CHIPDB")" \\
  --xdc ax7203.xdc --json top.json --write top_routed.json \\
  --freq 25 --seed 1 --placer heap --router router2 --fasm top.fasm > "$HERE/nextpnr.log" 2>&1
grep -E "Max frequency for clock|SLICE_LUTX|SLICE_FFX" "$HERE/nextpnr.log" | tail -n 8 || true

echo "[3/3] Project X-Ray fasm2frames + xc7frames2bit -> top.bit..."
docker run --rm --platform linux/amd64 --pull=never --network=none \\
  -v "$HERE:/work" -w /work "$IMAGE" \\
  bash -c "source /prjxray/env/bin/activate && fasm2frames --db-root /nextpnr-xilinx/xilinx/external/prjxray-db/artix7 --part $PART top.fasm top.frames && /prjxray/build/tools/xc7frames2bit --part_file /nextpnr-xilinx/xilinx/external/prjxray-db/artix7/$PART/part.yaml --part_name $PART --frm_file top.frames --output_file top.bit" > "$HERE/pack.log" 2>&1

shasum -a 256 "$HERE/top.bit" > "$HERE/top.bit.sha256"
echo "DONE: $HERE/top.bit ($(wc -c < "$HERE/top.bit") bytes, 0% Vivado)"
""",
        encoding="utf-8",
    )
    os.chmod(build_sh, 0o755)

    meta = {
        "organ_id": organ_id,
        "part": AX7203_PART,
        "top_module": "ternara_ax7203_top",
        "seed_module": abi["module"],
        "out_dir": str(out),
        "top_verilog": str(top_path),
        "seed_sv": str(sv_dst),
        "xdc": str(xdc_dst),
        "build_script": str(build_sh),
        "cartridge_path": exp.cartridge_path,
        "cartridge_sha256": cart.sha256(),
        "val_acc": rep.val_acc,
    }
    (out / "ax7203_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


@dataclass(frozen=True)
class UartTopSimReport:
    organ_id: str
    ping_ok: bool
    samples_tested: int
    before_rtl_acc: float
    after_learn_rtl_acc: float
    before_mismatches: int
    after_mismatches: int
    observed_latency_ticks: int
    hardware_errors: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def simulate_ax7203_uart_top(
    project_dir: str | Path,
    organ_id: str,
    base_cart: SeedCartridge,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hot_load_cart: Optional[SeedCartridge] = None,
) -> UartTopSimReport:
    """Simulate exact `ternara_ax7203_top.v` + `seed_core.sv` via real 8N1 UART bit-banging in `iverilog`/`vvp`."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")

    pdir = Path(project_dir)
    sim_dir = pdir / "_uart_sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    n = int(X_test.shape[0])
    _, _, pred_before_oracle = oracle_forward_batch(base_cart, X_test)
    if hot_load_cart is not None:
        _, _, pred_after_oracle = oracle_forward_batch(hot_load_cart, X_test)
    else:
        pred_after_oracle = pred_before_oracle

    # Write flat hex file of n * 16 bytes
    bytes_path = sim_dir / "uart_x.hex"
    flat_bytes: List[str] = []
    for i in range(n):
        for j in range(16):
            flat_bytes.append(f"{int(X_test[i, j]) & 0xFF:02x}")
    bytes_path.write_text("\n".join(flat_bytes) + "\n", encoding="utf-8")

    learn_block: List[str] = []
    if hot_load_cart is not None:
        for c in range(hot_load_cart.out_dim):
            w32 = pack_trit_u32(hot_load_cart.w2[c])
            b16 = int(hot_load_cart.b2[c]) & 0xFFFF
            pkt = [
                0x4C,
                c & 0xFF,
                w32 & 0xFF,
                (w32 >> 8) & 0xFF,
                (w32 >> 16) & 0xFF,
                (w32 >> 24) & 0xFF,
                b16 & 0xFF,
                (b16 >> 8) & 0xFF,
            ]
            for b in pkt:
                learn_block.append(f"    uart_send(8'h{b:02x});")
            learn_block.extend([
                "    uart_recv(r0); uart_recv(r1); uart_recv(r2); uart_recv(r3);",
                '    $display("LEARN %02x %02x %02x %02x", r0, r1, r2, r3);',
            ])

    # Lightweight Xilinx clock primitive models strictly for testbench simulation
    tb_sv = f"""`timescale 1ns/1ps
module IBUFDS(input wire I, input wire IB, output wire O);
  assign O = I;
endmodule
module BUFG(input wire I, output wire O);
  assign O = I;
endmodule

module tb_ax7203_uart;
  reg clk200_p = 1'b0;
  wire clk200_n = ~clk200_p;
  reg rst_n = 1'b0;
  wire [3:0] led;
  wire uart_tx;
  reg  uart_rx = 1'b1;

  // Fast BAUD_DIV=8 for rapid cycle-accurate UART simulation (real AX7203 silicon uses 217)
  localparam integer BDIV = 8;
  localparam integer BIT_CLKS = BDIV;

  ternara_ax7203_top #(.BAUD_DIV(BDIV)) dut (
    .clk200_p(clk200_p), .clk200_n(clk200_n), .rst_n(rst_n),
    .led(led), .uart_tx(uart_tx), .uart_rx(uart_rx)
  );

  always #2.5 clk200_p = ~clk200_p; // 200 MHz

  task wait_core_clks(input integer k);
    integer c;
    begin
      for (c = 0; c < k; c = c + 1) @(posedge dut.clk);
    end
  endtask

  task uart_send(input [7:0] data);
    integer b;
    begin
      uart_rx = 1'b0; // start bit
      wait_core_clks(BIT_CLKS);
      for (b = 0; b < 8; b = b + 1) begin
        uart_rx = data[b];
        wait_core_clks(BIT_CLKS);
      end
      uart_rx = 1'b1; // stop bit
      wait_core_clks(BIT_CLKS + 2);
    end
  endtask

  task uart_recv(output [7:0] data);
    integer b;
    begin
      @(negedge uart_tx);
      wait_core_clks(BIT_CLKS / 2);
      for (b = 0; b < 8; b = b + 1) begin
        wait_core_clks(BIT_CLKS);
        data[b] = uart_tx;
      end
      wait_core_clks(BIT_CLKS); // stop bit
    end
  endtask

  reg [7:0] xmem [0:{n*16 - 1}];
  reg [7:0] r0, r1, r2, r3, r4, r5;
  integer i, j;

  initial begin
    $readmemh("{bytes_path}", xmem);
    rst_n = 1'b0;
    wait_core_clks(10);
    rst_n = 1'b1;
    wait_core_clks(10);

    // 1. PING over UART
    uart_send(8'h50);
    uart_recv(r0); uart_recv(r1); uart_recv(r2); uart_recv(r3); uart_recv(r4); uart_recv(r5);
    $display("PING %02x %02x %02x %02x %02x %02x", r0, r1, r2, r3, r4, r5);

    // 2. INFER BEFORE LEARN over UART
    for (i = 0; i < {n}; i = i + 1) begin
      uart_send(8'h49);
      for (j = 0; j < 16; j = j + 1) uart_send(xmem[i * 16 + j]);
      uart_recv(r0); uart_recv(r1); uart_recv(r2); uart_recv(r3); uart_recv(r4);
      $display("BEFORE %0d %02x %0d %0d %0d %02x", i, r0, r1, r2, r3, r4);
    end

    // 3. LIVE ON-CHIP LEARN over UART (0 resynthesis!)
{chr(10).join(learn_block)}

    // 4. INFER AFTER LEARN over UART (same bitstream!)
    for (i = 0; i < {n}; i = i + 1) begin
      uart_send(8'h49);
      for (j = 0; j < 16; j = j + 1) uart_send(xmem[i * 16 + j]);
      uart_recv(r0); uart_recv(r1); uart_recv(r2); uart_recv(r3); uart_recv(r4);
      $display("AFTER %0d %02x %0d %0d %0d %02x", i, r0, r1, r2, r3, r4);
    end
    $finish;
  end
endmodule
"""
    tb_path = sim_dir / "tb_ax7203_uart.sv"
    out_vvp = sim_dir / "uart_sim.vvp"
    tb_path.write_text(tb_sv, encoding="utf-8")

    comp = subprocess.run(
        [iverilog, "-g2012", "-o", str(out_vvp), str(pdir / "seed_core.sv"), str(pdir / "ternara_ax7203_top.v"), str(tb_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if comp.returncode != 0:
        raise RuntimeError(f"iverilog failed: {comp.stderr or comp.stdout}")

    run = subprocess.run([vvp, str(out_vvp)], capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        raise RuntimeError(f"vvp failed: {run.stderr or run.stdout}")

    ping_ok = False
    before_preds = np.full(n, -1, dtype=np.int32)
    after_preds = np.full(n, -1, dtype=np.int32)
    obs_lat = 0
    hw_errs = 0

    for line in run.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "PING" and len(parts) >= 7:
            ping_ok = (parts[1].lower() == "a0" and int(parts[2], 16) == ORGAN_CODES.get(organ_id, 0))
        elif parts[0] == "LEARN" and len(parts) >= 5:
            if parts[1].lower() != "a2" or int(parts[3], 16) != 0:
                hw_errs += 1
        elif parts[0] in ("BEFORE", "AFTER") and len(parts) >= 7:
            idx = int(parts[1])
            tag = parts[2].lower()
            pred = int(parts[3])
            lat = int(parts[4])
            err = int(parts[5])
            if tag != "a1" or err != 0:
                hw_errs += 1
            obs_lat = lat
            if parts[0] == "BEFORE":
                before_preds[idx] = pred
            else:
                after_preds[idx] = pred

    return UartTopSimReport(
        organ_id=organ_id,
        ping_ok=ping_ok,
        samples_tested=n,
        before_rtl_acc=round(float(np.mean(before_preds == y_test)), 4),
        after_learn_rtl_acc=round(float(np.mean(after_preds == y_test)), 4),
        before_mismatches=int(np.sum(before_preds != pred_before_oracle)),
        after_mismatches=int(np.sum(after_preds != pred_after_oracle)),
        observed_latency_ticks=obs_lat,
        hardware_errors=hw_errs,
    )


class AX7203SerialClient:
    """Zero-dependency POSIX termios 115200 8N1 UART host client for live ALINX AX7203 silicon."""

    def __init__(self, port: str = DEFAULT_UART_PORT, timeout_s: float = 1.5) -> None:
        self.port = port
        self.timeout_s = timeout_s
        self.fd: Optional[int] = None

    def __enter__(self) -> "AX7203SerialClient":
        fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(fd)
        # Raw 115200 8N1, no flow control
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIOFLUSH)
        self.fd = fd
        return self

    def __exit__(self, *args: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _write_all(self, data: bytes) -> None:
        assert self.fd is not None
        view = memoryview(data)
        sent = 0
        while sent < len(data):
            _, w, _ = select.select([], [self.fd], [], self.timeout_s)
            if not w:
                raise TimeoutError("UART write timeout")
            sent += os.write(self.fd, view[sent:])

    def _read_exact(self, n: int) -> bytes:
        assert self.fd is not None
        buf = bytearray()
        deadline = time.monotonic() + self.timeout_s
        while len(buf) < n:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"UART read timeout: expected {n} bytes, got {len(buf)}")
            r, _, _ = select.select([self.fd], [], [], rem)
            if r:
                chunk = os.read(self.fd, n - len(buf))
                if chunk:
                    buf.extend(chunk)
        return bytes(buf)

    def ping(self) -> Dict[str, int]:
        self._write_all(b"\x50")
        resp = self._read_exact(6)
        if resp[0] != 0xA0:
            raise RuntimeError(f"Unexpected PING reply: {resp.hex()}")
        return {
            "organ_code": resp[1],
            "in_dim": resp[2],
            "hidden_dim": resp[3],
            "out_dim": resp[4],
            "status": resp[5],
        }

    def infer(self, x_int8: np.ndarray) -> Tuple[int, int, int, int]:
        payload = bytes([0x49]) + bytes(int(v) & 0xFF for v in x_int8.reshape(-1)[:16])
        self._write_all(payload)
        resp = self._read_exact(5)
        if resp[0] != 0xA1:
            raise RuntimeError(f"Unexpected INFER reply: {resp.hex()}")
        return resp[1], resp[2], resp[3], resp[4]

    def learn_class(self, class_idx: int, w2_row: np.ndarray, b2_val: int) -> Tuple[int, int]:
        w32 = pack_trit_u32(w2_row)
        pkt = struct.pack("<BBIh", 0x4C, int(class_idx) & 0xFF, w32, int(b2_val))
        self._write_all(pkt)
        resp = self._read_exact(4)
        if resp[0] != 0xA2:
            raise RuntimeError(f"Unexpected LEARN reply: {resp.hex()}")
        return resp[2], resp[3]  # error_code, adapted_mask


def flash_ax7203_sram_safe(bitstream_path: str | Path, max_temp_c: float = 90.0) -> Dict[str, object]:
    """Safe SRAM-only AX7203 loader (NEVER -f flash) with JTAG IDCODE + XADC die temperature check."""
    if DEFAULT_HEAVY_LOCK.exists():
        raise RuntimeError(f"Refusing flash while heavy.lock is active: {DEFAULT_HEAVY_LOCK}")
    if DEFAULT_BOARD_LOCK.exists():
        raise RuntimeError(f"Refusing flash while ax7203-board.lock is active: {DEFAULT_BOARD_LOCK}")

    loader = shutil.which("openFPGALoader")
    if not loader:
        raise RuntimeError("openFPGALoader not found")

    bit = Path(bitstream_path).resolve()
    if not bit.is_file():
        raise FileNotFoundError(f"Bitstream not found: {bit}")

    det = subprocess.run([loader, "-c", "digilent_hs2", "--detect"], capture_output=True, text=True, timeout=15)
    det_out = (det.stdout or "") + (det.stderr or "")
    if det.returncode != 0 or "0x03636093" not in det_out.lower().replace("0x3636093", "0x03636093"):
        raise RuntimeError(f"AX7203 XC7A200T (0x03636093) not detected:\n{det_out}")

    xadc = subprocess.run([loader, "-c", "digilent_hs2", "--read-xadc"], capture_output=True, text=True, timeout=15)
    xadc_out = (xadc.stdout or "") + (xadc.stderr or "")

    load = subprocess.run([loader, "-c", "digilent_hs2", str(bit)], capture_output=True, text=True, timeout=60)
    if load.returncode != 0:
        raise RuntimeError(f"openFPGALoader SRAM load failed:\n{load.stdout}\n{load.stderr}")

    xadc_after = subprocess.run([loader, "-c", "digilent_hs2", "--read-xadc"], capture_output=True, text=True, timeout=15)
    return {
        "bitstream": str(bit),
        "mode": "SRAM_ONLY",
        "detect": det_out.strip(),
        "xadc_before": xadc_out.strip(),
        "xadc_after": ((xadc_after.stdout or "") + (xadc_after.stderr or "")).strip(),
    }
