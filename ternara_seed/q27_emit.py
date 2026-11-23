"""Genuine Q27 v2 (`версия 2`) RTL code generator & real `q27rtl2` compiler bridge (0% Vivado, 0 DSP).

Generates valid, deterministic Q27 v2 hardware cores featuring:
  - Frozen 0-DSP ternary feature extractor: `Σ(В1[j] ⊗ вход) + С1[j]` + `измерить(... порог ...)`
  - Live on-chip plastic output weight bank (`В2_ПЛАСТ`) with hardware write strobe (`обуч_вкл`)
    so newly attached physical modules can update BRAM weights directly in silicon without resynthesis!
  - Hardware latency cycle counter (`тики`) and deterministic argmax decision output (`решение`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .grower import GrownPlan
from .pack5 import SeedCartridge, pack5_encode

DEFAULT_Q27_WORKSPACE = Path("/Volumes/SSDD/Workspaces/q27-rtl-v02")


def _fmt_flat_int_list(arr: np.ndarray) -> str:
    return ", ".join(str(int(v)) for v in arr.reshape(-1))


def emit_q27_source(cart: SeedCartridge, plan: GrownPlan) -> str:
    """Generate genuine compilable `Q27 версия 2` hardware source for a grown Seed organ."""
    h = cart.hidden_dim
    m = cart.in_dim
    o = cart.out_dim
    t1 = abs(int(cart.thresh1))

    b1_min = min(-256, int(cart.b1.min()) - 1)
    b1_max = max(255, int(cart.b1.max()) + 1)
    b2_min = min(-256, int(cart.b2.min()) - 1)
    b2_max = max(255, int(cart.b2.max()) + 1)

    lines: List[str] = [
        "версия 2",
        f"# Ternara Seed-to-Silicon auto-grown Q27 organ: {cart.organ}",
        f"# Target board: {plan.board_name} ({plan.board_id}, hive={plan.hive_count})",
        f"# Budget: 0 DSP, 0 DRAM, {plan.parallel_cores} parallel cores, {plan.pack5_bytes_per_core} B/core pack5, {plan.latency_us} us @ {plan.clock_mhz} MHz",
        f"# Copyright (c) 2026 Dmitrii Fedorov / Ternara (Apache-2.0)",
        "",
        "ввод запуск : трит",
        f"ввод вход[{m}] : амплитуда(-128..127)",
        "ввод обуч_вкл : трит",
        f"ввод обуч_адр : амплитуда(0..{max(1, o - 1)})",
        f"ввод обуч_дан[{h}] : трит",
        "",
        f"таблица В1[{h}] : трит[{m}] = [{_fmt_flat_int_list(cart.w1)}]",
        f"таблица С1[{h}] : амплитуда({b1_min}..{b1_max}) = [{_fmt_flat_int_list(cart.b1)}]",
        f"таблица В2_БАЗА[{o}] : трит[{h}] = [{_fmt_flat_int_list(cart.w2)}]",
        f"таблица С2[{o}] : амплитуда({b2_min}..{b2_max}) = [{_fmt_flat_int_list(cart.b2)}]",
        "",
        f"регистр скрыт[{h}] : амплитуда(-1..1) = 0",
        f"регистр В2_ПЛАСТ[{o}] : трит[{h}] = 0",
        f"регистр адапт[{o}] : амплитуда(0..1) = 0",
        f"регистр счет[{o}] : амплитуда(-4096..4095) = 0",
        "регистр фаза : амплитуда(0..2) = 0",
        "регистр тики : амплитуда(0..16777215) = 0",
        f"вывод решение : амплитуда(0..{max(1, o - 1)}) = 0",
        "вывод готов : трит = -1",
        "вывод латентность : амплитуда(0..16777215) = 0",
        "",
        "такт шаг {",
        "    если обуч_вкл == 1 {",
        "        В2_ПЛАСТ[обуч_адр] := обуч_дан",
        "        адапт[обуч_адр] := 1",
        "    }",
        "    если фаза == 0 {",
        "        готов := -1",
        "        если запуск == 1 {",
    ]

    # Layer 1: 0-DSP ternary dot + native Q27 ternary measure
    for j in range(h):
        lines.append(
            f"            скрыт[{j}] := измерить(Σ(В1[{j}] ⊗ вход) + С1[{j}] смещение 0 порог {t1})"
        )
    lines.extend([
        "            фаза := 1",
        "            тики := 1",
        "        }",
        "    } иначе если фаза == 1 {",
        "        готов := -1",
    ])

    # Layer 2: Plastic on-chip BRAM output head (uses live В2_ПЛАСТ once adapted, else В2_БАЗА)
    for c in range(o):
        lines.append(
            f"        если адапт[{c}] == 1 {{ счет[{c}] := Σ(В2_ПЛАСТ[{c}] ⊗ скрыт) + С2[{c}] }} "
            f"иначе {{ счет[{c}] := Σ(В2_БАЗА[{c}] ⊗ скрыт) + С2[{c}] }}"
        )
    lines.extend([
        "        фаза := 2",
        "        тики := тики + 1",
        "    } иначе {",
    ])

    # Deterministic lowest-index argmax tree in Q27
    # Build clean nested `если` comparisons matching np.argmax tie-breaking
    if o == 2:
        lines.append("        решение := если(счет[1] > счет[0], 1, 0)")
    elif o == 3:
        lines.append(
            "        решение := если((счет[1] > счет[0]) и (счет[1] >= счет[2]), 1, "
            "если((счет[2] > счет[0]) и (счет[2] > счет[1]), 2, 0))"
        )
    elif o == 4:
        lines.append(
            "        решение := если((счет[0] >= счет[1]) и (счет[0] >= счет[2]) и (счет[0] >= счет[3]), 0, "
            "если((счет[1] >= счет[2]) и (счет[1] >= счет[3]), 1, "
            "если(счет[2] >= счет[3], 2, 3)))"
        )
    else:
        expr = str(o - 1)
        for c in range(o - 2, -1, -1):
            conds = " и ".join(f"(счет[{c}] >= счет[{k}])" for k in range(c + 1, o))
            expr = f"если({conds}, {c}, {expr})"
        lines.append(f"        решение := {expr}")

    lines.extend([
        "        готов := +1",
        "        латентность := тики + 1",
        "        фаза := 0",
        "    }",
        "}",
        "",
    ])
    return "\n".join(lines)


@dataclass(frozen=True)
class Q27ExportResult:
    q27_path: str
    pack5_hex_path: str
    cartridge_path: str
    plan_json_path: str
    q27rtl2_compiled: bool
    verilog_path: Optional[str]
    verilog_lines: int
    state_ff_bits: int
    expanded_operators: int
    overflow_checks: int
    double_writes: int
    compiler_note: str


def export_and_compile_q27(
    cart: SeedCartridge,
    plan: GrownPlan,
    out_dir: str | Path,
    run_compiler: bool = True,
    q27_workspace: Path = DEFAULT_Q27_WORKSPACE,
) -> Q27ExportResult:
    """Export Q27 v2 source, pack5 hex ROM, cartridge, plan JSON, and optionally compile via q27rtl2."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = f"{cart.organ}_{plan.board_id}"
    q27_path = out / f"{stem}.q27"
    q27_src = emit_q27_source(cart, plan)
    q27_path.write_text(q27_src, encoding="utf-8")

    # Export pack5 hex ROM for BRAM initialization
    p5_bytes = pack5_encode(cart.w1) + pack5_encode(cart.w2)
    hex_path = out / f"{stem}_pack5.hex"
    hex_path.write_text("\n".join(f"{b:02x}" for b in p5_bytes) + "\n", encoding="utf-8")

    cart_path = cart.save(out / f"{stem}.tseed.bin")
    plan_path = out / f"{stem}_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")

    compiled = False
    v_path: Optional[str] = None
    v_lines = 0
    state_ff_bits = 0
    expanded_operators = 0
    overflow_checks = 0
    double_writes = 0
    note = "compiler skipped (--no-compile)"

    if run_compiler:
        py_bin = q27_workspace / ".venv/bin/python"
        if not py_bin.exists():
            note = f"q27rtl2 venv not found at {py_bin}; Q27 source exported only"
        else:
            rtl_out = out / f"rtl_{stem}"
            if rtl_out.exists():
                shutil.rmtree(rtl_out)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(q27_workspace)
            proc = subprocess.run(
                [
                    str(py_bin),
                    "-m",
                    "q27rtl2",
                    "compile",
                    "--profile",
                    "q27-rtl-machine-v2",
                    "--out",
                    str(rtl_out),
                    str(q27_path),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                v_files = sorted(list(rtl_out.glob("*.sv")) + list(rtl_out.glob("*.v")))
                compiled = True
                if v_files:
                    v_path = str(v_files[0])
                    v_lines = sum(1 for _ in v_files[0].open("r", encoding="utf-8"))
                res_json = rtl_out / "resources.json"
                if res_json.exists():
                    rdata = json.loads(res_json.read_text(encoding="utf-8"))
                    state_ff_bits = int(rdata.get("state_ff_bits", 0))
                    overflow_checks = int(rdata.get("overflow_checks", 0))
                    for lim in rdata.get("limits", []):
                        if lim.get("name") == "expanded_operators":
                            expanded_operators = int(lim.get("value", 0))
                dw_txt = rtl_out / "lint-double-write.txt"
                if dw_txt.exists():
                    for ln in dw_txt.read_text(encoding="utf-8").splitlines():
                        if ln.startswith("double-write:"):
                            parts = ln.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                double_writes = int(parts[1])
                note = (
                    f"q27rtl2 PASS -> {v_files[0].name if v_files else 'rtl'} ({v_lines} lines, "
                    f"{state_ff_bits} state FFs, {expanded_operators} ops, 0 DSP, {overflow_checks} overflow checks, "
                    f"{double_writes} double-writes, 0% Vivado)"
                )
            else:
                err_tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                note = f"q27rtl2 exit={proc.returncode}: {' | '.join(err_tail)}"

    return Q27ExportResult(
        q27_path=str(q27_path),
        pack5_hex_path=str(hex_path),
        cartridge_path=str(cart_path),
        plan_json_path=str(plan_path),
        q27rtl2_compiled=compiled,
        verilog_path=v_path,
        verilog_lines=v_lines,
        state_ff_bits=state_ff_bits,
        expanded_operators=expanded_operators,
        overflow_checks=overflow_checks,
        double_writes=double_writes,
        compiler_note=note,
    )
