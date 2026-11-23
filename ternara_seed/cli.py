"""Command-line interface for Ternara Seed-to-Silicon (scan, grow, attach, mesh-sim, matrix)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .boards import BOARD_ORDER, BOARDS, get_board
from .grower import grow_seed_for_board
from .mesh import run_mesh_simulation
from .organs import ORGAN_ORDER, ORGANS, train_organ_seed
from .pack5 import verify_all_reserved_fault_bytes
from .plug_and_learn import MODULES, attach_and_adapt_module
from .q27_emit import export_and_compile_q27


def cmd_scan(args: argparse.Namespace) -> int:
    rows = []
    for bid in BOARD_ORDER:
        b = get_board(bid, hive_count=args.hive)
        rows.append({
            "board_id": b.board_id,
            "name": b.name,
            "family": b.family,
            "jtag_idcode": b.jtag_idcode,
            "lut": b.lut,
            "bram_blocks": b.bram_blocks,
            "bram_kb": round(b.total_bram_bytes / 1024.0, 1),
            "max_pack5_trits": b.max_pack5_trits,
            "clock_mhz": b.default_mhz,
            "est_power_w": b.est_power_w,
            "retail_usd": b.retail_usd,
            "qualification": b.qualification,
        })
    print(json.dumps({"hive_count": args.hive, "boards": rows}, indent=2))
    return 0


def cmd_grow(args: argparse.Namespace) -> int:
    organ_ids = ORGAN_ORDER if args.organ == "all" else [args.organ]
    results = []
    for oid in organ_ids:
        cart, plan, rep = train_organ_seed(oid, board_id=args.board, hive_count=args.hive)
        exp = export_and_compile_q27(
            cart, plan, out_dir=Path(args.out_dir) / oid, run_compiler=not args.no_compile
        )
        results.append({
            "organ": oid,
            "board": plan.board_id,
            "val_acc": rep.val_acc,
            "bit_exact_diff": rep.bit_exact_diff,
            "cartridge_bytes": rep.cartridge_bytes,
            "zero_fraction": rep.zero_fraction,
            "parallel_cores": plan.parallel_cores,
            "hot_cartridge_slots": plan.hot_cartridge_slots,
            "latency_us": plan.latency_us,
            "speedup_vs_cpu_p99": plan.speedup_vs_cpu_p99,
            "dsp_used": 0,
            "dram_bytes_used": 0,
            "est_power_w": plan.est_power_w,
            "q27_path": exp.q27_path,
            "q27rtl2_compiled": exp.q27rtl2_compiled,
            "verilog_lines": exp.verilog_lines,
            "state_ff_bits": exp.state_ff_bits,
            "expanded_operators": exp.expanded_operators,
            "overflow_checks": exp.overflow_checks,
            "double_writes": exp.double_writes,
            "compiler_note": exp.compiler_note,
            "sha256": rep.sha256,
        })
    print(json.dumps({"results": results}, indent=2))
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    mod = MODULES[args.module]
    cart, plan, _ = train_organ_seed(mod.target_organ, board_id=args.board)
    _, rep = attach_and_adapt_module(cart, args.module, clock_mhz=plan.clock_mhz)
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


def cmd_mesh_sim(args: argparse.Namespace) -> int:
    cart, _, _ = train_organ_seed(args.organ, board_id=args.board)
    _, rep = run_mesh_simulation(cart, honest_nodes=args.nodes, poison_nodes=args.poison)
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Build the complete verified 5 organs x 6 boards Seed-to-Silicon matrix + real q27rtl2 proofs."""
    rej, total_res = verify_all_reserved_fault_bytes()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    organs_summary = []
    for oid in ORGAN_ORDER:
        spec = ORGANS[oid]
        # Train & compile canonical AX7203 organ through real q27rtl2
        cart, ax_plan, rep = train_organ_seed(oid, board_id="ax7203_xc7a200t")
        exp = export_and_compile_q27(cart, ax_plan, out_dir=out_dir / oid, run_compiler=not args.no_compile)

        # Compute auto-grown resource scaling across all 6 boards + 4x ECP5 Hive
        board_fits = [grow_seed_for_board(spec, bid).to_dict() for bid in BOARD_ORDER]
        board_fits.append(grow_seed_for_board(spec, "ecp5_85f", hive_count=4).to_dict())

        # Run mesh proof on this organ
        _, mesh_rep = run_mesh_simulation(cart, honest_nodes=6, poison_nodes=2)

        organs_summary.append({
            "organ_id": oid,
            "title": spec.title,
            "tagline": spec.tagline,
            "domain": spec.domain,
            "default_bus": spec.default_bus,
            "class_names": list(spec.class_names),
            "val_acc": rep.val_acc,
            "bit_exact_diff": rep.bit_exact_diff,
            "cartridge_bytes": rep.cartridge_bytes,
            "total_trits_per_core": rep.total_trits,
            "zero_fraction": rep.zero_fraction,
            "sha256": rep.sha256,
            "q27rtl2_compiled": exp.q27rtl2_compiled,
            "verilog_lines": exp.verilog_lines,
            "compiler_note": exp.compiler_note,
            "mesh_proof": mesh_rep.to_dict(),
            "board_scaling": board_fits,
        })

    # Plug-and-Learn module proofs
    plug_proofs = []
    for mid, mod in MODULES.items():
        cart, plan, _ = train_organ_seed(mod.target_organ, board_id="ax7203_xc7a200t")
        _, prep = attach_and_adapt_module(cart, mid, clock_mhz=plan.clock_mhz)
        plug_proofs.append(prep.to_dict())

    payload = {
        "project": "Ternara Seed-to-Silicon",
        "version": "0.1.0",
        "license": "Apache-2.0",
        "pack5_reserved_bytes_rejected": f"{rej}/{total_res}",
        "organs": organs_summary,
        "plug_and_learn_proofs": plug_proofs,
    }
    matrix_path = out_dir / "seed_matrix.json"
    matrix_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote verified matrix -> {matrix_path}")
    for o in organs_summary:
        print(
            f"  [{o['organ_id']:8s}] acc={o['val_acc']*100:5.1f}% diff={o['bit_exact_diff']} "
            f"pack5={o['cartridge_bytes']:3d}B q27rtl2={'PASS' if o['q27rtl2_compiled'] else 'SKIP'} "
            f"verilog={o['verilog_lines']:4d}L mesh={o['mesh_proof']['base_global_acc']*100:4.1f}%->"
            f"{o['mesh_proof']['collective_mesh_global_acc']*100:4.1f}% (rej {o['mesh_proof']['deltas_rejected']} poison)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ternara-seed", description="Ternara Seed-to-Silicon Q27 FPGA AI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sc = sub.add_parser("scan", help="List supported FPGA board passports and Hive cluster specs")
    p_sc.add_argument("--hive", type=int, default=1, help="Number of daisy-chained boards in local Ternara Hive")
    p_sc.set_defaults(func=cmd_scan)

    p_gr = sub.add_parser("grow", help="Auto-grow a Q27 Seed for any FPGA board and compile via q27rtl2")
    p_gr.add_argument("--organ", choices=ORGAN_ORDER + ["all"], default="all")
    p_gr.add_argument("--board", default="ax7203_xc7a200t")
    p_gr.add_argument("--hive", type=int, default=1)
    p_gr.add_argument("--out-dir", default="build/grown")
    p_gr.add_argument("--no-compile", action="store_true")
    p_gr.set_defaults(func=cmd_grow)

    p_at = sub.add_parser("attach", help="Simulate hot-plugging a physical module with on-chip BRAM Plug-and-Learn")
    p_at.add_argument("--module", choices=list(MODULES.keys()), default="bent_rotor_esc")
    p_at.add_argument("--board", default="ax7203_xc7a200t")
    p_at.set_defaults(func=cmd_attach)

    p_ms = sub.add_parser("mesh-sim", help="Run verified federated pack5 Ternara Mesh simulation with poison rejection")
    p_ms.add_argument("--organ", choices=ORGAN_ORDER, default="guard")
    p_ms.add_argument("--board", default="ax7203_xc7a200t")
    p_ms.add_argument("--nodes", type=int, default=6)
    p_ms.add_argument("--poison", type=int, default=2)
    p_ms.set_defaults(func=cmd_mesh_sim)

    p_mx = sub.add_parser("matrix", help="Build full 5 organs x 6 boards verified matrix + q27rtl2 compile proofs")
    p_mx.add_argument("--out-dir", default="build/matrix")
    p_mx.add_argument("--no-compile", action="store_true")
    p_mx.set_defaults(func=cmd_matrix)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
