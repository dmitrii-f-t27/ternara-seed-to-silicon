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
from .mesh_client import connect_node_to_mesh
from .mesh_server import create_mesh_server
from .relays import MESH_RELAYS, probe_mesh_relays, recommend_best_relays
from .rtl_sim import simulate_compiled_q27_rtl


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
        "mesh_relays": [r.to_dict() for r in MESH_RELAYS],
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



def cmd_best_server(args: argparse.Namespace) -> int:
    results = probe_mesh_relays(samples=args.samples, timeout_s=args.timeout)
    rec = recommend_best_relays(results)
    if args.json:
        print(json.dumps(rec, indent=2))
        return 0
    print("=== Ternara Mesh — Lowest-Ping Relay Discovery ===")
    for i, r in enumerate(results, start=1):
        if r.reachable and r.min_rtt_ms is not None and r.median_rtt_ms is not None:
            tag = "  <-- RECOMMENDED" if i == 1 else ""
            print(
                f"  #{i} {r.min_rtt_ms:6.1f} ms (med {r.median_rtt_ms:6.1f} ms, jit {r.jitter_ms:5.1f} ms) | "
                f"{r.relay_id:10s} | {r.city:28s} | {r.mesh_endpoint}{tag}"
            )
        else:
            print(f"  #{i}   UNREACHABLE                         | {r.relay_id:10s} | {r.city:28s} | {r.mesh_endpoint}")
    print(f"\nRecommended primary : {rec['primary_relay']} -> {rec['primary_endpoint']} ({rec['primary_min_rtt_ms']} ms)")
    if rec.get("backup_relay"):
        print(f"Recommended backup  : {rec['backup_relay']} -> {rec['backup_endpoint']} ({rec['backup_min_rtt_ms']} ms)")
    if rec.get("fastest_regional_relay") and rec["fastest_regional_relay"] != rec["primary_relay"]:
        print(
            f"Fastest direct region: {rec['fastest_regional_relay']} -> "
            f"{rec['fastest_regional_endpoint']} ({rec['fastest_regional_min_rtt_ms']} ms)"
        )
    print(f"Connect command     : {rec['connect_hint']}")
    return 0



def cmd_verify_rtl(args: argparse.Namespace) -> int:
    from .organs import generate_organ_dataset
    targets = list(ORGAN_ORDER) if args.organ == "all" else [args.organ]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for oid in targets:
        cart, plan, _ = train_organ_seed(oid, board_id=args.board)
        exp = export_and_compile_q27(cart, plan, out_root / oid, run_compiler=True)
        if not exp.q27rtl2_compiled or not exp.verilog_path:
            raise RuntimeError(f"q27rtl2 compile failed for {oid}: {exp.compiler_note}")
        X_val, y_val = generate_organ_dataset(oid, n_samples=args.samples, seed=9999)
        sim = simulate_compiled_q27_rtl(Path(exp.verilog_path).parent, X_val, y_val, cart)
        results.append({
            "organ_id": oid,
            "verilog_lines": exp.verilog_lines,
            **sim.to_dict(),
        })
        print(
            f"  [{oid:8s}] iverilog/vvp RTL acc={sim.rtl_accuracy*100:5.1f}% "
            f"oracle={sim.oracle_accuracy*100:5.1f}% diff={sim.bit_exact_mismatches} "
            f"hw_err={sim.hardware_errors} lat={sim.observed_latency_ticks}ticks"
        )
    if args.json:
        print(json.dumps({"rtl_verification": results}, indent=2))
    return 0


def cmd_mesh_server(args: argparse.Namespace) -> int:
    server, coord = create_mesh_server(host=args.host, port=args.port, state_dir=args.state_dir)
    print(f"Ternara Mesh Coordinator listening on http://{args.host}:{args.port} (state: {args.state_dir})")
    print(f"Loaded {len(coord.states)} organs with deterministic Proof Gate + pack5 CRC32 verification.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Ternara Mesh Coordinator cleanly.")
    finally:
        server.server_close()
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    url = args.url
    if not url:
        rec = recommend_best_relays(probe_mesh_relays(samples=2))
        print(f"Auto-selected lowest-ping relay: {rec['primary_relay']} ({rec['primary_min_rtt_ms']} ms) -> {rec['primary_endpoint']}")
        print("Pass --url http://127.0.0.1:8795 to connect to a running local/remote Mesh coordinator.")
        return 0
    cart, rep = connect_node_to_mesh(
        server_url=url,
        organ_id=args.organ,
        node_id=args.node_id,
        out_dir=args.out_dir,
        board_id=args.board,
        detect_jtag=args.detect_jtag,
        poison=args.poison,
        env_seed=args.env_seed,
    )
    print(json.dumps(rep.to_dict(), indent=2))
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

    p_bs = sub.add_parser("best-server", help="Probe global Ternara Mesh relays and recommend lowest-ping server")
    p_bs.add_argument("--samples", type=int, default=3, help="TCP RTT samples per relay")
    p_bs.add_argument("--timeout", type=float, default=1.8, help="Per-probe TCP timeout in seconds")
    p_bs.add_argument("--json", action="store_true", help="Output structured JSON report")
    p_bs.set_defaults(func=cmd_best_server)

    p_vr = sub.add_parser("verify-rtl", help="Compile Q27 v2 via q27rtl2 and simulate compiled design.sv in iverilog/vvp")
    p_vr.add_argument("--organ", choices=ORGAN_ORDER + ["all"], default="all")
    p_vr.add_argument("--board", default="ax7203_xc7a200t")
    p_vr.add_argument("--samples", type=int, default=120)
    p_vr.add_argument("--out-dir", default="build/rtl_verify")
    p_vr.add_argument("--json", action="store_true")
    p_vr.set_defaults(func=cmd_verify_rtl)

    p_srv = sub.add_parser("mesh-server", help="Run a live Ternara Mesh HTTP/pack5 Coordinator server with Proof Gate")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8795)
    p_srv.add_argument("--state-dir", default="build/mesh_state")
    p_srv.set_defaults(func=cmd_mesh_server)

    p_con = sub.add_parser("connect", help="Connect an FPGA/host node to a live Ternara Mesh server and sync pack5 deltas")
    p_con.add_argument("--url", default="", help="Live Mesh coordinator base URL (e.g. http://127.0.0.1:8795)")
    p_con.add_argument("--organ", choices=ORGAN_ORDER, default="guard")
    p_con.add_argument("--node-id", default="fpga-node-01")
    p_con.add_argument("--board", default="ax7203_xc7a200t")
    p_con.add_argument("--out-dir", default="build/mesh_node")
    p_con.add_argument("--detect-jtag", action="store_true", help="Opt-in USB JTAG IDCODE board detection")
    p_con.add_argument("--poison", action="store_true", help="Send adversarial poisoned delta to test Proof Gate rejection")
    p_con.add_argument("--env-seed", type=int, default=101)
    p_con.set_defaults(func=cmd_connect)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
