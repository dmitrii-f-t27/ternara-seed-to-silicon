"""Lowest-ping Ternara Mesh relay discovery and regional server recommender.

Probes global Anycast and regional cloud/bare-metal relay points concurrently via TCP
handshake RTT (zero privileged ICMP required) and recommends the fastest primary +
fallback endpoints for FPGA boards joining Ternara Mesh.
"""

from __future__ import annotations

import socket
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MeshRelayNode:
    relay_id: str
    name: str
    region: str
    city: str
    provider: str
    tier: str
    mesh_endpoint: str
    probe_host: str
    probe_port: int
    browser_probe_url: str
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


MESH_RELAYS: Tuple[MeshRelayNode, ...] = (
    MeshRelayNode(
        relay_id="anycast-cf",
        name="Cloudflare Global Anycast (Primary)",
        region="Global Anycast",
        city="Nearest of 330+ PoPs",
        provider="Cloudflare Edge",
        tier="anycast_primary",
        mesh_endpoint="https://www.ternara.net/api/mesh",
        probe_host="www.ternara.net",
        probe_port=443,
        browser_probe_url="https://www.ternara.net/api/mesh/v1/health",
        notes="Recommended default: single BGP Anycast address routes every FPGA to its nearest PoP (<25 ms typical).",
    ),
    MeshRelayNode(
        relay_id="sa-gru",
        name="South America · Sao Paulo (GRU)",
        region="South America",
        city="Sao Paulo, BR",
        provider="Fly.io / AWS sa-east-1",
        tier="regional_relay",
        mesh_endpoint="wss://gru.mesh.ternara.net/v1/delta",
        probe_host="gru.fly.dev",
        probe_port=443,
        browser_probe_url="https://ec2.sa-east-1.amazonaws.com/ping",
        notes="Lowest latency direct regional relay for Brazil, Argentina, Uruguay, Chile, and LatAm.",
    ),
    MeshRelayNode(
        relay_id="us-iad",
        name="North America East · Ashburn (IAD)",
        region="North America",
        city="Ashburn, VA, US",
        provider="Hetzner US-East / AWS us-east-1",
        tier="regional_relay",
        mesh_endpoint="wss://iad.mesh.ternara.net/v1/delta",
        probe_host="ash-speed.hetzner.com",
        probe_port=443,
        browser_probe_url="https://ec2.us-east-1.amazonaws.com/ping",
        notes="Primary US East Coast, Canada East, and transatlantic hub.",
    ),
    MeshRelayNode(
        relay_id="us-pdx",
        name="North America West · Oregon (PDX)",
        region="North America",
        city="Boardman / Hillsboro, OR, US",
        provider="AWS us-west-2 / Hetzner US-West",
        tier="regional_relay",
        mesh_endpoint="wss://pdx.mesh.ternara.net/v1/delta",
        probe_host="ec2.us-west-2.amazonaws.com",
        probe_port=443,
        browser_probe_url="https://ec2.us-west-2.amazonaws.com/ping",
        notes="Lowest latency relay for US West Coast and Pacific Northwest.",
    ),
    MeshRelayNode(
        relay_id="eu-fra",
        name="Europe Central · Frankfurt (FRA)",
        region="Europe",
        city="Frankfurt / Falkenstein, DE",
        provider="Hetzner fsn1 / DE-CIX",
        tier="regional_relay",
        mesh_endpoint="wss://fra.mesh.ternara.net/v1/delta",
        probe_host="fsn1-speed.hetzner.com",
        probe_port=443,
        browser_probe_url="https://ec2.eu-central-1.amazonaws.com/ping",
        notes="DE-CIX European core hub for Germany, Central, and Southern Europe.",
    ),
    MeshRelayNode(
        relay_id="eu-lhr",
        name="Europe West · London (LHR)",
        region="Europe",
        city="London, UK",
        provider="AWS eu-west-2 / LINX",
        tier="regional_relay",
        mesh_endpoint="wss://lhr.mesh.ternara.net/v1/delta",
        probe_host="ec2.eu-west-2.amazonaws.com",
        probe_port=443,
        browser_probe_url="https://ec2.eu-west-2.amazonaws.com/ping",
        notes="Low-latency UK, Ireland, and Western Europe relay.",
    ),
    MeshRelayNode(
        relay_id="eu-hel",
        name="Europe North · Helsinki (HEL)",
        region="Europe",
        city="Helsinki, FI",
        provider="Hetzner hel1",
        tier="regional_relay",
        mesh_endpoint="wss://hel.mesh.ternara.net/v1/delta",
        probe_host="hel1-speed.hetzner.com",
        probe_port=443,
        browser_probe_url="https://ec2.eu-north-1.amazonaws.com/ping",
        notes="Nordics, Baltics, and Eastern Europe low-RTT relay.",
    ),
    MeshRelayNode(
        relay_id="ap-nrt",
        name="Asia East · Tokyo (NRT)",
        region="Asia-Pacific",
        city="Tokyo, JP",
        provider="AWS ap-northeast-1 / Equinix TY",
        tier="regional_relay",
        mesh_endpoint="wss://nrt.mesh.ternara.net/v1/delta",
        probe_host="ec2.ap-northeast-1.amazonaws.com",
        probe_port=443,
        browser_probe_url="https://ec2.ap-northeast-1.amazonaws.com/ping",
        notes="Japan, Korea, and Taiwan low-latency relay.",
    ),
    MeshRelayNode(
        relay_id="ap-sin",
        name="Asia Southeast · Singapore (SIN)",
        region="Asia-Pacific",
        city="Singapore, SG",
        provider="Hetzner sin / AWS ap-southeast-1",
        tier="regional_relay",
        mesh_endpoint="wss://sin.mesh.ternara.net/v1/delta",
        probe_host="sin-speed.hetzner.com",
        probe_port=443,
        browser_probe_url="https://ec2.ap-southeast-1.amazonaws.com/ping",
        notes="Southeast Asia, India, and Australia relay.",
    ),
)

RELAY_BY_ID: Dict[str, MeshRelayNode] = {r.relay_id: r for r in MESH_RELAYS}


@dataclass(frozen=True)
class RelayProbeResult:
    relay_id: str
    name: str
    region: str
    city: str
    provider: str
    tier: str
    mesh_endpoint: str
    probe_host: str
    reachable: bool
    samples_ok: int
    samples_total: int
    min_rtt_ms: Optional[float]
    median_rtt_ms: Optional[float]
    jitter_ms: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def tcp_rtt_probe(host: str, port: int, timeout_s: float = 1.8) -> Optional[float]:
    """Measure one TCP connect RTT in milliseconds, or None on timeout/error."""
    t0 = time.perf_counter()
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sock.close()
        return round(dt_ms, 2)
    except OSError:
        return None


def _probe_single_relay(
    relay: MeshRelayNode,
    samples: int,
    timeout_s: float,
    probe_fn: Callable[[str, int, float], Optional[float]],
) -> RelayProbeResult:
    if probe_fn is tcp_rtt_probe:
        try:
            socket.getaddrinfo(relay.probe_host, relay.probe_port, type=socket.SOCK_STREAM)
        except OSError:
            pass
    rtts: List[float] = []
    for _ in range(max(1, samples)):
        val = probe_fn(relay.probe_host, relay.probe_port, timeout_s)
        if val is not None:
            rtts.append(val)
    if not rtts:
        return RelayProbeResult(
            relay_id=relay.relay_id,
            name=relay.name,
            region=relay.region,
            city=relay.city,
            provider=relay.provider,
            tier=relay.tier,
            mesh_endpoint=relay.mesh_endpoint,
            probe_host=relay.probe_host,
            reachable=False,
            samples_ok=0,
            samples_total=max(1, samples),
            min_rtt_ms=None,
            median_rtt_ms=None,
            jitter_ms=None,
        )
    mn = round(min(rtts), 2)
    med = round(float(statistics.median(rtts)), 2)
    jit = round(max(rtts) - min(rtts), 2) if len(rtts) > 1 else 0.0
    return RelayProbeResult(
        relay_id=relay.relay_id,
        name=relay.name,
        region=relay.region,
        city=relay.city,
        provider=relay.provider,
        tier=relay.tier,
        mesh_endpoint=relay.mesh_endpoint,
        probe_host=relay.probe_host,
        reachable=True,
        samples_ok=len(rtts),
        samples_total=max(1, samples),
        min_rtt_ms=mn,
        median_rtt_ms=med,
        jitter_ms=jit,
    )


def probe_mesh_relays(
    relays: Sequence[MeshRelayNode] = MESH_RELAYS,
    samples: int = 3,
    timeout_s: float = 1.8,
    probe_fn: Callable[[str, int, float], Optional[float]] = tcp_rtt_probe,
) -> List[RelayProbeResult]:
    """Concurrently probe all candidate Ternara Mesh relays and sort by lowest RTT."""
    results: List[RelayProbeResult] = []
    max_workers = max(1, min(len(relays), 12))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_probe_single_relay, r, samples, timeout_s, probe_fn) for r in relays]
        for fut in as_completed(futs):
            results.append(fut.result())

    def _sort_key(r: RelayProbeResult) -> Tuple[int, float, float, str]:
        if not r.reachable or r.min_rtt_ms is None or r.median_rtt_ms is None:
            return (1, 1e9, 1e9, r.relay_id)
        return (0, r.min_rtt_ms, r.median_rtt_ms, r.relay_id)

    results.sort(key=_sort_key)
    return results


def recommend_best_relays(results: Sequence[RelayProbeResult]) -> Dict[str, object]:
    """Select the lowest-ping primary and backup Ternara Mesh relays with CLI connect hint."""
    reachable = [r for r in results if r.reachable and r.min_rtt_ms is not None]
    if not reachable:
        fallback = MESH_RELAYS[0]
        return {
            "status": "offline_fallback",
            "primary_relay": fallback.relay_id,
            "primary_endpoint": fallback.mesh_endpoint,
            "primary_min_rtt_ms": None,
            "backup_relay": None,
            "backup_endpoint": None,
            "connect_hint": f"python3 -m ternara_seed.cli mesh-sim --organ guard  # default endpoint: {fallback.mesh_endpoint}",
            "results": [r.to_dict() for r in results],
        }

    primary = reachable[0]
    backup = reachable[1] if len(reachable) > 1 else None
    regional_best = next((r for r in reachable if r.tier == "regional_relay"), primary)

    return {
        "status": "ok",
        "primary_relay": primary.relay_id,
        "primary_name": primary.name,
        "primary_endpoint": primary.mesh_endpoint,
        "primary_min_rtt_ms": primary.min_rtt_ms,
        "primary_median_rtt_ms": primary.median_rtt_ms,
        "backup_relay": backup.relay_id if backup else None,
        "backup_endpoint": backup.mesh_endpoint if backup else None,
        "backup_min_rtt_ms": backup.min_rtt_ms if backup else None,
        "fastest_regional_relay": regional_best.relay_id,
        "fastest_regional_endpoint": regional_best.mesh_endpoint,
        "fastest_regional_min_rtt_ms": regional_best.min_rtt_ms,
        "connect_hint": f"ternara-seed connect --relay {primary.relay_id} --endpoint {primary.mesh_endpoint}",
        "results": [r.to_dict() for r in results],
    }
