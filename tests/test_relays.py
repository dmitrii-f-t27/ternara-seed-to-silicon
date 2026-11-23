"""Tests for Ternara Mesh lowest-ping relay discovery and recommendation logic."""

from __future__ import annotations

from ternara_seed.relays import (
    MESH_RELAYS,
    RELAY_BY_ID,
    probe_mesh_relays,
    recommend_best_relays,
)


def test_relay_catalog_covers_global_regions_and_anycast() -> None:
    assert len(MESH_RELAYS) >= 9
    assert "anycast-cf" in RELAY_BY_ID
    regions = {r.region for r in MESH_RELAYS}
    assert {"Global Anycast", "South America", "North America", "Europe", "Asia-Pacific"}.issubset(regions)
    for r in MESH_RELAYS:
        assert r.mesh_endpoint.startswith(("https://", "wss://"))
        assert r.browser_probe_url.startswith("https://")
        assert r.probe_port == 443


def test_probe_ranks_lowest_rtt_and_picks_primary_backup_and_regional() -> None:
    fake_rtts = {
        "www.ternara.net": 11.2,
        "gru.fly.dev": 16.8,
        "ash-speed.hetzner.com": 131.0,
        "ec2.us-west-2.amazonaws.com": 205.0,
        "fsn1-speed.hetzner.com": 220.0,
        "ec2.eu-west-2.amazonaws.com": 244.0,
        "hel1-speed.hetzner.com": 278.0,
        "ec2.ap-northeast-1.amazonaws.com": 304.0,
        "sin-speed.hetzner.com": None,  # simulated unreachable node
    }

    def fake_probe(host: str, port: int, timeout_s: float):
        return fake_rtts.get(host)

    results = probe_mesh_relays(samples=2, probe_fn=fake_probe)
    assert len(results) == len(MESH_RELAYS)
    assert results[0].relay_id == "anycast-cf"
    assert results[1].relay_id == "sa-gru"
    assert results[-1].relay_id == "ap-sin"
    assert results[-1].reachable is False

    rec = recommend_best_relays(results)
    assert rec["status"] == "ok"
    assert rec["primary_relay"] == "anycast-cf"
    assert rec["backup_relay"] == "sa-gru"
    assert rec["fastest_regional_relay"] == "sa-gru"
    assert "--relay anycast-cf" in rec["connect_hint"]


def test_offline_fallback_when_all_probes_fail() -> None:
    results = probe_mesh_relays(samples=1, probe_fn=lambda h, p, t: None)
    rec = recommend_best_relays(results)
    assert rec["status"] == "offline_fallback"
    assert rec["primary_relay"] == "anycast-cf"
    assert rec["primary_endpoint"] == "https://www.ternara.net/api/mesh"
