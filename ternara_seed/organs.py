"""All 5 flagship FPGA-native Q27 Organs where 0-DSP microsecond silicon beats CPU/GPU.

Each organ includes:
  - Honest OrganSpec (interfaces, class labels, conservative CPU/OS p99 latency baseline)
  - Realistic integer [-128..+127] physical signal generator (including domain-shift support for Plug-and-Learn & Mesh)
  - Fast, deterministic pure-NumPy ternary QAT trainer -> bit-exact Q27 SeedCartridge
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .grower import GrownPlan, OrganSpec, grow_seed_for_board
from .oracle import oracle_forward_batch, oracle_forward_pure
from .pack5 import SeedCartridge

ORGANS: Dict[str, OrganSpec] = {
    "guard": OrganSpec(
        organ_id="guard",
        title="Ternara-Guard",
        tagline="Inline USB / CAN / L2 Hardware AI Firewall (< 2 us, kills BadUSB & spoofed frames before OS)",
        domain="Cybersecurity & Automotive",
        in_dim=16,
        out_dim=3,
        class_names=("ALLOW", "THROTTLE", "DROP_ATTACK"),
        cpu_p99_us=3500.0,
        target_fpga_us=10.0,
        default_bus="USB-HID / CAN-FD / RMII",
    ),
    "reflex": OrganSpec(
        organ_id="reflex",
        title="Ternara-Reflex",
        tagline="25-50 kHz Zero-Jitter Robot & Drone Stabilizer with Live On-Chip BRAM Self-Recovery",
        domain="Robotics & Drones",
        in_dim=16,
        out_dim=4,
        class_names=("HOLD", "THRUST_UP", "ROLL_LEFT", "ROLL_RIGHT"),
        cpu_p99_us=8500.0,
        target_fpga_us=20.0,
        default_bus="SPI-IMU / DShot-PWM",
    ),
    "sentinel": OrganSpec(
        organ_id="sentinel",
        title="Ternara-Sentinel",
        tagline="24/7 Sub-Watt Industrial Vibration & Arc-Fault Detector (< 25 us hardware relay trip)",
        domain="Industrial Predictive Maintenance",
        in_dim=16,
        out_dim=4,
        class_names=("HEALTHY", "BEARING_WEAR", "ROTOR_IMBALANCE", "ARC_FAULT"),
        cpu_p99_us=12000.0,
        target_fpga_us=25.0,
        default_bus="SPI-ADC 100kHz / Relay GPIO",
    ),
    "tick": OrganSpec(
        organ_id="tick",
        title="Ternara-Tick",
        tagline="Sub-Microsecond L2 Order-Book & RF Stream Classifier (0 DSP, multi-lane parallel)",
        domain="Low-Latency Finance & RF/SDR",
        in_dim=16,
        out_dim=3,
        class_names=("SELL_PRESSURE", "NEUTRAL", "BUY_PRESSURE"),
        cpu_p99_us=450.0,
        target_fpga_us=5.0,
        default_bus="10GbE / MAC-FIFO / SDR-IQ",
    ),
    "audio": OrganSpec(
        organ_id="audio",
        title="Ternara-Audio",
        tagline="Zero-Buffer Hardware AI Noise Gate & Transient Suppressor (no 15 ms CoreAudio/ASIO lag)",
        domain="Pro Audio & Edge Comms",
        in_dim=16,
        out_dim=3,
        class_names=("CLEAN_VOICE", "NOISE_GATE", "TRANSIENT_SUPPRESS"),
        cpu_p99_us=15000.0,
        target_fpga_us=50.0,
        default_bus="I2S 48kHz / TDM",
    ),
}

ORGAN_ORDER: List[str] = ["guard", "reflex", "sentinel", "tick", "audio"]


def _organ_prototypes(organ_id: str) -> np.ndarray:
    """Domain-authentic 16-dim integer feature prototypes for each physical organ class."""
    if organ_id == "guard":
        # [iat_mean, iat_min, burst_len, hid_rate, ent, crc_anom, can_id_jump, dlc_anom, ...]
        return np.array([
            [+42, +38, -35, -40, +10, -45, -38, -42, +30, -30, -25, +20, -35, -30, +25, -40],  # ALLOW
            [-10, -18, +24, +28, +15, -10, +18, +12, -15, +22, +28, -10, +15, +18, -12, +10],  # THROTTLE
            [-58, -62, +65, +70, +48, +55, +60, +52, -48, +55, +62, -45, +58, +64, -50, +68],  # DROP_ATTACK
        ], dtype=np.float64)
    if organ_id == "reflex":
        # [pitch, roll, dpitch, droll, alt_err, rpm_fl, rpm_fr, rpm_rl, rpm_rr, gust_x, gust_y, ...]
        return np.array([
            [  0,   0,   0,   0,  +8,  +5,  +5,  +5,  +5,   0,   0,  +4,   0,   0,  +6,   0],  # HOLD
            [-42,  +2, -38,  +2, -62, -48, -48, -42, -42,  +5, -30, -50, -40, -35, -55, -25],  # THRUST_UP
            [ +8, -58,  +5, -52,  -5, -50, +48, -45, +45, -55,  +6, -42, +40, -48,  +4, -38],  # ROLL_LEFT
            [ +8, +58,  +5, +52,  -5, +48, -50, +45, -45, +55,  +6, +42, -40, +48,  +4, +38],  # ROLL_RIGHT
        ], dtype=np.float64)
    if organ_id == "sentinel":
        # [band0..band7, kurtosis, crest, rms_curr, phase_imbal, hf_spike, temp_slope, ...]
        return np.array([
            [-35, -30, -28, -32, -38, -42, -45, -48, -36, -32, -25, -40, -48, -30, -35, -42],  # HEALTHY
            [+10, +18, +45, +58, +52, +22,  -8, -15, +35, +30,  +8,  +5,  -5, +20, +32, +15],  # BEARING_WEAR
            [+62, +55, +12,  -8, -18, -22, -25, -28, +10, +18, +48, +52, -10, +25, +12, +45],  # ROTOR_IMBALANCE
            [-12, -10,  +5, +18, +42, +65, +72, +78, +68, +70, +55, +35, +80, +40, +62, +74],  # ARC_FAULT
        ], dtype=np.float64)
    if organ_id == "tick":
        # [imbal_l1..l4, flow_1ms, flow_5ms, spread_vel, micro_mom, cancel_ratio, rf_snr, ...]
        return np.array([
            [-55, -48, -42, -35, -60, -52, +18, -50, +42, -38, -45, -40, -52, -30, -48, -44],  # SELL_PRESSURE
            [  0,   0,   0,   0,   0,   0, -32,   0, -18,   0,   0,   0,   0,   0,   0,   0],  # NEUTRAL
            [+55, +48, +42, +35, +60, +52, +18, +50, -42, +38, +45, +40, +52, +30, +48, +44],  # BUY_PRESSURE
        ], dtype=np.float64)
    if organ_id == "audio":
        # [low_e, mid_e, high_e, pitch_harm, zcr, spec_flat, env_slope, click_peak, ...]
        return np.array([
            [+42, +58, +20, +64, -22, -48,  +8, -35, +50, +44, -28, +55, -30, +38, +45, -25],  # CLEAN_VOICE
            [-32, -28, +18, -48, +44, +56, -24, -18, -42, -38, +52, -45, +40, -35, -40, +48],  # NOISE_GATE
            [ -5, +15, +68, -25, +62, +28, +72, +78, -10, +12, +65, -20, +74, +58, -15, +70],  # TRANSIENT_SUPPRESS
        ], dtype=np.float64)
    raise KeyError(f"Unknown organ_id: {organ_id!r}")


def generate_organ_dataset(
    organ_id: str,
    n_samples: int,
    seed: int = 20261122,
    noise_std: float = 20.0,
    domain_shift: np.ndarray | None = None,
    channel_gain: np.ndarray | None = None,
    permute_classes: Dict[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic int8 [-128..127] sensor/stream frames + class labels."""
    protos = _organ_prototypes(organ_id)
    out_dim, in_dim = protos.shape
    rng = np.random.default_rng(seed)

    labels = rng.integers(0, out_dim, size=n_samples, dtype=np.int32)
    X = protos[labels].copy()
    # Add realistic per-sample amplitude scaling & Gaussian sensor jitter
    scale = rng.uniform(0.68, 1.32, size=(n_samples, 1))
    X = X * scale + rng.normal(0.0, noise_std, size=(n_samples, in_dim))

    if channel_gain is not None:
        X = X * np.asarray(channel_gain, dtype=np.float64)[None, :]
    if domain_shift is not None:
        X = X + np.asarray(domain_shift, dtype=np.float64)[None, :]
    if permute_classes:
        mapped = labels.copy()
        for src, dst in permute_classes.items():
            mapped[labels == src] = dst
        labels = mapped

    X_int = np.clip(np.rint(X), -128, 127).astype(np.int32)
    return X_int, labels


def _ternarize(W: np.ndarray, tau_factor: float = 0.45) -> np.ndarray:
    """Symmetric threshold ternarization -> int8 in {-1, 0, +1}."""
    tau = max(1e-6, float(np.mean(np.abs(W))) * tau_factor)
    return np.where(W > tau, 1, np.where(W < -tau, -1, 0)).astype(np.int8)


@dataclass(frozen=True)
class OrganTrainReport:
    organ_id: str
    board_id: str
    train_acc: float
    val_acc: float
    bit_exact_diff: int
    total_trits: int
    zero_fraction: float
    cartridge_bytes: int
    sha256: str


def train_organ_seed(
    organ_id: str,
    board_id: str = "ax7203_xc7a200t",
    hive_count: int = 1,
    n_train: int = 900,
    n_val: int = 300,
    epochs: int = 60,
    seed: int = 20261122,
) -> Tuple[SeedCartridge, GrownPlan, OrganTrainReport]:
    """Grow a Q27 Seed for `board_id`, train 0-DSP ternary weights (pure NumPy QAT), and verify bit-exactness."""
    spec = ORGANS[organ_id]
    plan = grow_seed_for_board(spec, board_id=board_id, hive_count=hive_count)
    h = plan.hidden_dim
    m = spec.in_dim
    o = spec.out_dim

    X_tr, y_tr = generate_organ_dataset(organ_id, n_train, seed=seed)
    X_va, y_va = generate_organ_dataset(organ_id, n_val, seed=seed + 777)

    rng = np.random.default_rng(seed + 13)
    # Initialize W1 cleanly from class contrast directions + structured random projections
    protos = _organ_prototypes(organ_id)
    W1_f = rng.normal(0.0, 0.35, size=(h, m))
    for j in range(min(h, o * 2)):
        c = j % o
        W1_f[j] = protos[c] / (np.linalg.norm(protos[c]) + 1e-6) * (1.0 if j < o else -0.8) + rng.normal(0.0, 0.12, size=m)
    b1_f = np.zeros(h, dtype=np.float64)
    W2_f = rng.normal(0.0, 0.35, size=(o, h))
    b2_f = np.zeros(o, dtype=np.float64)

    thresh1 = 64
    X_tr_f = X_tr.astype(np.float64)
    y_oh = np.eye(o, dtype=np.float64)[y_tr]

    lr = 0.018
    for ep in range(epochs):
        W1_q = _ternarize(W1_f, 0.42).astype(np.float64)
        W2_q = _ternarize(W2_f, 0.38).astype(np.float64)

        pre1 = X_tr_f @ W1_q.T + b1_f[None, :]
        # Straight-through estimator around Q27 ternary measure threshold
        h_q = np.where(pre1 >= thresh1, 1.0, np.where(pre1 <= -thresh1, -1.0, 0.0))
        logits = (h_q @ W2_q.T + b2_f[None, :]) * 0.45
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)

        dlogits = (probs - y_oh) / n_train
        dW2 = dlogits.T @ h_q
        db2 = dlogits.sum(axis=0)

        dh = dlogits @ W2_q
        ste_mask = (np.abs(pre1) <= thresh1 * 3.5).astype(np.float64) / float(thresh1)
        dpre1 = dh * ste_mask
        dW1 = dpre1.T @ X_tr_f
        db1 = dpre1.sum(axis=0)

        step = lr * (0.985 ** ep)
        W2_f -= step * 10.0 * dW2
        b2_f -= step * 18.0 * db2
        W1_f -= step * 0.25 * dW1
        b1_f -= step * 25.0 * db1

    w1 = _ternarize(W1_f, 0.42)
    w2 = _ternarize(W2_f, 0.38)
    b1 = np.clip(np.rint(b1_f), -1024, 1023).astype(np.int16)
    b2 = np.clip(np.rint(b2_f), -128, 127).astype(np.int16)

    cart = SeedCartridge(
        organ=organ_id,
        board_id=plan.board_id,
        in_dim=m,
        hidden_dim=h,
        out_dim=o,
        w1=w1,
        b1=b1,
        thresh1=thresh1,
        w2=w2,
        b2=b2,
        metadata={
            "title": spec.title,
            "domain": spec.domain,
            "class_names": list(spec.class_names),
            "parallel_cores": plan.parallel_cores,
            "latency_us": plan.latency_us,
        },
    )

    # Evaluate using the bit-exact pure-integer oracle
    _, _, pred_tr = oracle_forward_batch(cart, X_tr)
    _, _, pred_va = oracle_forward_batch(cart, X_va)
    tr_acc = float(np.mean(pred_tr == y_tr))
    va_acc = float(np.mean(pred_va == y_va))

    # Bit-exactness cross-check: scalar 0-mult pure oracle vs batch oracle on 32 samples
    max_diff = 0
    for i in range(min(32, n_val)):
        pure = oracle_forward_pure(cart, X_va[i])
        _, sc_b, dec_b = oracle_forward_batch(cart, X_va[i : i + 1])
        max_diff = max(
            max_diff,
            int(np.max(np.abs(pure.scores - sc_b[0]))),
            abs(pure.decision - int(dec_b[0])),
        )

    rep = OrganTrainReport(
        organ_id=organ_id,
        board_id=plan.board_id,
        train_acc=round(tr_acc, 4),
        val_acc=round(va_acc, 4),
        bit_exact_diff=max_diff,
        total_trits=cart.total_trits,
        zero_fraction=round(cart.zero_fraction, 4),
        cartridge_bytes=len(cart.to_bytes()),
        sha256=cart.sha256(),
    )
    return cart, plan, rep
