"""Layer 2 of Ternara Seed-to-Silicon: Plug-and-Learn Module Discovery & On-Chip BRAM Adaptation.

When a user plugs in a new physical sensor/motor module (or a drone rotor bends / sensor drifts),
the FPGA does NOT need resynthesis or a PC:
  1. Auto-discovery reads the module's bus/EEPROM descriptor (I2C/SPI/CAN/PMOD-ID) or signal signature.
  2. The frozen 0-DSP Q27 trunk (`W1`) extracts ternary hidden features `h in {-1, 0, +1}^H`.
  3. An integer on-chip plastic trainer (`0 floats`, `0 backprop`, `0 GPU`) updates the BRAM head
     `W2` (`В2_ПЛАСТ` in Q27) via mirrored integer latent accumulators + margin perceptron updates
     in `< 5 ms` on AX7203, recovering full control/detection accuracy directly inside silicon!
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

from .oracle import oracle_forward_batch, oracle_forward_pure
from .organs import generate_organ_dataset
from .pack5 import SeedCartridge


@dataclass(frozen=True)
class HardwareModuleDescriptor:
    module_id: str
    name: str
    bus: str
    eeprom_hex_id: str
    target_organ: str
    physical_scenario: str
    # Physical disturbance / calibration shift simulated when plugging this new module in
    invert_roll_or_polarity: bool = False
    gain_scale: float = 1.0
    offset_bias: float = 0.0
    swap_classes: Tuple[Tuple[int, int], ...] = ()


MODULES: Dict[str, HardwareModuleDescriptor] = {
    "bent_rotor_esc": HardwareModuleDescriptor(
        module_id="bent_rotor_esc",
        name="BLDC ESC + Bent Left Rotor / Swapped Roll Polarity",
        bus="DShot-PWM + SPI-IMU",
        eeprom_hex_id="0x7E01A0",
        target_organ="reflex",
        physical_scenario="Mid-flight damaged rotor / inverted roll wiring; FPGA re-learns stabilization in BRAM",
        gain_scale=0.85,
        offset_bias=10.0,
        swap_classes=((2, 3), (3, 2)),
    ),
    "bosch_bmi270_imu": HardwareModuleDescriptor(
        module_id="bosch_bmi270_imu",
        name="Bosch BMI270 6-Axis IMU (Rotated 180 deg mount)",
        bus="I2C / PMOD-ID",
        eeprom_hex_id="0x680270",
        target_organ="reflex",
        physical_scenario="New IMU mounted upside-down on drone frame; instant on-chip polarity adaptation",
        gain_scale=1.15,
        offset_bias=-8.0,
        swap_classes=((2, 3), (3, 2)),
    ),
    "can_fd_new_ecu": HardwareModuleDescriptor(
        module_id="can_fd_new_ecu",
        name="Aftermarket CAN-FD Vehicle Bus Transceiver",
        bus="CAN-FD 5Mbps",
        eeprom_hex_id="0xCA4FD1",
        target_organ="guard",
        physical_scenario="New vehicle ECU timing profile with higher legitimate burst rate",
        gain_scale=1.10,
        offset_bias=14.0,
    ),
    "piezo_cnc_spindle": HardwareModuleDescriptor(
        module_id="piezo_cnc_spindle",
        name="100kHz Piezo Accelerometer on High-RPM CNC Spindle",
        bus="SPI-ADC",
        eeprom_hex_id="0x5E4701",
        target_organ="sentinel",
        physical_scenario="Mounted on different aluminum spindle housing with shifted resonance floor",
        gain_scale=0.90,
        offset_bias=16.0,
    ),
    "i2s_mems_mic_noisy": HardwareModuleDescriptor(
        module_id="i2s_mems_mic_noisy",
        name="I2S MEMS Microphone in High-Fan Server Room",
        bus="I2S 48kHz",
        eeprom_hex_id="0xA0D109",
        target_organ="audio",
        physical_scenario="New microphone capsule + constant server fan noise floor",
        gain_scale=1.05,
        offset_bias=-12.0,
    ),
}


@dataclass(frozen=True)
class PlugAndLearnReport:
    module_id: str
    module_name: str
    bus: str
    eeprom_hex_id: str
    organ_id: str
    board_id: str
    steps_used: int
    est_onchip_time_ms: float
    acc_before_adapt: float
    acc_after_adapt: float
    accuracy_gain_pct: float
    trits_changed: int
    total_head_trits: int
    bit_exact_diff: int
    adapted_sha256: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def onchip_plastic_adapt(
    base_cart: SeedCartridge,
    X_adapt: np.ndarray,
    y_adapt: np.ndarray,
    epochs: int = 12,
    latent_threshold: int = 4,
    margin: int = 3,
) -> SeedCartridge:
    """Pure-integer on-chip BRAM ternary trainer (0 floats, 0 multiplications, 0 GPU).

    Maintains signed int16 latent accumulators per weight in the plastic head `W2` (matching
    Ternara's `mirrored latent_v2` principle): when a sample violates the integer margin,
    the true class accumulator moves `+h` and the winning wrong class moves `-h`; thresholding
    the latent accumulator at `±latent_threshold` emits updated balanced trits `{-1, 0, +1}`
    directly into the Q27 `В2_ПЛАСТ` BRAM array!
    """
    h_trits, _, _ = oracle_forward_batch(base_cart, X_adapt)
    o, h_dim = base_cart.out_dim, base_cart.hidden_dim

    # Initialize int16 latent mirrors from current ternary head
    latent = base_cart.w2.astype(np.int32) * (latent_threshold + 1)
    b2 = base_cart.b2.astype(np.int32).copy()
    w2 = base_cart.w2.astype(np.int8).copy()

    n = len(y_adapt)
    for _ in range(epochs):
        for i in range(n):
            hi = h_trits[i].astype(np.int32)
            yi = int(y_adapt[i])
            scores = w2.astype(np.int32) @ hi + b2

            # Find strongest competing wrong class
            best_wrong = -1
            best_wrong_score = -(1 << 30)
            for c in range(o):
                if c != yi and int(scores[c]) > best_wrong_score:
                    best_wrong_score = int(scores[c])
                    best_wrong = c

            if int(scores[yi]) < best_wrong_score + margin:
                latent[yi] = np.clip(latent[yi] + hi, -64, 64)
                latent[best_wrong] = np.clip(latent[best_wrong] - hi, -64, 64)
                b2[yi] = min(127, int(b2[yi]) + 1)
                b2[best_wrong] = max(-128, int(b2[best_wrong]) - 1)
                # Re-quantize the two touched BRAM rows to balanced trits {-1, 0, +1}
                w2[yi] = np.where(latent[yi] >= latent_threshold, 1, np.where(latent[yi] <= -latent_threshold, -1, 0)).astype(np.int8)
                w2[best_wrong] = np.where(
                    latent[best_wrong] >= latent_threshold,
                    1,
                    np.where(latent[best_wrong] <= -latent_threshold, -1, 0),
                ).astype(np.int8)

    return SeedCartridge(
        organ=base_cart.organ,
        board_id=base_cart.board_id,
        in_dim=base_cart.in_dim,
        hidden_dim=base_cart.hidden_dim,
        out_dim=base_cart.out_dim,
        w1=base_cart.w1.copy(),
        b1=base_cart.b1.copy(),
        thresh1=base_cart.thresh1,
        w2=w2,
        b2=b2.astype(np.int16),
        metadata={**base_cart.metadata, "onchip_adapted": True},
    )


def attach_and_adapt_module(
    base_cart: SeedCartridge,
    module_id: str,
    clock_mhz: float = 25.0,
    seed: int = 9001,
) -> Tuple[SeedCartridge, PlugAndLearnReport]:
    """Simulate plugging in a physical hardware module and adapting the FPGA's BRAM head on-chip."""
    mod = MODULES[module_id]
    if mod.target_organ != base_cart.organ:
        raise ValueError(
            f"Module {module_id!r} targets organ {mod.target_organ!r}, not {base_cart.organ!r}"
        )

    rng = np.random.default_rng(seed)
    shift = np.full(base_cart.in_dim, mod.offset_bias, dtype=np.float64) + rng.normal(0.0, 4.0, size=base_cart.in_dim)
    gain = np.full(base_cart.in_dim, mod.gain_scale, dtype=np.float64)
    perm = dict(mod.swap_classes) if mod.swap_classes else None

    X_cal, y_cal = generate_organ_dataset(
        base_cart.organ, 160, seed=seed + 1, domain_shift=shift, channel_gain=gain, permute_classes=perm
    )
    X_test, y_test = generate_organ_dataset(
        base_cart.organ, 240, seed=seed + 2, domain_shift=shift, channel_gain=gain, permute_classes=perm
    )

    _, _, pred_before = oracle_forward_batch(base_cart, X_test)
    acc_before = float(np.mean(pred_before == y_test))

    adapted = onchip_plastic_adapt(base_cart, X_cal, y_cal, epochs=10)
    _, _, pred_after = oracle_forward_batch(adapted, X_test)
    acc_after = float(np.mean(pred_after == y_test))

    # Verify bit-exactness of adapted cartridge
    max_diff = 0
    for i in range(20):
        pure = oracle_forward_pure(adapted, X_test[i])
        _, sc_b, dec_b = oracle_forward_batch(adapted, X_test[i : i + 1])
        max_diff = max(max_diff, int(np.max(np.abs(pure.scores - sc_b[0]))), abs(pure.decision - int(dec_b[0])))

    steps_used = len(y_cal) * 10
    # ~24 cycles/update step in Q27 BRAM plastic loop
    est_ms = round((steps_used * 24.0) / (clock_mhz * 1000.0), 3)
    trits_changed = int(np.sum(adapted.w2 != base_cart.w2))

    report = PlugAndLearnReport(
        module_id=mod.module_id,
        module_name=mod.name,
        bus=mod.bus,
        eeprom_hex_id=mod.eeprom_hex_id,
        organ_id=base_cart.organ,
        board_id=base_cart.board_id,
        steps_used=steps_used,
        est_onchip_time_ms=est_ms,
        acc_before_adapt=round(acc_before, 4),
        acc_after_adapt=round(acc_after, 4),
        accuracy_gain_pct=round((acc_after - acc_before) * 100.0, 2),
        trits_changed=trits_changed,
        total_head_trits=int(base_cart.w2.size),
        bit_exact_diff=max_diff,
        adapted_sha256=adapted.sha256(),
    )
    return adapted, report
