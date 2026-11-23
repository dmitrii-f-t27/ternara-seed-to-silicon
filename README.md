# Ternara Seed-to-Silicon

**Self-fitting balanced ternary (`{-1, 0, +1}`) AI that grows into any FPGA, adapts to newly attached hardware modules on-chip in milliseconds, and shares `< 400 B` `pack5` knowledge deltas across a proof-gated `Ternara Mesh`.**

Built on the **Q27 v2** (`версия 2`) ternary hardware language and open-source FPGA toolchain (`0% Vivado`, `0 DSP`, `0 external DRAM`).

> **Hardware Transparency Notice:** Current targets are commercial **binary FPGAs** (Lattice iCE40, Lattice ECP5, Xilinx Artix-7, AMD UltraScale+) emulating balanced ternary logic via `pack5` (`5 trits/byte`, radix-243, `13/13` hardware fault bytes rejected) and multiplier-free sign-select LUT logic (`0 DSP`). This repository does not claim physical 3-voltage-level ternary silicon.
>
> **Verification Boundary:** All 5 flagship organs in this repo are **trained, bit-exact integer-verified (`0` diff), and compiled by the real `q27rtl2` compiler (`--profile q27-rtl-machine-v2`) into 0-DSP SystemVerilog (`design.sv`, `0` overflow checks, `0` double-writes)**, ready for live FPGA bitstream flashing. Live AX7203 board proofs for Ternara's B-XQ-S / Tetris AI / 4-in-a-Row cores live on [ternara.net](https://www.ternara.net).

---

## Why Seed-to-Silicon?

Conventional AI is compiled for fixed megawatt GPU clusters and shipped as multi-gigabyte float weights. **Ternara Seed-to-Silicon** inverts that stack:

1. **One Compact Seed (`< 430 B` `pack5` cartridge):** A deterministic integer ternary seed with CRC32 + SHA-256 verification.
2. **Auto-Grow to Any Silicon (`iCE40-UP5K` → `ECP5-85F` → `XC7A200T` → `Alveo U50` → `N`-board `Ternara Hive`):** Reads the FPGA passport / JTAG IDCODE and scales parallel Q27 cores + hot-swappable BRAM cartridge slots to fill available LUTs & BRAM with **`0 DSP`** and **`0 external DRAM`**.
3. **Layer-2 Plug-and-Learn (`1.536 ms` @ 25 MHz, `0` resynthesis):** Attach a new sensor, IMU, ESC, or actuator — the on-chip integer plastic head (`В2_ПЛАСТ`) adapts directly inside BRAM via hardware write strobe (`обуч_вкл`) without Vivado, GPU, or bitstream rebuild.
4. **Ternara Mesh (`< 250 B` verified deltas):** Connected FPGA boards share signed `pack5` head deltas (never raw telemetry) gated by deterministic accuracy verification — rejecting `2/2` poisoned nodes while boosting collective accuracy by `+52% to +64%` over the unadapted base.

---

## Verified 5 Flagship FPGA-Advantage Organs (`q27rtl2` Compiled)

Every organ targets workloads where **deterministic sub-microsecond latency (`< 1.2 us`), `0` OS jitter, `0` external DRAM, and `< 1.3 W` power** beat CPU/GPU/NPU stacks:

| Organ | Application | Val Acc | Bit-Exact Diff | `pack5` Cartridge | Real `q27rtl2` SystemVerilog | Ternara Mesh Collective |
|---|---|---:|---:|---:|---|---|
| `guard` | **Ternara-Guard** — Wire-speed CAN-FD / Industrial Ethernet Firewall | `96.0%` | `0` | `389 B` | `1233 L` (`0 DSP`, `0` ovf, `0` dw) | `35.0% → 87.7%` (2/2 poison rej) |
| `reflex` | **Ternara-Reflex** — 1–10 kHz Drone / Actuator Sub-µs Reflex Core | `88.3%` | `0` | `398 B` | `1407 L` (`0 DSP`, `0` ovf, `0` dw) | `27.3% → 87.5%` (2/2 poison rej) |
| `sentinel` | **Ternara-Sentinel** — Industrial CNC / Turbine Vibration & Arc Fault | `96.0%` | `0` | `429 B` | `1406 L` (`0 DSP`, `0` ovf, `0` dw) | `27.3% → 91.2%` (2/2 poison rej) |
| `tick` | **Ternara-Tick** — Sub-µs Order-Book / Risk Circuit Breaker | `95.3%` | `0` | `397 B` | `1243 L` (`0 DSP`, `0` ovf, `0` dw) | `35.0% → 92.3%` (2/2 poison rej) |
| `audio` | **Ternara-Audio** — Always-On Ultra-Low-Power Voice / Acoustic Trigger | `97.7%` | `0` | `400 B` | `1237 L` (`0 DSP`, `0` ovf, `0` dw) | `35.0% → 96.9%` (2/2 poison rej) |

---

## Auto-Grow Silicon Scaling (`guard` organ across 6 boards + 4× Hive)

| Target Board | Family | Hidden Dim | Parallel Cores | Hot BRAM Slots | DSP | DRAM | Latency | Throughput | Est. Power |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iCEBreaker (iCE40-UP5K) | Lattice iCE40 | `8` | `1` | `105` | `0` | `0` | `1.17 us` | `857.1 kFPS` | `0.09 W` |
| Digilent Arty A7-35T | Xilinx Artix-7 | `16` | `5` | `1005` | `0` | `0` | `1.10 us` | `4,545.5 kFPS` | `0.45 W` |
| Ternara Edge / ULX3S (ECP5-85F) | Lattice ECP5 | `16` | `14` | `2085` | `0` | `0` | `0.88 us` | `15,909.1 kFPS` | `0.71 W` |
| Digilent Arty A7-100T | Xilinx Artix-7 | `16` | `18` | `2713` | `0` | `0` | `0.88 us` | `20,454.5 kFPS` | `0.95 W` |
| ALINX AX7203 / Ternara Pro (XC7A200T) | Xilinx Artix-7 | `16` | `32` | `7337` | `0` | `0` | `0.88 us` | `36,363.6 kFPS` | `1.28 W` |
| AMD Alveo U50 (Reference Ceiling) | AMD UltraScale+ | `16` | `32` | `27013` | `0` | `0` | `0.44 us` | `72,727.3 kFPS` | `4.99 W` |
| **Ternara Edge (ECP5-85F) ×4 Hive** | **Lattice ECP5** | `16` | **`61`** | **`8355`** | `0` | `0` | `0.88 us` | **`69,318.2 kFPS`** | `2.96 W` |

---

## Layer-2 Plug-and-Learn On-Chip Adaptation (0 Resynthesis, 0 Floats)

| Attached Hardware Module | Target Organ | Bus | Before → After Acc | Trits Adapted | Est. On-Chip Time @ 25 MHz |
|---|---|---|---:|---:|---:|
| `bent_rotor_esc` (Bent rotor / swapped roll polarity) | `reflex` | `DShot-PWM + SPI-IMU` | `45.0% → 95.4%` (`+50.4%`) | `43/64` | `1.536 ms` |
| `bosch_bmi270_imu` (Rotated 90° IMU axes) | `reflex` | `I2C / PMOD-ID` | `33.8% → 93.8%` (`+60.0%`) | `44/64` | `1.536 ms` |
| `can_fd_new_ecu` (New ECU arbitration ID) | `guard` | `CAN-FD 5Mbps` | `92.5% → 95.4%` (`+2.9%`) | `5/48` | `1.536 ms` |
| `piezo_cnc_spindle` (High-RPM spindle harmonics) | `sentinel` | `SPI-ADC` | `89.6% → 95.0%` (`+5.4%`) | `9/64` | `1.536 ms` |
| `i2s_mems_mic_noisy` (Factory floor noise shift) | `audio` | `I2S 48kHz` | `98.3% → 98.3%` (`+0.0%`) | `14/48` | `1.536 ms` |

---

## Quickstart

```bash
python3 -m pip install -e .
python3 -m ternara_seed.cli scan --board ecp5_85f --hive 4
python3 -m ternara_seed.cli grow --organ guard --board ax7203_xc7a200t --out-dir build/guard
python3 -m ternara_seed.cli attach --module bent_rotor_esc
python3 -m ternara_seed.cli mesh-sim --organ reflex --nodes 6 --poison 2
python3 -m ternara_seed.cli matrix --out-dir build/matrix
pytest -v
```

---

## License

Apache License 2.0 — clean-room open-source implementation by **Dmitrii Fedorov / Ternara** (`ternara.net`).
