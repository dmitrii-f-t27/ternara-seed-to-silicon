"""Bit-exact pure-integer Q27 Seed oracle (0 floats, 0 multiplications in ternary layers).

Matches the generated `Q27 версия 2` RTL semantics identically:
  - Layer 1: h[j] = измерить(Σ(x ⊗ W1[j]) смещение b1[j] порог thresh1) in {-1, 0, +1}
  - Layer 2 (plastic BRAM layer): score[c] = Σ(h ⊗ W2[c]) + b2[c]
  - Decision: argmax(score) with deterministic lowest-index tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pack5 import SeedCartridge


def q27_measure(val: int, bias: int, thresh: int) -> int:
    """Exact semantics of Q27 `измерить(val смещение bias порог thresh)` -> trit in {-1, 0, +1}."""
    shifted = int(val) + int(bias)
    t = abs(int(thresh))
    if shifted >= t:
        return 1
    if shifted <= -t:
        return -1
    return 0


def ternary_dot_pure(x_int: Sequence[int], w_trits: Sequence[int]) -> int:
    """Pure addition/subtraction ternary dot product (0 multiplications, 0 floats)."""
    acc = 0
    for xv, wv in zip(x_int, w_trits):
        if wv == 1:
            acc += int(xv)
        elif wv == -1:
            acc -= int(xv)
    return acc


@dataclass(frozen=True)
class OracleOutput:
    hidden_trits: np.ndarray  # (hidden_dim,) int8 in {-1, 0, +1}
    scores: np.ndarray        # (out_dim,) int32
    decision: int             # argmax class index


def oracle_forward_pure(cart: SeedCartridge, x_in: Sequence[int] | np.ndarray) -> OracleOutput:
    """Scalar reference oracle with zero floats and zero multiplications."""
    xs = [int(v) for v in x_in]
    if len(xs) != cart.in_dim:
        raise ValueError(f"Expected input length {cart.in_dim}, got {len(xs)}")

    h = np.empty(cart.hidden_dim, dtype=np.int8)
    for j in range(cart.hidden_dim):
        acc = ternary_dot_pure(xs, cart.w1[j])
        h[j] = q27_measure(acc, int(cart.b1[j]), cart.thresh1)

    scores = np.empty(cart.out_dim, dtype=np.int32)
    best_c = 0
    best_v = -(1 << 30)
    for c in range(cart.out_dim):
        s = ternary_dot_pure(h, cart.w2[c]) + int(cart.b2[c])
        scores[c] = s
        if s > best_v:
            best_v = s
            best_c = c

    return OracleOutput(hidden_trits=h, scores=scores, decision=best_c)


def oracle_forward_batch(cart: SeedCartridge, x_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-exact vectorized integer batch oracle (matches oracle_forward_pure with 0 diff)."""
    X = np.asarray(x_batch, dtype=np.int32)
    if X.ndim != 2 or X.shape[1] != cart.in_dim:
        raise ValueError(f"Expected X shape (N, {cart.in_dim}), got {X.shape}")
    acc1 = X @ cart.w1.astype(np.int32).T + cart.b1.astype(np.int32)[None, :]
    t = abs(int(cart.thresh1))
    h = np.where(acc1 >= t, 1, np.where(acc1 <= -t, -1, 0)).astype(np.int8)
    scores = h.astype(np.int32) @ cart.w2.astype(np.int32).T + cart.b2.astype(np.int32)[None, :]
    decisions = np.argmax(scores, axis=1).astype(np.int32)
    return h, scores, decisions
