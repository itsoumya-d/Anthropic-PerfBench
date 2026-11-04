#!/usr/bin/env python3
"""Generate a deterministic INT4 GEMV parity fixture.

The fixture is intentionally independent of the Rust and JavaScript
implementations: weights, scales, activations and expected outputs are
computed here in Python float64 using the same on-disk packing convention
as benchmarks/kernels.c (low nibble = even k, high nibble = odd k,
one scale per 32-element group).
"""

from __future__ import annotations

import json
import pathlib


class Lcg:
    def __init__(self, seed: int = 0x5EED1234) -> None:
        self.state = seed & 0xFFFFFFFF

    def byte(self) -> int:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) & 0xFF

    def unit(self) -> float:
        return self.byte() / 255.0


def nibble(byte: int, high: bool) -> int:
    q = (byte >> 4) if high else (byte & 0x0F)
    return q - 16 if q >= 8 else q


def main() -> None:
    rows, cols, group = 16, 256, 32
    rng = Lcg()
    weights = bytes(rng.byte() for _ in range(rows * cols // 2))
    groups = cols // group
    scales = [round(0.001 + rng.unit() * 0.01, 10) for _ in range(rows * groups)]
    x = [round(rng.unit() * 2.0 - 1.0, 10) for _ in range(cols)]

    expected = []
    for r in range(rows):
        total = 0.0
        for k in range(cols):
            byte = weights[r * (cols // 2) + (k >> 1)]
            total += nibble(byte, bool(k & 1)) * scales[r * groups + k // group] * x[k]
        expected.append(total)

    fixture = {
        "rows": rows,
        "cols": cols,
        "group": group,
        "weights_hex": weights.hex(),
        "scales": scales,
        "x": x,
        "expected": expected,
    }
    out = pathlib.Path(__file__).parent / "fixtures" / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
