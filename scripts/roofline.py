#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compute a memory-roofline table and SVG from results/bench.json."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "bench.json"
CSV_OUT = ROOT / "results" / "roofline.csv"
SVG_OUT = ROOT / "results" / "roofline.svg"


def main() -> int:
    data = json.loads(DATA.read_text())
    bandwidth = data["copy_bandwidth_gbps"]
    records = data["records"]
    for record in records:
        record["arithmetic_intensity"] = record["flops"] / record["bytes"]
        record["bandwidth_ceiling_gflops"] = bandwidth * record["arithmetic_intensity"]
        record["roofline_utilization_pct"] = (
            100.0 * record["gflops"] / record["bandwidth_ceiling_gflops"]
            if record["bandwidth_ceiling_gflops"]
            else 0.0
        )

    by_kernel = {}
    for record in records:
        by_kernel.setdefault(record["kernel"], {})[record["variant"]] = record["ms"]
    for kernel, variants in by_kernel.items():
        scalar = variants.get("scalar", 1.0)
        for record in records:
            if record["kernel"] == kernel:
                record["speedup_vs_scalar"] = scalar / record["ms"]

    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kernel", "variant", "ms", "gflops", "gbps",
                "arithmetic_intensity", "bandwidth_ceiling_gflops",
                "roofline_utilization_pct", "speedup_vs_scalar",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})

    print("| Kernel | Variant | ms | GFLOP/s | GB/s | AI | Roof GFLOP/s | Util | Speedup |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for record in records:
        print(
            f"| {record['kernel']} | {record['variant']} | {record['ms']:.3f} | "
            f"{record['gflops']:.2f} | {record['gbps']:.2f} | "
            f"{record['arithmetic_intensity']:.2f} | "
            f"{record['bandwidth_ceiling_gflops']:.2f} | "
            f"{record['roofline_utilization_pct']:.1f}% | "
            f"{record['speedup_vs_scalar']:.2f}x |"
        )
    print(f"\nMeasured copy bandwidth: {bandwidth:.2f} GB/s")

    width, height = 960, 560
    left, top, right, bottom = 70, 40, 30, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -1.0, 6.0

    def sx(ai):
        return left + (math.log2(ai) - x_min) / (x_max - x_min) * plot_w

    def sy(gflops):
        return top + plot_h - (math.log2(max(gflops, 1e-6)) - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace, monospace">',
        f'<rect width="{width}" height="{height}" fill="#0b1020"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#11182b" stroke="#334155"/>',
    ]
    for power in range(int(x_min), int(x_max) + 1):
        x = left + (power - x_min) / (x_max - x_min) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#1e293b"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+18}" fill="#94a3b8" font-size="11" text-anchor="middle">2^{power}</text>')
    for power in range(int(y_min), int(y_max) + 1):
        y = top + plot_h - (power - y_min) / (y_max - y_min) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#1e293b"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">2^{power}</text>')

    x0, x1 = 2 ** x_min, 2 ** x_max
    parts.append(
        f'<line x1="{sx(x0):.1f}" y1="{sy(bandwidth*x0):.1f}" '
        f'x2="{sx(x1):.1f}" y2="{sy(bandwidth*x1):.1f}" stroke="#f59e0b" stroke-width="2"/>'
    )
    colors = {"scalar": "#ef4444", "unrolled": "#38bdf8", "neon": "#22c55e"}
    for record in records:
        x = sx(record["arithmetic_intensity"])
        y = sy(record["gflops"])
        color = colors.get(record["variant"], "#a78bfa")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        parts.append(
            f'<text x="{x+10:.1f}" y="{y-8:.1f}" fill="{color}" font-size="11">'
            f'{record["kernel"]}/{record["variant"]}</text>'
        )
    parts.append(f'<text x="{width/2}" y="24" fill="#e2e8f0" font-size="15" text-anchor="middle">NEON memory roofline (roof = {bandwidth:.1f} GB/s)</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" fill="#94a3b8" font-size="12" text-anchor="middle">Arithmetic intensity (FLOP/byte, log2)</text>')
    parts.append(f'<text x="18" y="{height/2}" fill="#94a3b8" font-size="12" transform="rotate(-90 18 {height/2})" text-anchor="middle">Attained GFLOP/s (log2)</text>')
    parts.append("</svg>")
    SVG_OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {CSV_OUT} and {SVG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
