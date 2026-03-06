#!/usr/bin/env python3
"""
Power monitoring via powermetrics plist output.

Can be used standalone or as a library alongside benchmarks.

Standalone usage:
    # Parse an existing powermetrics plist file
    python tests/parse_power.py /tmp/asitop_powermetrics*

Library usage (in benchmark.py):
    from tests.parse_power import PowerMonitor
    pm = PowerMonitor()          # auto-discovers asitop's powermetrics file
    pm.mark("gpu_prefill")       # start tracking a phase
    run_gpu_prefill(...)
    pm.mark("gpu_decode")        # end previous phase, start new one
    run_gpu_decode(...)
    pm.stop()                    # end last phase
    pm.report()                  # print summary
    pm.save("results/power.json")
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import plistlib
import re
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path


@dataclass
class PowerSample:
    """A single powermetrics sample."""
    cpu_power_mw: float = 0.0
    gpu_power_mw: float = 0.0
    ane_power_mw: float = 0.0
    combined_power_mw: float = 0.0
    thermal_pressure: str = ""


def find_powermetrics_file() -> str | None:
    """Find the powermetrics plist file written by asitop."""
    candidates = glob.glob("/tmp/asitop_powermetrics*")
    if not candidates:
        return None
    # Return the most recently modified one
    return max(candidates, key=os.path.getmtime)


def parse_plist_samples(data: bytes) -> list[PowerSample]:
    """Parse powermetrics plist data into PowerSample objects.

    The plist file is a sequence of plist documents concatenated together.
    Each document is a dict with keys like cpu_power, gpu_power, etc.
    """
    samples = []

    # Split on plist boundaries - each entry starts with <?xml or <plist
    # and ends with </plist>
    chunks = re.split(rb'(?=<\?xml|<plist)', data)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or b'</plist>' not in chunk:
            continue

        # Ensure chunk ends at </plist>
        end = chunk.find(b'</plist>')
        if end < 0:
            continue
        chunk = chunk[:end + len(b'</plist>')]

        try:
            d = plistlib.loads(chunk)
        except Exception:
            continue

        # Extract processor power from the plist dict
        sample = PowerSample()

        # Top-level power keys (direct from powermetrics)
        if "processor" in d:
            proc = d["processor"]
            sample.cpu_power_mw = float(proc.get("cpu_power", 0) or 0)
            sample.gpu_power_mw = float(proc.get("gpu_power", 0) or 0)
            sample.ane_power_mw = float(proc.get("ane_power", 0) or 0)
            sample.combined_power_mw = float(proc.get("combined_power", 0) or 0)
        else:
            # Flat structure (some powermetrics versions)
            sample.cpu_power_mw = float(d.get("cpu_power", 0) or 0)
            sample.gpu_power_mw = float(d.get("gpu_power", 0) or 0)
            sample.ane_power_mw = float(d.get("ane_power", 0) or 0)
            sample.combined_power_mw = float(d.get("combined_power", 0) or 0)

        if "thermal_pressure" in d:
            sample.thermal_pressure = str(d["thermal_pressure"])

        # Keep samples that have any power data
        if sample.cpu_power_mw > 0 or sample.gpu_power_mw > 0 or sample.ane_power_mw > 0:
            samples.append(sample)

    return samples


def summarize(samples: list[PowerSample]) -> dict:
    """Compute stats for a list of samples."""
    if not samples:
        return {"count": 0}

    def stats(vals):
        if not vals:
            return {"avg": 0, "min": 0, "max": 0}
        return {
            "avg": round(sum(vals) / len(vals), 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
        }

    return {
        "count": len(samples),
        "cpu_power_mw": stats([s.cpu_power_mw for s in samples]),
        "gpu_power_mw": stats([s.gpu_power_mw for s in samples]),
        "ane_power_mw": stats([s.ane_power_mw for s in samples]),
        "combined_power_mw": stats([s.combined_power_mw for s in samples]),
    }


class PowerMonitor:
    """Real-time power monitor that reads asitop's powermetrics plist file.

    Usage:
        pm = PowerMonitor()
        pm.mark("phase1")
        ...
        pm.mark("phase2")
        ...
        pm.stop()
        pm.report()
    """

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or find_powermetrics_file()
        if not self.filepath:
            print("[PowerMonitor] WARNING: No powermetrics file found. "
                  "Is asitop running? (sudo asitop)")
            self.filepath = None

        self.phases: list[tuple[str, int, int]] = []  # (name, start_offset, end_offset)
        self._current_phase: str | None = None
        self._current_offset: int = 0

    def _file_size(self) -> int:
        if not self.filepath:
            return 0
        try:
            return os.path.getsize(self.filepath)
        except OSError:
            return 0

    def mark(self, phase_name: str):
        """Start tracking a new phase. Ends the previous phase if any."""
        now = self._file_size()

        if self._current_phase is not None:
            self.phases.append((self._current_phase, self._current_offset, now))

        self._current_phase = phase_name
        self._current_offset = now

    def stop(self):
        """End the current phase."""
        if self._current_phase is not None:
            now = self._file_size()
            self.phases.append((self._current_phase, self._current_offset, now))
            self._current_phase = None

    def _read_range(self, start: int, end: int) -> bytes:
        """Read a byte range from the plist file."""
        if not self.filepath or start >= end:
            return b""
        with open(self.filepath, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def get_phase_data(self) -> dict[str, dict]:
        """Parse and summarize power data for all tracked phases."""
        results = {}
        for name, start, end in self.phases:
            data = self._read_range(start, end)
            samples = parse_plist_samples(data)
            results[name] = summarize(samples)
        return results

    def report(self):
        """Print a summary of power data for all phases."""
        data = self.get_phase_data()
        if not data:
            print("[PowerMonitor] No phase data recorded.")
            return

        print(f"\n{'='*72}")
        print("POWER MONITOR REPORT")
        print(f"{'='*72}")

        header = f"{'Phase':<25} {'CPU(mW)':>10} {'GPU(mW)':>10} {'ANE(mW)':>10} {'Total(mW)':>10} {'N':>5}"
        print(header)
        print("-" * 72)

        for name, stats in data.items():
            n = stats["count"]
            if n == 0:
                print(f"{name:<25} {'(no data)':>45}")
                continue
            cpu = stats["cpu_power_mw"]["avg"]
            gpu = stats["gpu_power_mw"]["avg"]
            ane = stats["ane_power_mw"]["avg"]
            total = stats["combined_power_mw"]["avg"]
            print(f"{name:<25} {cpu:>10.1f} {gpu:>10.1f} {ane:>10.1f} {total:>10.1f} {n:>5}")

        print(f"{'='*72}")

    def save(self, path: str):
        """Save phase data to JSON."""
        data = self.get_phase_data()
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[PowerMonitor] Saved to {path}")

    def to_dict(self) -> dict[str, dict]:
        """Return phase data as dict (for embedding in benchmark results)."""
        return self.get_phase_data()


# ─── Standalone CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse powermetrics plist output and show power summary"
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="powermetrics plist file (auto-discovers asitop's file if omitted)")
    parser.add_argument("--tail", type=int, default=0,
                        help="Only parse the last N bytes (useful for large files)")
    parser.add_argument("--output", default=None,
                        help="Save JSON summary to this path")
    args = parser.parse_args()

    filepath = args.input or find_powermetrics_file()
    if not filepath:
        print("ERROR: No powermetrics file found. Run 'sudo asitop' first.")
        sys.exit(1)

    print(f"Reading {filepath} ({os.path.getsize(filepath) / 1024 / 1024:.1f} MB)")

    if args.tail > 0:
        size = os.path.getsize(filepath)
        offset = max(0, size - args.tail)
        with open(filepath, "rb") as f:
            f.seek(offset)
            data = f.read()
    else:
        with open(filepath, "rb") as f:
            data = f.read()

    samples = parse_plist_samples(data)
    print(f"Parsed {len(samples)} power samples")

    if not samples:
        print("No power data found.")
        sys.exit(0)

    stats = summarize(samples)
    print(f"\n{'='*60}")
    print(f"Power Summary ({stats['count']} samples)")
    print(f"{'='*60}")
    for key in ["cpu_power_mw", "gpu_power_mw", "ane_power_mw", "combined_power_mw"]:
        s = stats[key]
        label = key.replace("_mw", " (mW)").replace("_", " ").title()
        print(f"  {label:<30} avg={s['avg']:>8.1f}  min={s['min']:>8.1f}  max={s['max']:>8.1f}")

    if args.output:
        Path(args.output).write_text(json.dumps(stats, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
