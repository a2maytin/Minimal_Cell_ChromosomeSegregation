#!/usr/bin/env python3
"""Combine timepoint(s) from Jan22_* replicates into one lammpstrj or XYZ for VMD.

Reads dna_Jan22_{rep}_{timepoint}.bin from Jan22_{rep}/data/coords/ and optionally
ribo_* files. Each replicate = one frame (timestep); step through frames in VMD
to compare replicates.

Examples:
  python combine_replicates_timepoint.py 86 -o combined_t86
  python combine_replicates_timepoint.py 50 86 89 -o multi -f xyz
  python combine_replicates_timepoint.py 86 --include-ribo -r 1-4 -d /path/to/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_bin(path: Path, order: str = "row") -> np.ndarray:
    data = np.fromfile(path, dtype=np.float64)
    if data.size % 3 != 0:
        raise ValueError(f"Binary file size not divisible by 3: {path}")
    N = data.size // 3
    if order == "row":
        return data.reshape((N, 3))
    if order == "col":
        out = np.empty((N, 3), dtype=np.float64)
        out[:, 0] = data[0:N]
        out[:, 1] = data[N : 2 * N]
        out[:, 2] = data[2 * N : 3 * N]
        return out
    raise ValueError("order must be 'row' or 'col'")


def gather_coords_per_replicate(
    base_dir: Path,
    run_label: str,
    replicates: list[int],
    timepoint: int,
    include_ribo: bool,
    bin_order: str,
) -> list[tuple[int, np.ndarray, str | None]]:
    out = []
    for r in replicates:
        coords_dir = base_dir / "../runs/" / f"{run_label}_r{r}" / "data" / "coords"
        dna_path = coords_dir / f"dna_{run_label}_r{r}_{timepoint}.bin"
        #coords_dir = base_dir / "../runs/" / f"{run_label}_{r}" / "data" / "coords"
        #dna_path = coords_dir / f"dna_{run_label}_{r}_{timepoint}.bin"
        if not dna_path.exists():
            continue
        dna = read_bin(dna_path, order=bin_order)
        out.append((r, dna, "dna"))
        if include_ribo:
            # Ribosome files live alongside DNA files and follow the same naming
            # convention, with an added "r" prefix for the replicate index.
            #ribo_path = coords_dir / f"ribo_{run_label}_p{r}_{timepoint}.bin"
            ribo_path = coords_dir / f"ribo_{run_label}_r{r}_{timepoint}.bin"
            if ribo_path.exists():
                ribo = read_bin(ribo_path, order=bin_order)
                out.append((r, ribo, "ribo"))
    return out


def _frames_by_replicate(
    frames: list[tuple[int, list[tuple[int, np.ndarray, str | None]]]],
) -> list[tuple[int, int, list[tuple[np.ndarray, str | None]]]]:
    """Convert to one frame per (timepoint, replicate). Group per_rep by rep_id."""
    out: list[tuple[int, int, list[tuple[np.ndarray, str | None]]]] = []
    for timepoint, per_rep in frames:
        by_rep: dict[int, list[tuple[np.ndarray, str | None]]] = {}
        for rep_id, coords, label in per_rep:
            if rep_id not in by_rep:
                by_rep[rep_id] = []
            by_rep[rep_id].append((coords, label))
        for rep_id in sorted(by_rep.keys()):
            out.append((timepoint, rep_id, by_rep[rep_id]))
    return out


def write_lammpstrj(
    flat_frames: list[tuple[int, int, list[tuple[np.ndarray, str | None]]]],
    path: Path,
    box_padding: float = 100.0,
) -> None:
    """One frame per replicate; timestep = frame index (1, 2, ...). Step through in VMD."""
    with open(path, "w") as f:
        for frame_idx, (timepoint, rep_id, coord_labels) in enumerate(flat_frames):
            coords = np.vstack([c for (c, _) in coord_labels])
            n = coords.shape[0]
            xmin, ymin, zmin = coords.min(axis=0) - box_padding
            xmax, ymax, zmax = coords.max(axis=0) + box_padding
            timestep = frame_idx + 1
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n}\n")
            f.write("ITEM: BOX BOUNDS ff ff ff\n")
            f.write(f"{xmin:.6e} {xmax:.6e}\n")
            f.write(f"{ymin:.6e} {ymax:.6e}\n")
            f.write(f"{zmin:.6e} {zmax:.6e}\n")
            # Match downstream analysis expectations: include tracking columns
            # so we can distinguish DNA vs ribosomes.
            f.write("ITEM: ATOMS id type x y z c_id_track c_type_track\n")
            aid = 1
            # Ensure DNA appears first (lower ids), then ribosomes.
            ordered_blocks = [
                (c, label) for (c, label) in coord_labels if label != "ribo"
            ] + [
                (c, label) for (c, label) in coord_labels if label == "ribo"
            ]
            for coords_block, label in ordered_blocks:
                typ = 2 if label == "ribo" else 1
                ctype = 2 if label == "ribo" else 3  # 2=ribosomes, 3=DNA
                for x, y, z in coords_block:
                    f.write(
                        f"{aid} {typ} {x:.6f} {y:.6f} {z:.6f} {aid} {ctype}\n"
                    )
                    aid += 1


def write_xyz(
    flat_frames: list[tuple[int, int, list[tuple[np.ndarray, str | None]]]],
    path: Path,
) -> None:
    """One frame per replicate."""
    with open(path, "w") as f:
        for timepoint, rep_id, coord_labels in flat_frames:
            coords = np.vstack([c for (c, _) in coord_labels])
            n = coords.shape[0]
            f.write(f"{n}\n")
            f.write(f"timepoint={timepoint} replicate={rep_id}\n")
            for coords_block, label in coord_labels:
                el = "O" if label == "ribo" else "C"
                for x, y, z in coords_block:
                    f.write(f"{el} {x:.6f} {y:.6f} {z:.6f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine timepoint(s) from Jan22_* replicates into one lammpstrj or XYZ."
    )
    ap.add_argument("timepoints", type=int, nargs="+", help="Timepoint(s) e.g. 50 86 89")
    ap.add_argument("-o", "--output", default="combined_replicates", help="Output path")
    ap.add_argument("-f", "--format", choices=("lammpstrj", "xyz"), default="lammpstrj")
    ap.add_argument("-d", "--base-dir", type=Path, default=Path("."))
    ap.add_argument("-r", "--replicates", default="1-8", help="e.g. 1-8 or 1,3,5")
    ap.add_argument("--run-label")
    ap.add_argument("--include-ribo", action="store_true")
    ap.add_argument("--bin-order", choices=("row", "col"), default="row")
    ap.add_argument("--box-padding", type=float, default=100.0)
    args = ap.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.is_dir():
        print(f"Base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    reps_str = args.replicates.strip()
    if "-" in reps_str and "," not in reps_str:
        a, b = reps_str.split("-", 1)
        replicates = list(range(int(a), int(b) + 1))
    else:
        replicates = [int(x) for x in reps_str.replace(",", " ").split()]

    frames = []
    for tp in args.timepoints:
        per_rep = gather_coords_per_replicate(
            base_dir, args.run_label, replicates, tp,
            args.include_ribo, args.bin_order,
        )
        if not per_rep:
            print(f"No data for timepoint {tp}.", file=sys.stderr)
            continue
        frames.append((tp, per_rep))
        n_atoms = sum(c.shape[0] for (_, c, _) in per_rep)
        reps_used = sorted({r for (r, _, _) in per_rep})
        print(f"Timepoint {tp}: {n_atoms} atoms from replicates {reps_used}")

    if not frames:
        print("No frames to write.", file=sys.stderr)
        sys.exit(1)

    ext = "lammpstrj" if args.format == "lammpstrj" else "xyz"
    out_path = Path(args.output)
    if out_path.suffix.lower() not in (".lammpstrj", ".xyz"):
        out_path = out_path.with_suffix("." + ext)

    flat_frames = _frames_by_replicate(frames)

    if args.format == "lammpstrj":
        write_lammpstrj(flat_frames, out_path, box_padding=args.box_padding)
    else:
        write_xyz(flat_frames, out_path)

    print(f"Wrote {out_path} ({len(flat_frames)} frame(s), one per replicate).")
    print("Step through frames in VMD to compare replicates.")


if __name__ == "__main__":
    main()
