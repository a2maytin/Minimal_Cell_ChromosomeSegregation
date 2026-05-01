#!/usr/bin/env python3
"""
Compute daughter-chromosome partitioning from a single replicate's .lammpstrj file
(instead of per-minute .bin files). Processes one replicate at a time; streams the
trajectory to handle large files.

Usage:
  python compute_partitioning_lammpstrj.py <base_dir> <replicate> [replicate2 ...] [options]

Example:
  python compute_partitioning_lammpstrj.py ../runs Mar03_p1 -o ../runs/partitioning_lammps -c 60
  python compute_partitioning_lammpstrj.py ../runs Mar03_p1 Mar03_p2 ... Mar03_p27 -o ../runs/partitioning -c 300 -j 20

Expects: <base_dir>/<replicate>/data/<replicate>.lammpstrj
Output:  one .txt file with frame index (0,1,2,...) and partitioning (default: base_dir/partitioning_lammpstrj/<replicate>.txt)

Bead filtering (same indexing as .bin):
  - c_type_track 1 = boundary beads (excluded)
  - c_type_track 2 = ribosomes (excluded)
  - c_type_track >= 3 = DNA beads; ordered by c_id_track to match .bin convention.
"""
from pathlib import Path
import argparse
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reuse partitioning logic from compute_partitioning
from compute_partitioning import (
    daughter_partitioning,
    N_PARENT,
)


# -----------------------------------------------------------------------------
# LAMMPS trajectory parsing (streaming)
# -----------------------------------------------------------------------------

def iter_lammpstrj_frames(path, dna_only=True, dna_min_ctype=3):
    """
    Yield (frame_index, coords) from a LAMMPS trajectory file.
    path: path to .lammpstrj
    dna_only: if True, keep only beads with c_type_track >= dna_min_ctype (DNA beads; default 3)
    frame_index: 0-based frame number (0, 1, 2, ...); ITEM: TIMESTEP in file is ignored.
    coords: (N_dna, 3) float array in bead order by c_id_track (0-based index).
    """
    path = Path(path)
    frame_index = 0
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue
            f.readline()  # consume timestep line (not used)

            # NUMBER OF ATOMS
            line = f.readline()
            if "ITEM: NUMBER OF ATOMS" not in line:
                break
            n_atoms = int(f.readline().strip())

            # BOX BOUNDS
            line = f.readline()
            if "ITEM: BOX" not in line:
                break
            for _ in range(3):
                f.readline()

            # ATOMS header: "ITEM: ATOMS id type x y z c_id_track c_type_track"
            line = f.readline()
            if "ITEM: ATOMS" not in line:
                break
            # Column names start after "ITEM: ATOMS" so indices match data rows (7 columns)
            header = line.strip().split()[2:]
            # Find column indices (0-based)
            try:
                idx_id = header.index("id")
                idx_type = header.index("type")
                idx_x = header.index("x")
                idx_y = header.index("y")
                idx_z = header.index("z")
                idx_c_id = header.index("c_id_track")
                idx_c_type = header.index("c_type_track")
            except ValueError as e:
                raise ValueError(f"Expected ATOMS columns id type x y z c_id_track c_type_track: {header}") from e

            # Read n_atoms lines
            rows = []
            for _ in range(n_atoms):
                line = f.readline()
                if not line:
                    break
                parts = line.split()
                if len(parts) <= max(idx_c_type, idx_c_id, idx_z):
                    continue
                c_type = int(parts[idx_c_type])
                if dna_only and c_type < dna_min_ctype:
                    continue
                c_id = int(parts[idx_c_id])
                x = float(parts[idx_x])
                y = float(parts[idx_y])
                z = float(parts[idx_z])
                rows.append((c_id, x, y, z))

            if not rows:
                yield frame_index, np.empty((0, 3), dtype=np.float64)
                frame_index += 1
                continue

            # Sort by c_id_track to get same order as .bin (bead index 0, 1, 2, ...)
            rows.sort(key=lambda r: r[0])
            coords = np.array([[r[1], r[2], r[3]] for r in rows], dtype=np.float64)
            yield frame_index, coords
            frame_index += 1


def _compute_partitioning_one(args):
    """Compute partitioning for one frame (for use in thread pool). args = (frame_idx, coords, cutoff, box_length)."""
    frame_idx, coords, cutoff, box_length = args
    N = coords.shape[0]
    if N < N_PARENT:
        return frame_idx, N, np.nan
    p = daughter_partitioning(coords, cutoff=cutoff, box_length=box_length, verbose=False)
    return frame_idx, N, p


def run_partitioning_lammpstrj(
    base_dir,
    replicate,
    out_path=None,
    cutoff=60.0,
    box_length=None,
    stride=1,
    jobs=1,
    verbose=True,
):
    """
    Stream <base_dir>/<replicate>/data/<replicate>.lammpstrj, compute partitioning
    per frame (optionally every stride-th frame), write frame index and partitioning to out_path.
    jobs > 1 uses a process pool to compute partitioning in parallel (avoids Python GIL); frames are written in order.
    """
    base_dir = Path(base_dir)
    trj_path = base_dir / replicate / "data" / f"{replicate}.lammpstrj"
    if not trj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trj_path}")

    if out_path is None:
        out_path = base_dir / "partitioning_lammpstrj" / f"{replicate}.txt"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = max(1, 2 * jobs) if jobs > 1 else 1
    n_written = 0
    next_frame_to_write = 0
    results_buffer = {}

    with open(out_path, "w") as out_file:
        out_file.write("# partitioning from .lammpstrj (DNA beads: c_type_track >= 3)\n")
        out_file.write(f"# cutoff={cutoff}\n")
        out_file.write(f"# replicate={replicate}\n")
        out_file.write("#\n# frame\tpartitioning\n")
        out_file.flush()

        def process_batch(batch):
            """Run partitioning on a batch and write results in frame order."""
            nonlocal next_frame_to_write, n_written
            if not batch:
                return
            if jobs <= 1:
                for (frame_idx, coords) in batch:
                    _, N, p = _compute_partitioning_one((frame_idx, coords, cutoff, box_length))
                    out_file.write(f"{frame_idx}\t{p}\n")
                    n_written += 1
                    if verbose and n_written % 100 == 0:
                        print(f"  {replicate} frame {frame_idx}: N={N}, partitioning={p:.4f}  (wrote {n_written} frames)")
                out_file.flush()
                return
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(
                        _compute_partitioning_one,
                        (frame_idx, coords, cutoff, box_length),
                    ): frame_idx
                    for frame_idx, coords in batch
                }
                for future in as_completed(futures):
                    frame_idx, N, p = future.result()
                    results_buffer[frame_idx] = (N, p)
                    while next_frame_to_write in results_buffer:
                        N, p = results_buffer.pop(next_frame_to_write)
                        out_file.write(f"{next_frame_to_write}\t{p}\n")
                        n_written += 1
                        if verbose and n_written % 100 == 0:
                            print(f"  {replicate} frame {next_frame_to_write}: N={N}, partitioning={p:.4f}  (wrote {n_written} frames)")
                        next_frame_to_write += stride
            out_file.flush()

        batch = []
        for frame_index, coords in iter_lammpstrj_frames(trj_path, dna_only=True, dna_min_ctype=3):
            if (stride > 1) and (frame_index % stride != 0):
                continue
            batch.append((frame_index, coords))
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []
        process_batch(batch)

    if verbose:
        print(f"Wrote {n_written} frames to {out_path}")
    return out_path, n_written


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute partitioning from one or more replicate .lammpstrj files (streaming)."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory containing replicate folders (e.g. runs/)",
    )
    parser.add_argument(
        "replicates",
        type=str,
        nargs="+",
        metavar="replicate",
        help="Replicate name(s) (e.g. Mar03_p1, or Mar03_p1 Mar03_p2 ... Mar03_p27). File: base_dir/replicate/data/replicate.lammpstrj",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .txt file (single replicate) or output directory (multiple replicates; files <replicate>_lammpstrj.txt). Default: base_dir/partitioning_lammpstrj/",
    )
    parser.add_argument(
        "-c", "--cutoff",
        type=float,
        default=60.0,
        help="Contact cutoff distance (default: 60.0)",
    )
    parser.add_argument(
        "--box-length",
        type=float,
        default=None,
        help="Periodic box length for PBC (default: none)",
    )
    parser.add_argument(
        "-s", "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of processes for parallel partitioning (default: 1)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Less progress output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing output files; default is to skip replicates that already have output.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    replicates = args.replicates
    out_spec = args.output.resolve() if args.output else None
    verbose = not args.quiet

    for replicate in replicates:
        trj_path = base_dir / replicate / "data" / f"{replicate}.lammpstrj"
        if not trj_path.exists():
            if verbose:
                print(f"Skipping {replicate}: no trajectory at {trj_path}")
            continue

        if len(replicates) == 1:
            out_path = out_spec
        else:
            out_dir = out_spec if out_spec is not None else base_dir / "partitioning_lammpstrj"
            out_dir = Path(out_dir)
            if out_dir.suffix:
                out_dir = out_dir.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{replicate}_lammpstrj.txt"

        # Resolve effective output path for skip-if-exists (same as compute_partitioning.py)
        effective_out = out_path if out_path is not None else base_dir / "partitioning_lammpstrj" / f"{replicate}.txt"
        if effective_out.exists() and not args.overwrite:
            if verbose:
                print(f"Skipping {replicate}: already processed ({effective_out})")
            continue

        try:
            run_partitioning_lammpstrj(
                base_dir,
                replicate,
                out_path=out_path,
                cutoff=args.cutoff,
                box_length=args.box_length,
                stride=args.stride,
                jobs=args.jobs,
                verbose=verbose,
            )
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
