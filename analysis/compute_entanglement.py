#!/usr/bin/env python3
"""
Standalone script to compute daughter-chromosome entanglement (Gauss linking number)
for all replicates and save one .txt file per replicate in an entanglement/ folder.

This mirrors compute_partitioning.py, but instead of a partitioning metric based on
cross-daughter contacts, it computes the Gauss linking number between the two daughter
curves defined by their bead coordinates.
"""

from pathlib import Path
import re
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------


def read_bin(path, order="row"):
    data = np.fromfile(path, dtype=np.float64)
    n_coords = data.size
    if n_coords % 3 != 0:
        raise ValueError(f"{path} size not divisible by 3")
    N = n_coords // 3
    if order == "row":
        coords = data.reshape((N, 3))
    elif order == "col":
        coords = np.empty((N, 3))
        coords[:, 0] = data[0:N]
        coords[:, 1] = data[N:2 * N]
        coords[:, 2] = data[2 * N:3 * N]
    else:
        raise ValueError("order must be 'row' or 'col'")
    return coords


# -----------------------------------------------------------------------------
# Entanglement metric (Gauss linking number between daughters)
# -----------------------------------------------------------------------------


N_PARENT = 54338
MIDPOINT_1BASED = 27169  # 0-based index = 27168


def _daughter_indices(N):
    """
    Return (left_daughter_indices, right_daughter_indices) for a system with N beads,
    using the same convention as compute_partitioning.daughter_partitioning.
    """
    if N <= N_PARENT:
        return None, None
    N_repl = N - N_PARENT
    right_daughter = np.arange(N_PARENT, N)
    left_start_0 = (MIDPOINT_1BASED - 1) - (N_repl - 1) // 2
    left_end_0 = (MIDPOINT_1BASED - 1) + N_repl // 2 + 1
    left_daughter = np.arange(left_start_0, left_end_0)
    assert len(left_daughter) == N_repl and len(right_daughter) == N_repl
    return left_daughter, right_daughter


def _close_curve(points):
    """
    Ensure the curve is closed by connecting its endpoints with a straight segment.
    If the first and last points are (almost) identical, return as-is.
    """
    if len(points) < 2:
        return points
    if np.allclose(points[0], points[-1]):
        return points
    return np.vstack([points, points[0]])


def gauss_linking_number(curve1, curve2, close_curves=True, eps=1e-9):
    """
    Approximate Gauss linking number between two (possibly open) polygonal curves.

    For closed, disjoint curves this integral is an integer topological invariant.
    For open or partially overlapping curves it is a real-valued entanglement measure.

    curve1, curve2: (M, 3) and (N, 3) arrays of 3D points along each curve.
    close_curves: if True, close each curve by a straight segment between its endpoints
                  before computing the linking number.
    """
    if close_curves:
        c1 = _close_curve(np.asarray(curve1, dtype=float))
        c2 = _close_curve(np.asarray(curve2, dtype=float))
    else:
        c1 = np.asarray(curve1, dtype=float)
        c2 = np.asarray(curve2, dtype=float)

    if c1.shape[0] < 2 or c2.shape[0] < 2:
        return np.nan

    # Segment vectors and midpoints
    s1 = c1[1:] - c1[:-1]  # (n1, 3)
    s2 = c2[1:] - c2[:-1]  # (n2, 3)
    m1 = 0.5 * (c1[1:] + c1[:-1])
    m2 = 0.5 * (c2[1:] + c2[:-1])

    # Displacement between all midpoint pairs r[i,j] = m1[i] - m2[j]
    r = m1[:, None, :] - m2[None, :, :]  # (n1, n2, 3)
    dist = np.linalg.norm(r, axis=2)
    dist = np.maximum(dist, eps)  # avoid division by zero

    # Cross product of segment vectors for all pairs
    cross = np.cross(s1[:, None, :], s2[None, :, :])  # (n1, n2, 3)
    num = (cross * r).sum(axis=2)
    total = (num / (dist ** 3)).sum()
    return float(total / (4.0 * np.pi))


def daughter_linking_number(coords, stride=10, close_curves=True, verbose=False):
    """
    Compute daughter-daughter Gauss linking number for one frame.

    coords: (N, 3) array of bead coordinates.
    stride: subsampling stride along each daughter (1 = use every bead).
            Larger stride speeds up computation at the cost of accuracy.
    close_curves: if True, close each daughter by a straight segment between
                  its endpoints (replication forks) before computing Lk.

    Returns float (linking number, not rounded to integer) or NaN if no replication.
    """
    N = len(coords)
    left_daughter, right_daughter = _daughter_indices(N)
    if left_daughter is None or right_daughter is None:
        return np.nan  # no replication yet

    if stride is None or stride < 1:
        stride = 1

    left_pts = coords[left_daughter[::stride]]
    right_pts = coords[right_daughter[::stride]]

    if left_pts.shape[0] < 2 or right_pts.shape[0] < 2:
        return np.nan

    lk = gauss_linking_number(left_pts, right_pts, close_curves=close_curves)
    if verbose:
        print(f"    linking number (stride={stride}): {lk}")
    return lk


# -----------------------------------------------------------------------------
# Collect and save
# -----------------------------------------------------------------------------


def _load_existing_entanglement(txt_path):
    """
    Read existing minute and entanglement arrays from a .txt file.
    Returns (minutes_array, ent_array) or ([], []) if file missing/empty/invalid.
    """
    path = Path(txt_path)
    if not path.exists():
        return [], []
    minutes, ent = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    minutes.append(int(parts[0]))
                    ent.append(float(parts[1]))
                except (ValueError, TypeError):
                    pass
    return minutes, ent


def _process_one_replicate(args):
    """
    Worker: process a single replicate (discover files, compute entanglement, write .txt).
    If rep.txt already exists, only process bin files whose minute is not yet in the file and append.
    args: (base_dir, rep, stride, out_dir, verbose, close_curves).
    Returns (rep, minutes_array, ent_array) or (rep, None, None) if no files.
    """
    base_dir, rep, stride, out_dir, verbose, close_curves = args
    base_dir = Path(base_dir)
    coords_dir = base_dir / rep / "data" / "coords"
    if not coords_dir.exists():
        return (rep, None, None)
    pattern = re.compile(rf"dna_{re.escape(rep)}_(\d+)\.bin")
    all_files = sorted(
        [f for f in coords_dir.glob("dna_*.bin") if pattern.match(f.name)],
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not all_files:
        return (rep, None, None)
    out_path = Path(out_dir) / f"{rep}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_minutes_list, existing_ent_list = _load_existing_entanglement(out_path)
    existing_minutes = set(existing_minutes_list)
    files = [f for f in all_files if int(pattern.match(f.name).group(1)) not in existing_minutes]
    if not files:
        if verbose:
            print(f"{rep}: no new bin files (all {len(all_files)} already in {out_path})")
        if existing_minutes_list:
            return (rep, np.array(existing_minutes_list), np.array(existing_ent_list))
        return (rep, None, None)

    if verbose:
        print(
            f"{rep}: processing {len(files)} new bin files "
            f"(skipping {len(all_files) - len(files)} already done) ..."
        )
    minutes_new = []
    ent_new = []
    total = len(files)
    append_mode = out_path.exists()
    with open(out_path, "a" if append_mode else "w") as out_file:
        if not append_mode:
            out_file.write("# entanglement (Gauss linking number) time series\n")
            out_file.write(f"# stride={stride}\n")
            out_file.write(f"# base_dir={base_dir.resolve()}\n")
            out_file.write(f"# replicate={rep}\n")
            out_file.write("#\n# minute\tlinking_number\n")
            out_file.flush()
        for i, f in enumerate(files):
            m = pattern.match(f.name)
            minute = int(m.group(1))
            coords = read_bin(f)
            lk = daughter_linking_number(coords, stride=stride, close_curves=close_curves, verbose=verbose)
            minutes_new.append(minute)
            ent_new.append(lk)
            out_file.write(f"{int(minute)}\t{lk}\n")
            out_file.flush()
            if verbose:
                print(
                    f"  {rep} minute {minute}: linking_number={lk:.6f}  "
                    f"({i + 1}/{total} new files)"
                )
    if verbose:
        print(f"  Wrote {rep} to {out_path}" + (" (appended)" if append_mode else ""))
    # Merge existing + new and sort by minute
    all_minutes = existing_minutes_list + minutes_new
    all_ent = existing_ent_list + ent_new
    order = np.argsort(all_minutes)
    return (rep, np.array(all_minutes)[order], np.array(all_ent)[order])


def collect_entanglement_time_series(
    base_dir,
    replicates=None,
    stride=10,
    verbose=True,
    out_dir=None,
    n_jobs=1,
    close_curves=True,
    name_prefixes=None,
):
    """
    For each replicate folder under base_dir (e.g. Feb08_1), find all dna_<rep>_<minute>.bin
    in data/coords, compute daughter-daughter Gauss linking number per file, and return
    dict replicate -> (minutes, linking_numbers).

    replicates: list of names or None to discover all with coords.
    out_dir: if set, write each replicate's .txt incrementally (one line per bin file).
    n_jobs: number of parallel replicates (1 = sequential); use 0 for cpu_count().
    """
    base_dir = Path(base_dir)
    if name_prefixes is not None and not isinstance(name_prefixes, (list, tuple)):
        name_prefixes = [name_prefixes]
    if replicates is None:
        coords_dirs = list(base_dir.glob("*_*/data/coords"))
        replicates = sorted([d.parent.parent.name for d in coords_dirs])
        if name_prefixes:
            replicates = [
                r for r in replicates if any(r.startswith(p) for p in name_prefixes)
            ]
    else:
        replicates = sorted(replicates)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    if n_jobs != 1:
        n_workers = cpu_count() if n_jobs <= 0 else n_jobs
        args_list = [(base_dir, rep, stride, out_dir, verbose, close_curves) for rep in replicates]
        with Pool(processes=n_workers) as pool:
            results = pool.map(_process_one_replicate, args_list)
        result = {}
        for rep, minutes, ent in results:
            if minutes is not None:
                result[rep] = (minutes, ent)
        return result

    # Sequential path (incremental: only process new bin files, append to existing .txt)
    result = {}
    for rep in replicates:
        coords_dir = base_dir / rep / "data" / "coords"
        if not coords_dir.exists():
            continue
        pattern = re.compile(rf"dna_{re.escape(rep)}_(\d+)\.bin")
        all_files = sorted(
            [f for f in coords_dir.glob("dna_*.bin") if pattern.match(f.name)],
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not all_files:
            continue
        filepath = Path(out_dir) / f"{rep}.txt" if out_dir else None
        existing_minutes_list, existing_ent_list = (
            _load_existing_entanglement(filepath) if filepath else ([], [])
        )
        existing_minutes = set(existing_minutes_list)
        files = [f for f in all_files if int(pattern.match(f.name).group(1)) not in existing_minutes]
        if not files:
            if verbose:
                print(
                    f"{rep}: no new bin files (all {len(all_files)} already in {filepath})"
                )
            if existing_minutes_list:
                result[rep] = (np.array(existing_minutes_list), np.array(existing_ent_list))
            continue
        if verbose:
            print(
                f"{rep}: processing {len(files)} new bin files "
                f"(skipping {len(all_files) - len(files)} already done) ..."
            )
        minutes = []
        ents = []
        total = len(files)
        out_file = None
        append_mode = filepath.exists() if filepath else False
        if out_dir is not None:
            out_file = open(filepath, "a" if append_mode else "w")
            if not append_mode:
                out_file.write("# entanglement (Gauss linking number) time series\n")
                out_file.write(f"# stride={stride}\n")
                out_file.write(f"# base_dir={base_dir.resolve()}\n")
                out_file.write(f"# replicate={rep}\n")
                out_file.write("#\n# minute\tlinking_number\n")
                out_file.flush()
        try:
            for i, f in enumerate(files):
                m = pattern.match(f.name)
                minute = int(m.group(1))
                coords = read_bin(f)
                lk = daughter_linking_number(
                    coords, stride=stride, close_curves=close_curves, verbose=verbose
                )
                minutes.append(minute)
                ents.append(lk)
                if out_file is not None:
                    out_file.write(f"{int(minute)}\t{lk}\n")
                    out_file.flush()
                if verbose:
                    print(
                        f"  {rep} minute {minute}: linking_number={lk:.6f}  "
                        f"({i + 1}/{total} new files)"
                    )
        finally:
            if out_file is not None:
                out_file.close()
                if verbose:
                    print(
                        f"  Wrote {rep} to {filepath}" + (" (appended)" if append_mode else "")
                    )
        all_minutes = existing_minutes_list + minutes
        all_ent = existing_ent_list + ents
        order = np.argsort(all_minutes)
        result[rep] = (np.array(all_minutes)[order], np.array(all_ent)[order])
    return result


def save_entanglement_series(series, folder_path, stride, base_dir):
    """Save entanglement time series: one file per replicate in folder_path (e.g. entanglement/Feb08_1.txt)."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    base_dir_resolved = str(Path(base_dir).resolve())
    for rep in series:
        filepath = folder / f"{rep}.txt"
        with open(filepath, "w") as f:
            f.write("# entanglement (Gauss linking number) time series\n")
            f.write(f"# stride={stride}\n")
            f.write(f"# base_dir={base_dir_resolved}\n")
            f.write(f"# replicate={rep}\n")
            f.write("#\n# minute\tlinking_number\n")
            minutes, ent = series[rep]
            for m, lk in zip(minutes, ent):
                f.write(f"{int(m)}\t{lk}\n")
        print(f"  Saved {rep} to {filepath}")
    print(f"Saved entanglement results to {folder}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute daughter-chromosome entanglement (Gauss linking number) for all "
            "replicates and save to entanglement/*.txt"
        )
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        nargs="?",
        default=Path("../runs/"),
        help=(
            "Base directory containing replicate folders (e.g. Feb08_1/data/coords); "
            "default: ../runs/"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder for .txt files (default: base_dir/entanglement)",
    )
    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        default=10,
        help="Subsampling stride along each daughter (1 = use every bead; default: 10)",
    )
    parser.add_argument(
        "--no-close",
        action="store_true",
        help="Do not close daughters with a straight segment between endpoints "
        "before computing the linking number.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less progress output (still print per-replicate summary)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of replicates to process in parallel (default: 1). Use 0 for all CPUs.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        action="append",
        default=None,
        help=(
            "Only process replicates whose names start with this prefix "
            "(e.g. 'Feb16' or 'Feb16_p1'). Can be given multiple times."
        ),
    )
    args = parser.parse_args()
    base_dir = args.base_dir.resolve()
    out_dir = args.output if args.output is not None else base_dir / "entanglement"
    out_dir = out_dir.resolve()
    n_jobs = args.jobs
    close_curves = not args.no_close
    prefixes = args.prefix

    print(f"Base dir: {base_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Stride: {args.stride}")
    print(f"Close curves: {close_curves}")
    if n_jobs != 1:
        n_workers = cpu_count() if n_jobs <= 0 else n_jobs
        print(f"Parallel jobs: {n_workers}")
    series = collect_entanglement_time_series(
        base_dir,
        replicates=None,
        stride=args.stride,
        verbose=not args.quiet,
        out_dir=out_dir,
        n_jobs=n_jobs,
        close_curves=close_curves,
        name_prefixes=prefixes,
    )
    if not series:
        print("No replicates with dna_* bin files found.")
        return 1
    print(f"Computed entanglement for {len(series)} replicates; results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

