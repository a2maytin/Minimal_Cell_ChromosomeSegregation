#!/usr/bin/env python3
"""
Standalone script to compute daughter-chromosome partitioning for all replicates
and save one .txt file per replicate in a partitioning/ folder.
Uses the same logic as the analyze_trajectories.ipynb partitioning cells.
"""
from pathlib import Path
import re
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree

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
        coords[:, 1] = data[N:2*N]
        coords[:, 2] = data[2*N:3*N]
    else:
        raise ValueError("order must be 'row' or 'col'")
    return coords


# -----------------------------------------------------------------------------
# Partitioning metric
# -----------------------------------------------------------------------------

N_PARENT = 54338
MIDPOINT_1BASED = 27169  # 0-based index = 27168


def daughter_partitioning(coords, cutoff=60.0, box_length=None, verbose=False):
    """
    Compute partitioning metric for one frame using cKDTree neighbor search (or voxels if box_length set).
    Returns float in [0, 1]: 1 = no cross-daughter contacts, 0 = all daughter beads contact the other daughter.
    """
    N = len(coords)
    if N <= N_PARENT:
        return np.nan  # no replication
    N_repl = N - N_PARENT
    # 0-based indices: right daughter 54338..N-1; left 1-based 27169-N_repl/2+1 to 27169+N_repl/2
    right_daughter = np.arange(N_PARENT, N)
    left_start_0 = (MIDPOINT_1BASED - 1) - (N_repl - 1) // 2
    left_end_0 = (MIDPOINT_1BASED - 1) + N_repl // 2 + 1
    left_daughter = np.arange(left_start_0, left_end_0)
    assert len(left_daughter) == N_repl and len(right_daughter) == N_repl

    # Label: 0 = other, 1 = left daughter, 2 = right daughter
    label = np.zeros(N, dtype=np.int8)
    label[left_daughter] = 1
    label[right_daughter] = 2

    marked = set()
    if box_length is not None:
        # Periodic box: use voxel method (cKDTree doesn't handle PBC)
        _daughter_partitioning_voxel(
            coords, N, left_daughter, right_daughter, label, cutoff, box_length, marked
        )
    else:
        # No PBC: fast cKDTree path O(N_repl * log N) per frame
        tree = cKDTree(coords)
        cutoff_sq = cutoff * cutoff
        for i in left_daughter:
            neighbors = tree.query_ball_point(coords[i], cutoff)
            for j in neighbors:
                if i != j and label[j] == 2:
                    marked.add(i)
                    break
        for i in right_daughter:
            neighbors = tree.query_ball_point(coords[i], cutoff)
            for j in neighbors:
                if i != j and label[j] == 1:
                    marked.add(i)
                    break

    n_marked = len(marked)
    total_daughter = 2 * N_repl
    if verbose:
        print(f"    marked {n_marked} out of total {total_daughter}")
    partitioning = 1.0 - (n_marked / total_daughter)
    return partitioning


def _daughter_partitioning_voxel(coords, N, left_daughter, right_daughter, label, cutoff, box_length, marked):
    """Voxel-based contact search (used when box_length is set for PBC)."""
    cell_size = cutoff
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    box_extent = (max_coord - min_coord).max()
    ncell = max(1, int(np.ceil(box_extent / cell_size)))
    cell_idx = np.floor((coords - min_coord) / cell_size).astype(int)
    cell_idx = np.clip(cell_idx, 0, ncell - 1)
    flat_idx = cell_idx[:, 0] + ncell * (cell_idx[:, 1] + ncell * cell_idx[:, 2])
    cell_particles = [[] for _ in range(ncell ** 3)]
    for i, c in enumerate(flat_idx):
        cell_particles[c].append(i)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for cx in range(ncell):
        for cy in range(ncell):
            for cz in range(ncell):
                cflat = cx + ncell * (cy + ncell * cz)
                beads1 = cell_particles[cflat]
                if not beads1:
                    continue
                for dx, dy, dz in offsets:
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if not (0 <= nx < ncell and 0 <= ny < ncell and 0 <= nz < ncell):
                        continue
                    nflat = nx + ncell * (ny + ncell * nz)
                    beads2 = cell_particles[nflat]
                    for i in beads1:
                        li = label[i]
                        if li == 0:
                            continue
                        for j in beads2:
                            if i == j or label[j] == li:
                                continue
                            d = coords[i] - coords[j]
                            d -= box_length * np.round(d / box_length)
                            if np.dot(d, d) < cutoff ** 2:
                                marked.add(i)
                                marked.add(j)
                                break


# -----------------------------------------------------------------------------
# Collect and save
# -----------------------------------------------------------------------------

def _load_existing_partitioning(txt_path):
    """
    Read existing minute and partitioning arrays from a .txt file.
    Returns (minutes_array, part_array) or ([], []) if file missing/empty/invalid.
    """
    path = Path(txt_path)
    if not path.exists():
        return [], []
    minutes, part = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    minutes.append(int(parts[0]))
                    part.append(float(parts[1]))
                except (ValueError, TypeError):
                    pass
    return minutes, part


def _process_one_replicate(args):
    """
    Worker: process a single replicate (discover files, compute partitioning, write .txt).
    If rep.txt already exists, only process bin files whose minute is not yet in the file and append.
    args: (base_dir, rep, cutoff, out_dir, verbose).
    Returns (rep, minutes_array, part_array) or (rep, None, None) if no files.
    """
    base_dir, rep, cutoff, out_dir, verbose = args
    base_dir = Path(base_dir)
    coords_dir = base_dir / rep / "data" / "coords"
    if not coords_dir.exists():
        return (rep, None, None)
    pattern = re.compile(rf"dna_{re.escape(rep)}_(\d+)\.bin")
    all_files = sorted([f for f in coords_dir.glob("dna_*.bin") if pattern.match(f.name)],
                       key=lambda p: int(p.stem.split("_")[-1]))
    if not all_files:
        return (rep, None, None)
    out_path = Path(out_dir) / f"{rep}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_minutes_list, existing_part_list = _load_existing_partitioning(out_path)
    existing_minutes = set(existing_minutes_list)
    files = [f for f in all_files if int(pattern.match(f.name).group(1)) not in existing_minutes]
    if not files:
        if verbose:
            print(f"{rep}: no new bin files (all {len(all_files)} already in {out_path})")
        if existing_minutes_list:
            return (rep, np.array(existing_minutes_list), np.array(existing_part_list))
        return (rep, None, None)

    if verbose:
        print(f"{rep}: processing {len(files)} new bin files (skipping {len(all_files) - len(files)} already done) ...")
    minutes_new = []
    part_new = []
    total = len(files)
    append_mode = out_path.exists()
    with open(out_path, "a" if append_mode else "w") as out_file:
        if not append_mode:
            out_file.write("# partitioning time series\n")
            out_file.write(f"# cutoff={cutoff}\n")
            out_file.write(f"# base_dir={base_dir.resolve()}\n")
            out_file.write(f"# replicate={rep}\n")
            out_file.write("#\n# minute\tpartitioning\n")
            out_file.flush()
        for i, f in enumerate(files):
            m = pattern.match(f.name)
            minute = int(m.group(1))
            coords = read_bin(f)
            p = daughter_partitioning(coords, cutoff=cutoff, verbose=verbose)
            minutes_new.append(minute)
            part_new.append(p)
            out_file.write(f"{int(minute)}\t{p}\n")
            out_file.flush()
            if verbose:
                print(f"  {rep} minute {minute}: partitioning={p:.4f}  ({i+1}/{total} new files)")
    if verbose:
        print(f"  Wrote {rep} to {out_path}" + (" (appended)" if append_mode else ""))
    # Merge existing + new and sort by minute
    all_minutes = existing_minutes_list + minutes_new
    all_part = existing_part_list + part_new
    order = np.argsort(all_minutes)
    return (rep, np.array(all_minutes)[order], np.array(all_part)[order])


def collect_partitioning_time_series(base_dir, replicates=None, cutoff=60.0, verbose=True, out_dir=None, n_jobs=1):
    """
    For each replicate folder under base_dir (e.g. Feb08_1), find all dna_<rep>_<minute>.bin
    in data/coords, compute partitioning per file, return dict replicate -> (minutes, partitionings).
    replicates: list of names or None to discover all with coords.
    out_dir: if set, write each replicate's .txt incrementally (one line per bin file).
    n_jobs: number of parallel replicates (1 = sequential); use 0 for cpu_count().
    """
    base_dir = Path(base_dir)
    if replicates is None:
        coords_dirs = list(base_dir.glob("*_*/data/coords"))
        replicates = sorted([d.parent.parent.name for d in coords_dirs])
    else:
        replicates = sorted(replicates)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    if n_jobs != 1:
        n_workers = cpu_count() if n_jobs <= 0 else n_jobs
        args_list = [(base_dir, rep, cutoff, out_dir, verbose) for rep in replicates]
        with Pool(processes=n_workers) as pool:
            results = pool.map(_process_one_replicate, args_list)
        result = {}
        for rep, minutes, part in results:
            if minutes is not None:
                result[rep] = (minutes, part)
        return result

    # Sequential path (incremental: only process new bin files, append to existing .txt)
    result = {}
    for rep in replicates:
        coords_dir = base_dir / rep / "data" / "coords"
        if not coords_dir.exists():
            continue
        pattern = re.compile(rf"dna_{re.escape(rep)}_(\d+)\.bin")
        all_files = sorted([f for f in coords_dir.glob("dna_*.bin") if pattern.match(f.name)],
                           key=lambda p: int(p.stem.split("_")[-1]))
        if not all_files:
            continue
        filepath = Path(out_dir) / f"{rep}.txt" if out_dir else None
        existing_minutes_list, existing_part_list = _load_existing_partitioning(filepath) if filepath else ([], [])
        existing_minutes = set(existing_minutes_list)
        files = [f for f in all_files if int(pattern.match(f.name).group(1)) not in existing_minutes]
        if not files:
            if verbose:
                print(f"{rep}: no new bin files (all {len(all_files)} already in {filepath})")
            if existing_minutes_list:
                result[rep] = (np.array(existing_minutes_list), np.array(existing_part_list))
            continue
        if verbose:
            print(f"{rep}: processing {len(files)} new bin files (skipping {len(all_files) - len(files)} already done) ...")
        minutes = []
        part = []
        total = len(files)
        out_file = None
        append_mode = filepath.exists() if filepath else False
        if out_dir is not None:
            out_file = open(filepath, "a" if append_mode else "w")
            if not append_mode:
                out_file.write("# partitioning time series\n")
                out_file.write(f"# cutoff={cutoff}\n")
                out_file.write(f"# base_dir={base_dir.resolve()}\n")
                out_file.write(f"# replicate={rep}\n")
                out_file.write("#\n# minute\tpartitioning\n")
                out_file.flush()
        try:
            for i, f in enumerate(files):
                m = pattern.match(f.name)
                minute = int(m.group(1))
                coords = read_bin(f)
                p = daughter_partitioning(coords, cutoff=cutoff, verbose=verbose)
                minutes.append(minute)
                part.append(p)
                if out_file is not None:
                    out_file.write(f"{int(minute)}\t{p}\n")
                    out_file.flush()
                if verbose:
                    print(f"  {rep} minute {minute}: partitioning={p:.4f}  ({i+1}/{total} new files)")
        finally:
            if out_file is not None:
                out_file.close()
                if verbose:
                    print(f"  Wrote {rep} to {filepath}" + (" (appended)" if append_mode else ""))
        all_minutes = existing_minutes_list + minutes
        all_part = existing_part_list + part
        order = np.argsort(all_minutes)
        result[rep] = (np.array(all_minutes)[order], np.array(all_part)[order])
    return result


def save_partitioning_series(series, folder_path, cutoff, base_dir):
    """Save partitioning time series: one file per replicate in folder_path (e.g. partitioning/Feb08_1.txt)."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    base_dir_resolved = str(Path(base_dir).resolve())
    for rep in series:
        filepath = folder / f"{rep}.txt"
        with open(filepath, "w") as f:
            f.write("# partitioning time series\n")
            f.write(f"# cutoff={cutoff}\n")
            f.write(f"# base_dir={base_dir_resolved}\n")
            f.write(f"# replicate={rep}\n")
            f.write("#\n# minute\tpartitioning\n")
            minutes, part = series[rep]
            for m, p in zip(minutes, part):
                f.write(f"{int(m)}\t{p}\n")
        print(f"  Saved {rep} to {filepath}")
    print(f"Saved partitioning results to {folder}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute daughter-chromosome partitioning for all replicates and save to partitioning/*.txt"
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        nargs="?",
        default=Path("../runs/"),
        help="Base directory containing replicate folders (e.g. Feb08_1/data/coords); default: current dir",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output folder for .txt files (default: base_dir/partitioning)",
    )
    parser.add_argument(
        "-c", "--cutoff",
        type=float,
        default=100.0,
        help="Contact cutoff distance (default: 100.0)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Less progress output (still print per-replicate summary)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of replicates to process in parallel (default: 1). Use 0 for all CPUs.",
    )
    args = parser.parse_args()
    base_dir = args.base_dir.resolve()
    out_dir = args.output if args.output is not None else base_dir / "partitioning"
    out_dir = out_dir.resolve()
    n_jobs = args.jobs

    print(f"Base dir: {base_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Cutoff: {args.cutoff}")
    if n_jobs != 1:
        n_workers = cpu_count() if n_jobs <= 0 else n_jobs
        print(f"Parallel jobs: {n_workers}")
    series = collect_partitioning_time_series(
        base_dir,
        replicates=None,
        cutoff=args.cutoff,
        verbose=not args.quiet,
        out_dir=out_dir,
        n_jobs=n_jobs,
    )
    if not series:
        print("No replicates with dna_* bin files found.")
        return 1
    print(f"Computed partitioning for {len(series)} replicates; results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
