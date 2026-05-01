import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

def read_bin(path, order="row"):
    data = np.fromfile(path, dtype=np.float64)
    if data.size % 3 != 0:
        raise ValueError(f"{path} file size not divisible by 3")
    N = data.size // 3
    if order == "row":
        coords = data.reshape((N, 3))
    elif order == "col":
        coords = np.column_stack((data[0:N], data[N:2*N], data[2*N:3*N]))
    else:
        raise ValueError("order must be 'row' or 'col'")
    return coords

def radial_prob_density(coords, R, dr_over_R=0.025):
    """Compute normalized radial probability density per unit volume."""
    r = np.linalg.norm(coords, axis=1)
    r_over_R = r / R
    bins = np.arange(0, 1 + dr_over_R, dr_over_R)
    hist, edges = np.histogram(r_over_R, bins=bins)
    bin_volumes = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    prob_density = hist / bin_volumes
    prob_density /= prob_density.sum() * bin_volumes.mean()  # normalize to total probability = 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prob_density

def gather_files(prefix, date, rep_range, tp_range, directory="../runs", replicate_subdirs=False):
    """Collect matching files like dna_Date_rep_tp.bin.

    Supports both old and new naming conventions:
      - Old runs:  {directory}/{date}_{rep}/data/coords/{prefix}_{date}_{rep}_{tp}.bin
      - New runs:  {directory}/{date}_p{rep}/data/coords/{prefix}_{date}_p{rep}_{tp}.bin

    If replicate_subdirs is True, search under per-replicate subdirectories as above.
    Otherwise look for all files directly under directory using both name styles.
    """
    files = []
    base_dir = Path(directory)
    for rep in rep_range:
        rep_str = str(rep)
        if replicate_subdirs:
            # Try new-style and old-style replicate directory names
            subdirs = [
                base_dir / f"{date}_p{rep_str}" / "data" / "coords",
                base_dir / f"{date}_{rep_str}" / "data" / "coords",
            ]
        else:
            # Single directory; filenames may use either style
            subdirs = [base_dir]

        for subdir in subdirs:
            if not subdir.exists():
                continue
            for tp in tp_range:
                tp_str = str(tp)
                # New-style and old-style filenames
                candidate_names = [
                    f"{prefix}_{date}_p{rep_str}_{tp_str}.bin",
                    f"{prefix}_{date}_{rep_str}_{tp_str}.bin",
                ]
                for name in candidate_names:
                    path = subdir / name
                    if path.exists():
                        files.append(path)
    return files

def ensemble_average_density(files, R, dr_over_R=0.025, order="row"):
    densities = []
    for f in files:
        coords = read_bin(f, order=order)
        centers, density = radial_prob_density(coords, R, dr_over_R)
        densities.append(density)
    if not densities:
        raise ValueError(f"No files found for {files}")
    mean_density = np.mean(densities, axis=0)
    return centers, mean_density

def main():
    parser = argparse.ArgumentParser(
        description="Plot radial probability per unit volume of DNA and ribosomes."
    )
    parser.add_argument("--date", required=True, help="Date tag, e.g., Sep28")
    parser.add_argument("--rep_start", type=int, default=1)
    parser.add_argument("--rep_end", type=int, default=3)
    parser.add_argument("--tp_start", type=int, default=1)
    parser.add_argument("--tp_end", type=int, default=50)
    parser.add_argument("--R", type=float, required=True, help="Cell radius")
    parser.add_argument("--dir", default=".", help="Base directory (or dir containing .bin files if not --replicate-subdirs)")
    parser.add_argument("--replicate-subdirs", action="store_true",
                        help="Look in {dir}/{date}_{rep}/data/coords/ for each replicate")
    parser.add_argument("--save", default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    rep_range = range(args.rep_start, args.rep_end + 1)
    tp_range = range(args.tp_start, args.tp_end + 1)

    dna_files = gather_files("dna", args.date, rep_range, tp_range, args.dir, args.replicate_subdirs)
    ribo_files = gather_files("ribo", args.date, rep_range, tp_range, args.dir, args.replicate_subdirs)

    if not dna_files:
        print("Warning: no DNA files found.")
    if not ribo_files:
        print("Warning: no ribosome files found.")

    centers, dna_density = ensemble_average_density(dna_files, args.R, order=args.order)
    _, ribo_density = ensemble_average_density(ribo_files, args.R, order=args.order)

    plt.figure(figsize=(6, 4))
    width = centers[1] - centers[0]

    plt.bar(centers, dna_density, width=width, color="limegreen", alpha=0.5, label="DNA")
    plt.bar(centers, ribo_density, width=width, color="magenta", alpha=0.5, label="Ribosomes")

    plt.xlabel(r"Normalized radius $r/R$")
    plt.ylabel("Probability density per unit volume")
    plt.title(f"Radial distribution – {args.date}")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

