import numpy as np
from pathlib import Path
import sys
tag='Aug14'
reps = {
    tag: [2,3,4,5,6,7,8,9,10],
}

# === Settings ===
base_dir = Path("/projects/bdxm/amaytin/")  # adjust for server

METHOD = 1
N = 54338
REPL_MIDPOINT = 27169
CG = 20
R0 = 68
DELTA = 68
IVAL_CUTOFF = 0.5
OUT_RVAL_NPY = "contact_rval.npy"
OUT_IVAL_NPY = "contact_ival.npy"
OUT_RVAL_TXT = None
BIN_ORDER = "row"

# === Binary reader ===
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

def build_map(N_monomers, CG):
    """
    Build mapping arrays (1-based for documentation; returns Python 0-based slices list).
    Returns:
      map_slices: list of (start_idx, end_idx) inclusive, 0-based indices
      N_CG: number of CG loci
    Each locus covers CG monomers except possibly the last which may be shorter.
    """
    if CG <= 0:
        raise ValueError("CG must be >= 1")
    N_CG = int(np.ceil(N_monomers / CG))
    map_slices = []
    for i in range(N_CG):
        start = i * CG
        end = min((i + 1) * CG - 1, N_monomers - 1)
        map_slices.append((start, end))
    return map_slices, N_CG

def build_pairs(N_CG):
    """
    Build pair list including self pairs, matching Fortran's creation:
    pairs for i in 1..N, j in i..N (inclusive). Returns (N_pair, 2) array of ints (0-based).
    """
    pairs = []
    for i in range(N_CG):
        for j in range(i, N_CG):
            pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int32)

def center_of_mass(coords):
    """
    coords: (k, 3) array of coordinates for some subset
    returns (3,) COM
    """
    return coords.mean(axis=0)

def tanh_contact_from_distance(d, r0=R0, delta=DELTA):
    """
    Soft contact function mapping distance -> [0,1].
    Uses a shifted/scaled tanh so the transition occurs near r0, width ~ delta.
    f(d) = 0.5 * (1 - tanh( (d - r0) / delta ))
    This gives f ~ 1 for d << r0 and f ~ 0 for d >> r0.
    Vectorized for numpy arrays.
    """
    z = (d - r0) / delta
    return 0.5 * (1.0 - np.tanh(z))

def compute_contacts(coords, map_slices, method=METHOD, r0=R0, delta=DELTA):

    N_CG = len(map_slices)
    # Unreplicated monomer count (last CG bin's end + 1)

    N = map_slices[-1][1] + 1
    num_particles = coords.shape[0]

    map_slicesf, N_CGf = build_map(num_particles, CG)

    fork_width = (N_CGf - N_CG) // (2)  # 0 if no replication

    def _compute_for(coords_local):
        if method == 1:
            # COMs
            coms_local = np.empty((N_CGf, 3), dtype=np.float64)
            for i, (s, e) in enumerate(map_slicesf):
                coms_local[i] = center_of_mass(coords_local[s:e+1])
            diff = coms_local[:, None, :] - coms_local[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            rval_local = tanh_contact_from_distance(dists, r0=r0, delta=delta)
            return rval_local, coms_local
        elif method == 2:
            rval_local = np.zeros((N_CG, N_CG), dtype=np.float64)
            for idx_i, (s_i, e_i) in enumerate(map_slices):
                Xi = coords_local[s_i:e_i+1]
                ni = Xi.shape[0]
                for idx_j, (s_j, e_j) in enumerate(map_slices[idx_i:], start=idx_i):
                    Xj = coords_local[s_j:e_j+1]
                    dij = np.linalg.norm(Xi[:, None, :] - Xj[None, :, :], axis=2)
                    fij = tanh_contact_from_distance(dij, r0=r0, delta=delta)
                    total = fij.sum() / (ni * Xj.shape[0])
                    rval_local[idx_i, idx_j] = total
                    rval_local[idx_j, idx_i] = total
            return rval_local, None
        else:
            raise ValueError(f"Unknown method {method}")

    # --- No replication: behave like original
    #if fork_width <= 0:
    #    rval, coms = _compute_for(coords)
    #    ival = (rval >= IVAL_CUTOFF).astype(np.int32)
    #    return rval, ival

    # --- Replication present
    left_fork = N_CG//2 - fork_width
    right_fork = N_CG//2 + fork_width
    print(left_fork,right_fork)
    if left_fork < 0 or right_fork > N_CG:
        raise ValueError(
            f"Replication forks out of bounds: left={left_fork}, right={right_fork}, N={N}"
        )
    if left_fork == 0 : right_fork = 2717

    rval_full, _ = _compute_for(coords)

    rval_total = np.zeros((N_CG,N_CG), dtype=np.float64)
    #   o o o o
    #   # # # o
    #   # # # o
    #   # # # o
    rval_total += rval_full[:N_CG,:N_CG]

    #   # # # o
    #   o o o o
    #   0 0 0 o
    #   o o o o
    rval_total[left_fork:right_fork, :] += rval_full[N_CG:N_CGf, :N_CG]

    #   o o o o
    #   o 0 o #
    #   o 0 o #
    #   o 0 o #
    rval_total[:, left_fork:right_fork] += rval_full[:N_CG, N_CG:N_CGf]

    #   o o o #
    #   o o o o
    #   o 0 o o
    #   o o o o
    rval_total[left_fork:right_fork, left_fork:right_fork] += rval_full[N_CG:N_CGf, N_CG:N_CGf]

    np.maximum(rval_total, 0.0, out=rval_total)
    ival_total = (rval_total >= IVAL_CUTOFF).astype(np.int32)

    return rval_total, ival_total

# === Main loop ===
map_slices, N_CG = build_map(N, CG)
print(f"Using CG={CG} -> {N_CG} loci")
pairs = build_pairs(N_CG)
print(f"Built {pairs.shape[0]} pairs (including self-pairs)")
contact_sum = np.zeros((N_CG, N_CG), dtype=np.int32)
total_reads = 0
N_per_replicate = 30
for date, rep_list in reps.items():
    for rep in rep_list:
        # First check if the grouped folder exists
        grouped_path = base_dir / date / f"{date}_{rep}" / "data" / "coords"
        standalone_path = base_dir / f"{date}_{rep}" / "data" / "coords"

        if grouped_path.exists():
            rep_dir = grouped_path
        elif standalone_path.exists():
            rep_dir = standalone_path
        else:
            raise FileNotFoundError(f"No DNA folder found for {date}_{rep}")

        # Gather bin files
        raw_bin_files = list(rep_dir.glob(f"dna_{date}_{rep}_*.bin"))

        # Warn about skipped files
        #for p in raw_bin_files:
        #    frame_str = p.stem.split("_")[-1]
        #    if not frame_str.isdigit():
        #        print(f"  [Warning] Skipping non-frame file: {p.name}")

        # Keep only files with numeric frame indices
        bin_files = [p for p in raw_bin_files if p.stem.split("_")[-1].isdigit()]

        # Sort numerically by frame number
        bin_files.sort(key=lambda p: int(p.stem.split("_")[-1]))

        # Downsample to 30 evenly spaced frames
        if len(bin_files) > N_per_replicate:
            indices = np.linspace(0, len(bin_files) - 1, N_per_replicate, dtype=int)
            bin_files = [bin_files[i] for i in indices]

        total_files = len(bin_files)
        #print(f"  Using {total_files} files for {date}_{rep}")



        #print(f"Processing {date}_{rep} ({total_files} files)...")

        for i, bin_file in enumerate(bin_files, start=1):
            coords = read_bin(bin_file)
            rval, ival = compute_contacts(coords, map_slices)
            contact_sum += ival
            total_reads += np.count_nonzero(ival)
            print(i)
            if i % 10 == 0 or i == total_files:
                print("Contact matrix shape:", contact_sum.shape)
                print("Thresholded nonzeros:", np.count_nonzero(contact_sum))
                print("Total number of counts (reads):", total_reads)
                print("Min/max/mean:", contact_sum.min(), contact_sum.max(), contact_sum.mean())
                pct = (i / total_files) * 100
                print(f"  {i}/{total_files} ({pct:.1f}%) done for {date}_{rep}")

# Save aggregated contact map
np.save("contact_map_delta_cg"+str(CG*10)+tag+".npy", contact_sum)
print("Final contact map saved as contact_map_delta_cg"+str(CG*10)+tag+".npy")
