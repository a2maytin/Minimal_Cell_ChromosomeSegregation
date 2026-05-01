import numpy as np
def read_bin(path, order="row"):
    data = np.fromfile(path, dtype=np.float64)
    n_coords = data.size
    if n_coords % 3 != 0:
        print(n_coords)
        raise ValueError("size not divisible by 3")
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

def read_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()
    N = int(lines[0].strip())
    coords = []
    for line in lines[2:2+N]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords)


def voxel_contact_map(coords, seg_ids, n_segments, cutoff, box_length=None):
    N = len(coords)
    cell_size = cutoff
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    box_extent = (max_coord - min_coord).max()
    ncell = int(np.ceil(box_extent / cell_size))

    # map beads to voxels
    cell_idx = np.floor((coords - min_coord) / cell_size).astype(int) % ncell
    flat_idx = cell_idx[:,0] + ncell*(cell_idx[:,1] + ncell*cell_idx[:,2])

    cell_particles = [[] for _ in range(ncell**3)]
    for i, c in enumerate(flat_idx):
        cell_particles[c].append(i)

    contacts = np.zeros((n_segments, n_segments), dtype=int)

    # neighbor offsets (27 total)
    offsets = [(dx,dy,dz) for dx in (-1,0,1)
                          for dy in (-1,0,1)
                          for dz in (-1,0,1)]

    for cx in range(ncell):
        for cy in range(ncell):
            for cz in range(ncell):
                cflat = cx + ncell*(cy + ncell*cz)
                beads1 = cell_particles[cflat]
                if not beads1:
                    continue
                for dx,dy,dz in offsets:
                    nx, ny, nz = (cx+dx)%ncell, (cy+dy)%ncell, (cz+dz)%ncell
                    nflat = nx + ncell*(ny + ncell*nz)
                    beads2 = cell_particles[nflat]
                    for i in beads1:
                        for j in beads2:
                            if j <= i:  # avoid double count
                                continue
                            d = coords[i] - coords[j]
                            if box_length is not None:
                                d -= box_length*np.round(d/box_length)
                            if np.dot(d,d) < cutoff**2:
                                si, sj = seg_ids[i], seg_ids[j]
                                #if si != sj:
                                contacts[si, sj] = 1
                                contacts[sj, si] = 1
    return contacts

def super_contact_map(coords, seg_ids, n_segments, cutoff, box_length=None):
    
    rval_full = voxel_contact_map(coords, seg_ids, n_segments, cutoff, box_length=None)
    
    N = 54338
    REPL_MIDPOINT = 27169
    seg_ids = np.arange(N) // segsize
    N_CG = seg_ids.max() + 1
    num_particles = coords.shape[0]
    N_CGf = n_segments

    fork_width = (N_CGf - N_CG) // (2)  # 0 if no replication

    left_fork = N_CG//2 - fork_width
    right_fork = N_CG//2 + fork_width
    print(left_fork,right_fork)
    if left_fork < 0 or right_fork > N_CG:
        raise ValueError("Bad replication fork")
    if left_fork == 0 : right_fork = 2717
    rval_total = np.zeros((N_CG,N_CG), dtype=np.int32)
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
    
    return rval_total


from pathlib import Path
import sys

# Root of this project on the current machine
base_dir = Path("/raid/amaytin/protein_science")

segsize = 100
    
def calculate_contact_matrix(tag, minute):
    """
    Aggregate contact matrices across replicates for a given stationary set.
    For example, tag='Mar10_s1' will combine runs Mar10_s1_r1 ... Mar10_s1_r5.
    """
    # Build run names like Mar10_s1_r1, ..., Mar10_s1_r5
    run_names = [f"{tag}_r{rep}" for rep in range(1, 6)]
    total_reads = 0
    N = 54338
    seg_ids = np.arange(N) // segsize
    N_CG = seg_ids.max() + 1
    contact_sum = np.zeros((N_CG,N_CG), dtype=np.int32)
    
    for run_name in run_names:
        # Directory like /raid/amaytin/protein_science/runs/Mar10_s1_r1/data/coords
        rep_dir = base_dir / "runs" / run_name / "data" / "coords"

        if not rep_dir.exists():
            raise FileNotFoundError(f"No coords folder found for run {run_name} at {rep_dir}")

        # Files like dna_Mar10_s1_r1_3.bin
        bin_files = list(rep_dir.glob(f"dna_{run_name}_{minute}.bin"))
        if not bin_files:
            raise FileNotFoundError(f"No bin files found for run {run_name} and minute {minute} in {rep_dir}")

        total_files = len(bin_files)

        coords = read_bin(bin_files[0])
        N = len(coords)
        seg_ids = np.arange(N) // segsize
        n_segments = seg_ids.max() + 1
        contacts = np.zeros((n_segments, n_segments), dtype=np.int32)

        for i, bin_file in enumerate(bin_files, start=1):
            coords = read_bin(bin_file)
            ival = super_contact_map(coords, seg_ids, n_segments, cutoff=80.0)
            contact_sum += ival
            total_reads += np.count_nonzero(ival)

            if i % 10 == 0 or i == total_files:
                print(f"Run {run_name}, minute {minute}")
                print("Contact matrix shape:", contact_sum.shape)
                print("Thresholded nonzeros:", np.count_nonzero(contact_sum))
                print("Total number of counts (reads):", total_reads)
                print("Min/max/mean:", contact_sum.min(), contact_sum.max(), contact_sum.mean())
                pct = (i / total_files) * 100
                print(str(pct) + "% done")

    # Save aggregated contact map; use tag for the results subdirectory
    results_dir = Path("./results") / tag
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"contact_map_delta_cg{segsize*10}{tag}_{minute}.npy"
    np.save(out_path, contact_sum)
    print(f"Final contact map saved as {out_path}")



def recenter_contact_matrix(contact_matrix, origin=None):
    N = contact_matrix.shape[0]
    if origin is None:
        origin = N // 2

    # roll rows and columns
    rolled = np.roll(contact_matrix, -origin, axis=0)
    rolled = np.roll(rolled, -origin, axis=1)

    return rolled

def normalize_contact_matrix(contact_matrix):
    import numpy as np
    from scipy.sparse import csr_matrix, isspmatrix

    # Try to use krbalancing if available; otherwise fall back to a simple
    # iterative row/column normalization (Sinkhorn-style) so that the script
    # can run without the compiled extension.
    try:
        import krbalancing
        use_kr = True
    except ImportError:
        print("Warning: krbalancing not available, using iterative Sinkhorn-style normalisation instead.")
        use_kr = False

    # Convert dense to CSR if input is dense
    if not isspmatrix(contact_matrix):
        csr = csr_matrix(contact_matrix)
    else:
        csr = contact_matrix

    rows, cols = csr.shape
    if rows != cols:
        raise ValueError("Input matrix must be square.")

    nnz = csr.nnz
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)
    values = csr.data.astype(np.float64)

    if use_kr:
        # Initialize krbalancing
        kb = krbalancing.kr_balancing(rows, cols, nnz, indptr, indices, values)
        kb.computeKR()

        # Get normalized upper-triangular matrix (sparse CSC)
        A_norm_sparse = kb.get_normalised_matrix(True)

        # Convert to dense numpy array (float)
        A_norm = A_norm_sparse.toarray()
    else:
        # Fallback: simple iterative proportional fitting (Sinkhorn)
        # to make rows/columns approximately sum to 1.
        A = csr.astype(np.float64)
        # Avoid division by zero
        eps = 1e-12
        max_iter = 1000
        tol = 1e-6

        # Start with all-ones scalings
        r = np.ones(rows, dtype=np.float64)
        c = np.ones(cols, dtype=np.float64)

        for it in range(max_iter):
            # Row scaling
            row_sums = (A.multiply(c)).sum(axis=1).A.ravel()
            row_sums[row_sums == 0] = 1.0
            r_new = 1.0 / (row_sums + eps)

            # Apply row scaling
            A_rc = A.multiply(r_new[:, None] * c[None, :])

            # Column scaling
            col_sums = A_rc.sum(axis=0).A.ravel()
            col_sums[col_sums == 0] = 1.0
            c_new = 1.0 / (col_sums + eps)

            # Check convergence
            if np.max(np.abs(r_new - r)) < tol and np.max(np.abs(c_new - c)) < tol:
                r, c = r_new, c_new
                break

            r, c = r_new, c_new
            A = csr  # keep original structure; scalings are in r, c

        # Build dense matrix with final scalings applied
        A_norm = (csr.multiply(r[:, None] * c[None, :])).toarray()

    # Make symmetric full matrix
    A_full = A_norm + A_norm.T - np.diag(np.diag(A_norm))

    #hard coded
    print(np.max(A_full))
    A_full = A_full/A_full.sum(axis=1)[0]
    row_sums = A_full.sum(axis=1)
    col_sums = A_full.sum(axis=0)
    print(np.max(A_full))
    print(np.mean(A_full))
    print("Row sums:")
    print(row_sums)

    print("Column sums:")
    print(col_sums)

    # Optional: check if they are close to 1
    print("All row sums ~ 1?", np.allclose(row_sums, 1.0))
    print("All column sums ~ 1?", np.allclose(col_sums, 1.0))



    return A_full




import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
mm = 1/25.4

CG = 100
CG *= 10

def plot_contact_matrix(date):

    contact_matrix = np.zeros_like(np.load("./results/"+date+"/contact_map_delta_cg"+str(CG)+date+"_0.npy"))

    for minute in range(0,7,1):
        print(minute)
        contact_matrix_m = np.load("./results/"+date+"/contact_map_delta_cg"+str(CG)+date+"_"+str(minute)+".npy")
        
        contact_matrix += recenter_contact_matrix(contact_matrix_m)
        
    contact_matrix_normalized = normalize_contact_matrix(contact_matrix)
    np.save('./'+date+'.npy', contact_matrix_normalized)
    fig_size = [65,55]

    fig = plt.figure(figsize=(fig_size[0]*mm,fig_size[1]*mm))

    ax = plt.gca()

    #how to choose vmax? ben default is 0.025
    vmax = .002
    map_norm = colors.Normalize(vmin=0.0, vmax=vmax)
    # map_norm = colors.LogNorm(vmin=np.min(contact_matrix_normalized[contact_matrix_normalized != 0]), vmax=np.max(contact_matrix_normalized))

    # cmap = 'Reds'
    cmap = 'plasma'
    im = ax.imshow(contact_matrix_normalized,cmap=cmap,norm=map_norm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(im, cax=cax)
    
    cbar = fig.colorbar(im, cax=cax, ticks=np.arange(0, vmax + 0.0001, 0.0005))
    cbar.set_label(label=r'Contact Frequency', fontsize=5)
    cbar.ax.tick_params(labelsize=4)

    ax.set_xlabel(r'Genomic Position [kbp]', fontsize=5)
    ax.set_ylabel(r'Genomic Position [kbp]', fontsize=5)

    scale = 1
    formatter = FuncFormatter(lambda x, _: "{:.0f}".format(x*scale))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    plt.tight_layout()

    fig.savefig("./results/"+date+"/contact_map_delta_cg"+str(CG)+date+"_0thru11.png", dpi=300)
    print("Saved contact map figure for given range")
    # plt.show()


#plot Sep28 (cell cycle)
#for i in range(65,91,1):
#calculate_contact_matrix('Sep28',str(i))
#plot_contact_matrix('Oct30')
#plot_contact_matrix('Oct29')
#plot_contact_matrix('Nov04')

# revisions
#for i in range(0,12,1):
#    calculate_contact_matrix('Mar10_s1', str(i))
plot_contact_matrix('Mar10_s1')
