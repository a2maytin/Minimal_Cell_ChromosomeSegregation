import numpy as np
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
    """
    Compute segment–segment contact map using voxel neighbor search.

    coords : (N,3) bead coordinates
    seg_ids : (N,) array, segment index for each bead
    n_segments : int, total number of segments
    cutoff : float, contact distance
    box_length : optional float, for cubic periodic box
    """
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
        raise ValueError(
            f"Replication forks out of bounds: left={left_fork}, right={right_fork}, N={N}"
        )
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
base_dir = Path("/projects/bdxm/amaytin/")  # adjust for server
# tag='Sep28'
segsize = 100
    
def calculate_contact_matrix(tag, minute):
    reps = {
        tag: [1,2,3,4,5,6,7,8,9,10],
    }
    total_reads = 0
    N = 54338
    seg_ids = np.arange(N) // segsize
    N_CG = seg_ids.max() + 1
    contact_sum = np.zeros((N_CG,N_CG), dtype=np.int32)
    # minute=90

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
            bin_files = list(rep_dir.glob(f"dna_{date}_{rep}_{minute}.bin"))
            total_files = len(bin_files)

            coords = read_bin(bin_files[0])
            N = len(coords)
            seg_ids = np.arange(N) // segsize
            n_segments = seg_ids.max() + 1
            contacts = np.zeros((n_segments, n_segments), dtype=np.int32)

            for i, bin_file in enumerate(bin_files, start=1):
                coords = read_bin(bin_file)
                ival = super_contact_map(coords, seg_ids, n_segments, cutoff=60.0)
                contact_sum += ival
                total_reads += np.count_nonzero(ival)

                if i % 10 == 0 or i == total_files:
                    print("Contact matrix shape:", contact_sum.shape)
                    print("Thresholded nonzeros:", np.count_nonzero(contact_sum))
                    print("Total number of counts (reads):", total_reads)
                    print("Min/max/mean:", contact_sum.min(), contact_sum.max(), contact_sum.mean())
                    pct = (i / total_files) * 100
                    print(f"  {i}/{total_files} ({pct:.1f}%) done for {date}_{rep}")

    # Save aggregated contact map
    np.save("./results/"+date+"/contact_map_delta_cg"+str(segsize*10)+tag+"_"+minute+".npy", contact_sum)
    print("Final contact map saved as contact_map_delta_cg"+str(segsize*10)+tag+"_"+minute+".npy")



def recenter_contact_matrix(contact_matrix, origin=None):
    N = contact_matrix.shape[0]
    if origin is None:
        origin = N // 2

    # roll rows and columns
    rolled = np.roll(contact_matrix, -origin, axis=0)
    rolled = np.roll(rolled, -origin, axis=1)

    return rolled


import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
mm = 1/25.4

CG = 100
CG *= 10

def plot_contact_matrix(date,minute):
    contact_matrix = np.load("./results/"+date+"/contact_map_delta_cg"+str(CG)+date+"_"+minute+".npy")
    contact_matrix = recenter_contact_matrix(contact_matrix)

    fig_size = [65,55]

    fig = plt.figure(figsize=(fig_size[0]*mm,fig_size[1]*mm))

    ax = plt.gca()

    #how to choose vmax? ben default is 0.025
    vmax = 0.1
    map_norm = colors.Normalize(vmin=0.0, vmax=vmax)

    cmap = 'Reds'
    #cmap = 'plasma'
    im = ax.imshow(contact_matrix,cmap=cmap,norm=map_norm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label=r'Contact Frequency', fontsize=5)
    cbar.ax.tick_params(labelsize=4)

    ax.set_xlabel(r'Genomic Position [kbp]', fontsize=5)
    ax.set_ylabel(r'Genomic Position [kbp]', fontsize=5)

    scale = 1
    formatter = FuncFormatter(lambda x, _: f"{x*scale:.0f}")

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    plt.tight_layout()

    fig.savefig("./contact_map_delta_cg"+str(CG)+date+"_"+minute+".png", dpi=300)
    print("Saved contact map figure")
    # plt.show()


#plot Sep28 (cell cycle)
for i in range(0,61,1):
    calculate_contact_matrix('Nov21',str(i))
    plot_contact_matrix('Nov21',str(i)) #noblocking
    #calculate_contact_matrix('Oct29',str(i))
    #plot_contact_matrix('Oct29',str(i)) #shortdwell
    #calculate_contact_matrix('Nov04',str(i))
    #plot_contact_matrix('Nov04',str(i)) #blocking
