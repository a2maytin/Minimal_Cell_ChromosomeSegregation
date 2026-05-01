import os
import sys
import subprocess
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
dna_monomers_dir = os.path.join(base_dir, "../data/coords/")
output_dir = os.path.join(base_dir, "../data/")
template_dir = os.path.join(base_dir, "./")
btree_chromo_executable = "/Software/btree_chromo/build/apps/btree_chromo"

def add_ribosomes_to_bin(bin_path, N, R, L, n, order="row", seed=None):
    """
    Adds N randomly distributed ribosome coordinates to an existing binary file.

    Parameters
    ----------
    bin_path : str or Path
        Path to the existing binary file containing ribosome coordinates.
    N : int
        Number of new ribosome coordinates to add.
    R : float
        Radius of each of the two spheres.
    L : float
        Half the distance between the centers of the two spheres.
    n : numpy vector
        Unit vector for orientation.
    order : {'row', 'col'}, default='row'
        Storage order of coordinates in the binary file.
        'row' means [x1,y1,z1,x2,y2,z2,...].
        'col' means all x first, then y, then z.
    seed : int, optional
        Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Load existing coordinates ---
    data = np.fromfile(bin_path, dtype=np.float64)
    if data.size % 3 != 0:
        raise ValueError("Binary file size is not divisible by 3")
    N_existing = data.size // 3

    if order == "row":
        coords = data.reshape((N_existing, 3))
    elif order == "col":
        coords = np.column_stack((data[0:N_existing],
                                  data[N_existing:2*N_existing],
                                  data[2*N_existing:3*N_existing]))
    else:
        raise ValueError("order must be 'row' or 'col'")

    if N == 0:
        # No new ribosomes to add; write existing coords back and return
        if order == "row":
            coords.astype(np.float64).tofile(bin_path)
        else:
            out = np.concatenate([coords[:, 0], coords[:, 1], coords[:, 2]])
            out.astype(np.float64).tofile(bin_path)
        print(f"No new coordinates added to '{bin_path}' (N=0). Total: {coords.shape[0]} coordinates.")
        return

    # --- Generate new coordinates uniformly inside the union of two spheres ---
    def random_point_in_sphere(radius):
        """Generate a single random 3D point uniformly inside a sphere."""
        while True:
            point = np.random.uniform(-radius, radius, 3)
            if np.linalg.norm(point) <= radius:
                return point

    sphere1_center = -L * n
    sphere2_center = L * n

    new_coords = []
    while len(new_coords) < N:
        # Pick a random sphere to sample from (equal probability)
        center = sphere1_center if np.random.rand() < 0.5 else sphere2_center
        point = random_point_in_sphere(R) + center
        new_coords.append(point)
    new_coords = np.array(new_coords) if new_coords else np.empty((0, 3))

    # --- Combine and write back to binary file ---
    if new_coords.size == 0:
        combined = coords
    else:
        combined = np.vstack([coords, new_coords])

    if order == "row":
        combined.astype(np.float64).tofile(bin_path)
    else:  # col order
        out = np.concatenate([combined[:, 0], combined[:, 1], combined[:, 2]])
        out.astype(np.float64).tofile(bin_path)

    print(f"Appended {N} new coordinates to '{bin_path}'.")
    print(f"Final total: {combined.shape[0]} coordinates.")

def create_directives(run_name, seed, timestep, run_steps=20000, steps_before_output=40000):
    """
    Creates a directives file for btree_chromo by reading a template
    and replacing placeholders based on the timestep.
    run_steps: number of run steps (first arg to simulator_run_soft_harmonic).
    steps_before_output: third arg to simulator_run_soft_harmonic (e.g. 2*run_steps so output doesn't happen).
    """
    sim_prng_seed = seed

    dna_monomers_path = f"{dna_monomers_dir}dna_{run_name}_{timestep}.bin"
    ribos_path = f"{dna_monomers_dir}ribo_{run_name}_{timestep}.bin"
    template_file = template_dir + "template_replicate_delayed.inp"
    fudge_factor = 160
    sphere_radius = 2000 + fudge_factor # 2160
    # First 15 minutes: equilibration with no cell growth
    if timestep >= 15:
        sphere_radius += timestep * 10  # increase by 10 each minute, from 2000 to 2750
    load_boundary = "spherical_bdry:" + str(sphere_radius) + ".0, 0.0, 0.0, 0.0"
    run_dynamics = f"simulator_run_soft_harmonic:{run_steps},1000,{steps_before_output},append,nofirst"
    sphere_height = 0
    n_normalized = np.array([1,0,0])
    if timestep>=75 :
        sphere_radius = 2160 + 600*(timestep - 105)**2/30**2 # from 2760 to 2160 (shifted +15 min)
        sphere_height = 30 + np.sqrt(timestep - 75)*2130/np.sqrt(30)  # from 30 to 2160
        
        # Read in monomer coordinates
        with open(dna_monomers_path,'rb') as f:
            DNAbin = np.fromfile(f,dtype=np.float64,count=-1)
        DNAcoords = DNAbin.reshape((3,DNAbin.shape[0]//3),order='F').T
        L = DNAcoords[:54338]
        R = DNAcoords[54338:]
        # Calculate center of mass of both chromosomes
        LCom = np.average(L, axis=0)
        RCom = np.average(R, axis=0)
        # Calculate vector between centers of mass
        n = LCom - RCom
        # Normalize the vector
        n_normalized = n / np.linalg.norm(n)
        # Convert to string with 3 decimal places
        n_string = ', '.join(f"{x:.3f}" for x in n_normalized)
        load_boundary = "overlapping_spheres_bdry:" + str(sphere_height) + ".0, " + str(sphere_radius) + ".0, 0.0, 0.0, 0.0, " + n_string
        BD_steps = 20000 + (timestep-75)*180000/30  # from 20000 to 200000 (shifted +15 min)
        run_dynamics = "simulator_run_soft_harmonic:"+str(BD_steps)+",1000,"+str(BD_steps*2)+",append,nofirst"

    load_loops = "load_loops:"+f"{output_dir}loops/loops_{run_name}_{timestep}.txt"
    equilibrate_loops = ""
    # No replication during first 15 minutes (equilibration)
    replicate_transform = "" if timestep < 15 else "transform:m_cw20_ccw20"
    # State files are written with next_timestep, so rep_state_{run_name}_{N}.txt
    # represents the state at the END of timestep N-1 (which is the START of timestep N)
    # For restart from timestep N, we load state from timestep N
    input_state = f"input_state:{output_dir}rep_states/rep_state_{run_name}_{timestep}.txt"

    if timestep == 0:
        load_loops = ""
        equilibrate_loops = "translocate:360000,F"
        append_string = "noappend,first"
        input_state = f"input_state:{template_dir}rep_state_initial.txt"
    else:
        # State already set above - rep_state_{timestep}.txt is the state
        # at the end of timestep-1, which is the start of timestep
        append_string = "append,nofirst"

    # No new ribosomes during first 15 minutes (equilibration)
    n_ribosomes_to_add = 0 if timestep < 15 else 5
    add_ribosomes_to_bin(ribos_path, n_ribosomes_to_add, sphere_radius, sphere_height, n_normalized)

    # Read the template file and replace placeholders
    with open(template_file, 'r') as file:
        directives_content = file.read()

    directives_content = directives_content.format(
        sim_prng_seed=sim_prng_seed,
        base_dir=base_dir,
        run_name=run_name,
        input_state=input_state,
        dna_monomers_path=dna_monomers_path,
        ribos_path=ribos_path,
        output_dir=output_dir,
        timestep=timestep,
        next_timestep=timestep+1,
        dna_monomers_dir=dna_monomers_dir,
        load_boundary=load_boundary,
        load_loops=load_loops,
        equilibrate_loops=equilibrate_loops,
        replicate_transform=replicate_transform,
        append_string=append_string,
        run_dynamics=run_dynamics
    )

    # Write to a directives file for this timestep
    directives_filename = f"{output_dir}/btree_chromo_directives_{timestep}_{run_name}.inp"
    with open(directives_filename, 'w') as f:
        f.write(directives_content)

    return directives_filename



def run_btree_chromo(directives_file, env):
    """
    Runs the btree_chromo simulation with a specific directives file.
    """
    try:
        subprocess.run(
            [btree_chromo_executable, directives_file],
            check=True,
            env=env  # Pass the updated environment here
        )
        print(f"Successfully ran btree_chromo with directives: {directives_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running btree_chromo with directives {directives_file}: {e}")
        raise


def main():
    seed = int(sys.argv[1])       # First argument (convert to int)
    run_name = sys.argv[2]              # Second argument (string)
    
    # Optional arguments for restart functionality
    start_time = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end_time = int(sys.argv[4]) if len(sys.argv) > 4 else 90
    is_restart = (sys.argv[5].lower() == 'true') if len(sys.argv) > 5 else False
    run_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 20000
    steps_before_output = int(sys.argv[7]) if len(sys.argv) > 7 else 40000

    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LD_LIBRARY_PATH"] = "/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/Software/LAMMPS/OMP_GPU_Kokkos/lib"

    print(f"Running simulation: {run_name}")
    print(f"Start time: {start_time}, End time: {end_time}, Restart: {is_restart}, Run steps: {run_steps}, Steps before output: {steps_before_output}")

    for step in range(start_time, end_time + 1, 1):
       directives_file = create_directives(run_name, seed, step, run_steps, steps_before_output)
       run_btree_chromo(directives_file, env)
       os.remove(directives_file)


if __name__ == "__main__":
    main()
