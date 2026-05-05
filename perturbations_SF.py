import matplotlib.pyplot as plt
import numpy as np
from SetUp import *
import yt
from ndustria import Pipeline
from numba import njit

pipe = Pipeline(parallel=True)
RERUN = False

@njit(cache=True)
def accumulate_structure_function(pos, field, pair_i, pair_j, r_edges, counts, sums):
    nbins = counts.shape[0]

    for n in range(pair_i.shape[0]):
        i = pair_i[n]
        j = pair_j[n]

        if i == j:
            continue

        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < r_edges[0] or dist >= r_edges[nbins]:
            continue

        left = 0
        right = nbins
        while left < right:
            mid = (left + right) // 2
            if dist >= r_edges[mid + 1]:
                left = mid + 1
            else:
                right = mid

        bin_index = left
        diff = field[j] - field[i]
        counts[bin_index] += 1
        sums[bin_index] += diff * diff

@pipe.AddFunction(rerun = RERUN)
def extract_sim_data(name, df, z_dir, weight_field=None):

    # Load in data 
    ds = yt.load(name)
    center = get_center(z_dir, df, ds)
    sphere = ds.sphere(center=center, radius=(300, 'kpc'))
    
    # Radius and density data for the profile plot.
    radius_dat = sphere['index','radius'].in_units('kpc').v
    density_dat = sphere['gas','density'].in_units('g/cm**3').v
    pressure_dat = sphere['gas','pressure'].in_units('g/cm/s**2').v
    x = sphere['index', 'x'].in_units('kpc').v
    y = sphere['index', 'y'].in_units('kpc').v
    z = sphere['index', 'z'].in_units('kpc').v
    pos = np.column_stack([x, y, z])  # shape (N,3)

    # Evautate density cutoff for every cell 
    r0 = 50.0  # kpc
    rho0 = 1e-25
    alpha = -4
    rho_cutoff = rho0 * (radius_dat / r0)**alpha
    keep = density_dat < rho_cutoff

    # Define which data points to keep (below the cutoff) and which to cut (above the cutoff).
    # This is so we can keep out large satellite galaxies and focus on the diffuse CGM (our galaxies will be isolated).
    radius_data = radius_dat[keep]
    density_data = density_dat[keep]
    pressure_data = pressure_dat[keep]
    position_data = pos[keep]

    # Decide if we want to weight the histogram by a field (e.g. mass) or not (i.e. all cells count equally).
    if weight_field is None:
        weight_data = np.ones_like(radius_data)
    else:
        weight_data = sphere[weight_field][keep]
        
    return {'Radius': radius_data, 
            'Density': density_data, 
            'Pressure': pressure_data,
            'Position': position_data,
            'Weight': weight_data}

@pipe.AddFunction(rerun = RERUN)
def structure_function(dictionary, nbins=30, n_pairs=50000, rng=np.random.default_rng(42), batch_size=250000):
    """
    Compute the second-order structure function S2(r) = <|f(x+r) - f(x)|^2> for a field f (e.g. density).
    This version samples random point pairs across the full domain instead of only using local nearest neighbors.
    """

    pos = np.asarray(dictionary['Position'])
    field = np.asarray(dictionary['Density'])

    n_points = pos.shape[0]

    n_pairs = int(max(n_pairs, nbins * 200))
    batch_size = int(max(batch_size, 1000))

    span = pos.max(axis=0) - pos.min(axis=0)
    r_max = np.linalg.norm(span)
    V_box = np.prod(span)
    if V_box <= 0:
        V_box = max(r_max**3, 1e-30)
    mean_spacing = (V_box / n_points)**(1.0 / 3.0)
    r_min = max(0.5 * mean_spacing, r_max / 1e5, 1e-12)
    if not (r_max > r_min):
        r_max = r_min * 1.01

    r_edges = np.geomspace(r_min, r_max, nbins + 1)
    counts = np.zeros(nbins, dtype=np.int64)
    sums = np.zeros(nbins, dtype=np.float64)

    processed = 0
    while processed < n_pairs:
        m = min(batch_size, n_pairs - processed)
        pair_i = rng.integers(0, n_points, size=m)
        pair_j = rng.integers(0, n_points, size=m)

        accumulate_structure_function(pos, field, pair_i, pair_j, r_edges, counts, sums)

        processed += m

    with np.errstate(divide='ignore', invalid='ignore'):
        S2 = sums / counts

    r_bins = np.sqrt(r_edges[:-1] * r_edges[1:])
    S2[counts == 0] = np.nan

    return {"r": r_bins, "S2": S2}

@pipe.AddFunction(rerun = RERUN)
def plot_structure_function(all_halos, filename='structure_function.png'):
    """
    Plot the structure function S2(r) vs scale r for a given halo and redshift.
    """

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Scale [kpc]', fontsize=14)
    ax.set_ylabel('S2(r)', fontsize=14)

    for h in all_halos:
        for z in all_halos[h]:
            dic = all_halos[h][z]
            r = np.asarray(dic["r"])
            s2 = np.asarray(dic["S2"])
            valid = np.isfinite(r) & np.isfinite(s2)
            if not np.any(valid):
                continue
            order = np.argsort(r[valid])
            ax.plot(r[valid][order], s2[valid][order] / np.nanmax(s2[valid]), marker='o', linestyle='-', label=f'Halo {h} at {z}')
            if h == "004123" and z == "RD0042":
                with open('structure_function_004123_RD0042.txt', 'w') as f:
                    for r_val, s2_val in zip(r[valid][order], s2[valid][order]):
                        f.write(f"{r_val:.6e} {s2_val:.6e}\n")

    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{filename}')

@pipe.AddFunction(rerun = RERUN)
def list_to_dict(dicts, names):
    """Combine parallel lists of dictionaries and names into a single dictionary.

    Parameters:
        dicts (list[dict]): List of dictionaries to be combined. 
        names (list[str]): List of names/keys to assign to each dictionary in list

    Returns:
        dict: A dictionary mapping each name from ``names`` to the corresponding dictionary from ``dicts``.
    """
    overarching_dict = {}
    for n, d in zip(names, dicts):
        overarching_dict[n] = d
    return dict(overarching_dict)

def main():
    """
    Main function to process simulation data, compute power spectra, and generate plots.
    """
    rng = np.random.default_rng(42)

    # Define simulation dataset info
    target_redshifts = ["RD0042"] # ["RD0016" ,"RD0020", "RD0027", "RD0032", "RD0042"]
    halos = ["002392", "002878", "004123", "005016", "005036", "008508"]

    collect_SFs = []
    for halo_n in halos:
        z_dirs           = get_dirs(halo_n)
        df               = read_halo_c_v(z_dirs, halo_n)

        collect_structure_functions = []
        for redshift in target_redshifts:
            name = f"/mnt/research/turbulence/FOGGIE/halo_{halo_n}/nref11c_nref9f/{redshift}/{redshift}"
            dictionary = extract_sim_data(name, df, redshift, weight_field=None)
            dic_sf = structure_function(dictionary, nbins=100, n_pairs=1e11, rng=rng)
            collect_structure_functions.append(dic_sf)

        all_structure_functions = list_to_dict(collect_structure_functions, target_redshifts)
        collect_SFs.append(all_structure_functions)

    all_SFs = list_to_dict(collect_SFs, halos)      
    plot_structure_function(all_SFs, filename='structure_function_1e11.png')

    pipe.run()

main()