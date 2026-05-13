import yt
import numpy as np
import matplotlib.pyplot as plt
from ndustria import Pipeline
from SetUp import get_center, get_dirs, read_halo_c_v
from numba import njit
import pandas as pd

pipe = Pipeline(parallel=True)
RERUN = False

@pipe.AddFunction(rerun = RERUN)
def extract_sim_data(name, df, z_dir, weight_field=None):

    # Load in data 
    ds = yt.load(name)
    center = get_center(z_dir, df, ds)
    inner_sphere = ds.sphere(center, (15, 'kpc'))
    outer_sphere = ds.sphere(center, (300, 'kpc')) 
    cgm = outer_sphere - inner_sphere 
    
    # Radius and density data for the profile plot.
    density_dat = cgm['gas','density'].in_units('g/cm**3').v
    pressure_dat = cgm['gas','pressure'].in_units('g/cm/s**2').v
    x = cgm['index', 'x'].in_units('kpc').v - center[0].in_units('kpc').v
    y = cgm['index', 'y'].in_units('kpc').v - center[1].in_units('kpc').v
    z = cgm['index', 'z'].in_units('kpc').v - center[2].in_units('kpc').v
    radius_dat = np.sqrt(x**2 + y**2 + z**2)  # shape (N,)
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
        weight_data = cgm[weight_field][keep]
       
    return {'Radius': radius_data,
            'Density': density_data, 
            'Pressure': pressure_data,
            'Position': position_data,
            'Weight': weight_data,
            "Cell_Size": np.min(cgm['index', 'dx'].in_units('kpc'))}

@pipe.AddFunction(rerun = RERUN)
def normalize_density(dictionary):

    # Normalize by profile
    f = pd.read_csv('average_median_profile_RD0042.txt', sep='\\s+', header=None, skiprows=1)
    f.columns = ['Radius', 'Median_Density', "Median_Pressure"]

    # Interpolate the median density profile to the radius values in our data.
    density_interpolated = np.interp(dictionary['Radius'], f['Radius'], f['Median_Density'])
    dictionary['Density'] = dictionary['Density'] / density_interpolated

    # Normalize the density by the mean density in the CGM to get dimensionless fluctuations.
    mean_density = np.mean(dictionary['Density'])
    dictionary['Density'] = (dictionary['Density'] - mean_density) / mean_density

    return dictionary

@njit(cache=True)
def inner_structure_function(pos, density, r_bins, nbins):
    # First-order structure function: mean absolute density difference per separation bin.
    power_spectrum = np.zeros(nbins)
    counts = np.zeros(nbins)

    for i, pos_i in enumerate(pos):
        for j in range(i+1, len(pos)):
            r_ij = np.linalg.norm(pos_i - pos[j])
            if r_ij > 0:
                bin_index = np.searchsorted(r_bins, r_ij) - 1
                if 0 <= bin_index < nbins:
                    power_spectrum[bin_index] += abs(density[i] - density[j])
                    counts[bin_index] += 1

    for i in range(nbins):
        if counts[i] > 0:
            power_spectrum[i] /= counts[i]

    return power_spectrum

@pipe.AddFunction(rerun = True)
def compute_first_order_structure_function(dictionary, nbins=100, keep_fraction=1.0, rng=np.random.default_rng(42)):
    """
    Compute the first-order structure function.
    """
    positions = dictionary['Position']
    density = dictionary['Density']

    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1].")
    if keep_fraction < 1.0:
        n_total = positions.shape[0]
        n_keep = max(2, int(np.ceil(n_total * keep_fraction)))
        idx = rng.choice(n_total, size=n_keep, replace=False)
        positions = positions[idx]
        density = density[idx]

    radii = np.linalg.norm(positions, axis=1)
    rmin = 2 * dictionary["Cell_Size"]
    rmax = 2 * np.max(radii)

    # Define separation bins
    r_bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)  # Adjust as needed
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    power_spectrum = inner_structure_function(positions, density, r_bins, nbins)

    return {'r_centers': r_centers, 'power_spectrum': power_spectrum}

@pipe.AddFunction(rerun = RERUN)
def plot_structure_function(all_halos, save_path='structure_function.png'):
    """
    Plot the structure function from the results.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Separation (kpc)')
    ax.set_ylabel('First-order Structure Function')

    for h in all_halos:
        for z in all_halos[h]:
            dic = all_halos[h][z]
            ax.plot(dic['r_centers'], dic['power_spectrum']/np.nanmax(dic['power_spectrum']), marker='o', label=f"Halo {h} at {z}")
    ax.legend()
    fig.savefig(save_path)


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

    # Define simulation dataset info
    target_redshifts = ["RD0042"] # ["RD0016" ,"RD0020", "RD0027", "RD0032", "RD0042"]
    halos = ["002392", "002878", "004123", "005016", "005036", "008508"]

    collect_halos = []
    for halo_n in halos:
        z_dirs           = get_dirs(halo_n)
        df               = read_halo_c_v(z_dirs, halo_n)

        collect_redshifts = []
        for redshift in target_redshifts:
            name = f"/mnt/research/turbulence/FOGGIE/halo_{halo_n}/nref11c_nref9f/{redshift}/{redshift}"
            dictionary = extract_sim_data(name, df, redshift, weight_field=None)
            dictionary = normalize_density(dictionary)
            dict = compute_first_order_structure_function(dictionary, nbins=40, keep_fraction=0.001)
            collect_redshifts.append(dict)
        collect_halos.append(list_to_dict(collect_redshifts, target_redshifts))
    all_halos = list_to_dict(collect_halos, halos)
    plot_structure_function(all_halos, save_path=f"structure_function.png")

    pipe.run()

main()