import matplotlib.pyplot as plt
import numpy as np
from SetUp import *
import yt
import matplotlib as mpl
from ndustria import Pipeline
from numba import njit

pipe = Pipeline(parallel=True)
RERUN = False

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

@njit(cache=True)
def inner_power_spectrum(pos, w, V, ks_per_bin, nbins):

    Pk = np.zeros(nbins)

    for i in range(nbins):
        ks = ks_per_bin[i]
        modes = ks.shape[0]
        mode_power_sum = 0.0

        for m in range(modes):
            kx = ks[m, 0]
            ky = ks[m, 1]
            kz = ks[m, 2]

            real_sum = 0.0
            imag_sum = 0.0

            for j in range(pos.shape[0]):
                phase = kx * pos[j, 0] + ky * pos[j, 1] + kz * pos[j, 2]
                c = np.cos(phase)
                s = np.sin(phase)
                real_sum += w[j] * c
                imag_sum -= w[j] * s

            mode_power_sum += real_sum * real_sum + imag_sum * imag_sum

        Pk[i] = (mode_power_sum / modes) / V

    return Pk

@pipe.AddFunction(rerun = RERUN)
def compute_point_power_spectrum(dictionary, nbins=30, modes_per_bin=64, rng=None, keep_fraction=0.5):
    """
    Estimate isotropic P(k) directly from irregular point samples (cell centers)
    without resampling onto a grid.

    Method: compute F(k) = sum_j w_j exp(-i k.dot(x_j)) for many random k vectors
    with |k| in each bin, average |F|^2 per bin and return P(k).

    Notes:
    - Positions are in kpc, k in 1/kpc.
    - We use weight w_j = (rho_j - <rho>) * cell_volume_j so the zero mode is removed.
    - Normalization here is P(k) ~ <|F|^2> / V (V = total sampled volume). Adjust as needed.
    - This is O(N_cells * modes) and can be expensive for very large samples.
    """

    # read arrays
    pos = np.asarray(dictionary['Position'])
    dens = np.asarray(dictionary['Density'])
    weight_field = np.asarray(dictionary.get('Weight', np.ones_like(dens)))

    # optional random downsampling to avoid OOM
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1].")
    if keep_fraction < 1.0:
        n_total = pos.shape[0]
        print(n_total, "cells before downsampling.")
        n_keep = max(2, int(np.ceil(n_total * keep_fraction)))
        idx = rng.choice(n_total, size=n_keep, replace=False)
        pos = pos[idx]
        dens = dens[idx]
        weight_field = weight_field[idx]

    span = pos.max(axis=0) - pos.min(axis=0)
    L = np.linalg.norm(span)  # characteristic size of sampled domain
    w = (dens - np.mean(dens)) * (L**3 / len(pos))  # cell volume ~ total volume / N
    kmin = 2.0 * np.pi / L  # more robust than L if volume is irregular
    V = L**3

    # approximate kmax from mean inter-point spacing (no nearest-neighbor tree)
    V_box = np.prod(span)
    if V_box <= 0:
        V_box = V
    delta = (V_box / len(pos))**(1.0 / 3.0)
    kmax = np.pi / (delta + 1e-12)
    kmax = max(kmax, 1.2 * kmin)
    kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])

    ks_per_bin = np.empty((nbins, modes_per_bin, 3), dtype=np.float64)
    for i in range(nbins):
        u = rng.normal(size=(modes_per_bin, 3))
        norms = np.sqrt(np.sum(u * u, axis=1))
        u /= norms[:, None]
        ks_per_bin[i] = k_centers[i] * u

    Pk = inner_power_spectrum(pos, w, V, ks_per_bin, nbins)

    return {"k": k_centers, "Pk": Pk}

@pipe.AddFunction(rerun = RERUN)
def plot_power_spectrum(all_halos, filename='power_spectrum.png'):
    """
    Plot the power spectrum P(k) vs physical scale in kpc for a given halo and redshift.
    """

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Scale [kpc] = 2π/k', fontsize=14)
    ax.set_ylabel('P(k)', fontsize=14)

    for h in all_halos:
        for z in all_halos[h]:
            dic = all_halos[h][z]
            scale_kpc = 2.0 * np.pi / np.asarray(dic["k"])
            order = np.argsort(scale_kpc)
            ax.plot(scale_kpc[order], dic["Pk"][order] / np.max(dic["Pk"]), marker='o', linestyle='-', label=f'Halo {h}')
            if h == "004123" and z == "RD0042":
                with open('power_spectrum_004123_RD0042.txt', 'w') as f:
                    for k_val, pk_val in zip(dic["k"][order], dic["Pk"][order]):
                        f.write(f"{k_val:.6e} {pk_val:.6e}\n")
                

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

    collect_halos = []
    for halo_n in halos:
        z_dirs           = get_dirs(halo_n)
        df               = read_halo_c_v(z_dirs, halo_n)

        collect_redshifts = []
        for redshift in target_redshifts:
            name = f"/mnt/research/turbulence/FOGGIE/halo_{halo_n}/nref11c_nref9f/{redshift}/{redshift}"
            dictionary = extract_sim_data(name, df, redshift, weight_field=None)
            dic = compute_point_power_spectrum(dictionary, nbins=100, rng=rng, modes_per_bin=128, keep_fraction=0.2)
            collect_redshifts.append(dic)
        
        all_redshifts = list_to_dict(collect_redshifts, target_redshifts)
        collect_halos.append(all_redshifts)

    all_halos = list_to_dict(collect_halos, halos)
    plot_power_spectrum(all_halos, filename='power_spectrum_1.png')

    pipe.run()

main()