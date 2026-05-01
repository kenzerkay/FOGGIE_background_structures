import matplotlib.pyplot as plt
import numpy as np
from SetUp import *
import yt
import matplotlib as mpl
from ndustria import Pipeline
from scipy.spatial import cKDTree

pipe = Pipeline(parallel=True)
RERUN = False

def density_cutoff(r_kpc):
    """
    Radial density cutoff in g/cm**3.
    Tune these numbers for your halo.
    """
    r0 = 50.0  # kpc
    rho0 = 1e-25
    alpha = -4
    return rho0 * (r_kpc / r0)**alpha

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
    rho_cutoff = density_cutoff(radius_dat)
    keep = density_dat < rho_cutoff

    # Define which data points to keep (below the cutoff) and which to cut (above the cutoff).
    # This is so we can keep out large satellite galaxies and focus on the diffuse CGM (our galaxies will be isolated).
    radius_data = radius_dat[keep]
    density_data = density_dat[keep]
    pressure_data = pressure_dat[keep]
    position_data = pos[keep]

    cut_radius_data = radius_dat[keep == False]
    cut_data = density_dat[keep == False]
    pressure_cut_data = pressure_dat[keep == False]

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

    if rng is None:
        rng = np.random.default_rng()

    # read arrays
    pos = np.asarray(dictionary['Position'])
    dens = np.asarray(dictionary['Density'])
    weight_field = np.asarray(dictionary.get('Weight', np.ones_like(dens)))

    # optional random downsampling to avoid OOM
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1].")
    if keep_fraction < 1.0:
        n_total = pos.shape[0]
        n_keep = max(2, int(np.ceil(n_total * keep_fraction)))
        idx = rng.choice(n_total, size=n_keep, replace=False)
        pos = pos[idx]
        dens = dens[idx]
        weight_field = weight_field[idx]

    L = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))  # size of the sampled volume
    w = (dens - np.mean(dens)) * (L**3 / len(pos))  # cell volume ~ total volume / N
    kmin = 2.0 * np.pi / L  # more robust than L if volume is irregular
    V = L**3

    # approximate kmax from median nearest-neighbor separation
    tree = cKDTree(pos)
    dists, _ = tree.query(pos, k=2)
    nn = dists[:, 1]
    delta = np.median(nn)
    kmax = np.pi / (delta + 1e-12)
    kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])

    Pk = np.zeros(nbins)

    for i in range(nbins):
        k0 = k_centers[i]
        # sample random directions on the sphere
        u = rng.normal(size=(modes_per_bin, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        ks = (k0 * u).astype(np.float64)  # shape (M,3)

        # compute dot products pos @ ks.T -> shape (Npts, M)
        dots = pos.dot(ks.T)

        # F for each mode: sum_j w_j * exp(-i k·x_j)
        exps = np.exp(-1j * dots)
        F_modes = np.dot(w, exps)  # shape (M,)
        P_modes = np.abs(F_modes)**2

        Pk[i] = P_modes.mean() / V

    return {"k": k_centers, "Pk": Pk}

@pipe.AddFunction(rerun = RERUN)
def plot_power_spectrum(all_halos):
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
            ax.plot(scale_kpc[order], dic["Pk"][order] / np.max(dic["Pk"]), marker='o', linestyle='-', label=f'Halo {h} at {z}')

    ax.legend()
    fig.tight_layout()
    fig.savefig(f'power_spectrum.png')

@pipe.AddFunction(rerun = RERUN)
def structure_function(dictionary, nbins=30, keep_fraction=0.5):
    """
    Compute the second-order structure function S2(r) = <|f(x+r) - f(x)|^2> for a field f (e.g. density).
    This is an alternative to the power spectrum that can be more robust for irregular samples.
    """

    pos = np.asarray(dictionary['Position'])
    field = np.asarray(dictionary['Density'])

    tree = cKDTree(pos)
    dists, idxs = tree.query(pos, k=nbins+1)  # include self (0 distance)
    
    S2 = np.zeros(nbins)
    counts = np.zeros(nbins)

    for i in range(pos.shape[0]):
        for j in range(1, nbins+1):  # skip self
            r = dists[i, j]
            if r > 0:
                dr = field[idxs[i, j]] - field[i]
                S2[j-1] += dr**2
                counts[j-1] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        S2 /= counts
        r_bins = dists[:, 1:nbins+1].mean(axis=0)

    return {"r": r_bins, "S2": S2}

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
    NUM_BINS = 200

    collect_halos = []
    for halo_n in halos:
        z_dirs           = get_dirs(halo_n)
        df               = read_halo_c_v(z_dirs, halo_n)
        collect_redshifts = []
        for redshift in target_redshifts:
            name = f"/mnt/research/turbulence/FOGGIE/halo_{halo_n}/nref11c_nref9f/{redshift}/{redshift}"
            dictionary = extract_sim_data(name, df, redshift, weight_field=None)
            dic = compute_point_power_spectrum(dictionary, nbins=50, modes_per_bin=128, keep_fraction=0.25)
            collect_redshifts.append(dic)
            all_redshifts = list_to_dict(collect_redshifts, target_redshifts)
        collect_halos.append(all_redshifts)
    all_halos = list_to_dict(collect_halos, halos)

    plot_power_spectrum(all_halos)

    pipe.run()

main()