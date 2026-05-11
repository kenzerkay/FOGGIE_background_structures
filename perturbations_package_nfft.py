import matplotlib.pyplot as plt
import numpy as np
from SetUp import *
import yt
import matplotlib as mpl
from ndustria import Pipeline
from numba import njit
import traceback

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

@pipe.AddFunction(rerun = RERUN)
def non_uniform_fft(dictionary, ks_per_bin=128, nbins=100):
    """
    Compute the power spectrum using a direct 3D nonuniform Fourier transform.
    
    Parameters:
    - pos: (N,3) array of positions
    - w: (N,) array of weights (e.g. density fluctuations)
    - V: total volume of the sampled region
    - ks_per_bin: number of k-modes to sample per bin
    - nbins: number of logarithmic bins in k-space
    
    Returns:
    - k_centers: centers of the k bins
    - Pk: power spectrum values for each k bin
    """
    pos = np.asarray(dictionary['Position'], dtype=np.float64)
    dens = np.asarray(dictionary['Density'], dtype=np.float64)

    # Center the field and keep the full irregular 3D sample list.
    dens = dens - dens.mean()

    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    spans = maxs - mins
    spans[spans == 0] = 1.0
    V = np.prod(spans)

    # k-range from the box size; no gridding of positions is used.
    k_min = 2.0 * np.pi / spans.max()
    k_max = np.pi / spans.min()
    if not np.isfinite(k_min) or not np.isfinite(k_max) or k_min <= 0 or k_max <= k_min:
        return {'Pk': np.zeros(nbins), 'k_centers': np.arange(nbins)}

    k_centers = np.logspace(np.log10(k_min), np.log10(k_max), nbins)
    Pk = np.zeros(nbins, dtype=np.float64)

    rng = np.random.default_rng(42)
    chunk_size = 50000
    n_points = pos.shape[0]

    def sample_unit_vectors(n_vectors):
        vecs = rng.normal(size=(n_vectors, 3))
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0] = 1.0
        return vecs / norms[:, None]

    for i, k_mag in enumerate(k_centers):
        directions = sample_unit_vectors(ks_per_bin)
        k_vectors = k_mag * directions
        power_samples = np.zeros(ks_per_bin, dtype=np.float64)

        for j, k_vec in enumerate(k_vectors):
            amplitude = 0.0j
            for start in range(0, n_points, chunk_size):
                stop = min(start + chunk_size, n_points)
                phase = pos[start:stop] @ k_vec
                amplitude += np.sum(dens[start:stop] * np.exp(-1j * phase))
            power_samples[j] = (np.abs(amplitude) ** 2) / V

        Pk[i] = power_samples.mean()

    return {'Pk': Pk, 'k_centers': k_centers}

@pipe.AddFunction(rerun = RERUN)
def plot_non_uniform_fft(halo, filename='non_uniform_fft.png'):
    print("k_centers:", halo['k_centers'])
    print("Pk:", halo['Pk'])

    # plt.figure(figsize=(8, 6))
    # k = halo['k_centers']
    # Pk = halo['Pk']
    # plt.loglog(k, Pk, label=f"Halo")
    # plt.xlabel(r'$k$ [1/kpc]')
    # plt.ylabel(r'$P(k)$ [units of density^2 * volume]')
    # plt.title('Non-uniform FFT Power Spectrum')
    # plt.legend()
    # plt.grid(True, which='both', ls='--', lw=0.5)
    # plt.tight_layout()
    # plt.savefig(filename)
    # plt.close()

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
            dic = non_uniform_fft(dictionary, ks_per_bin=1, nbins=100)
            plot_non_uniform_fft(dic, filename=f'non_uniform_fft_{halo_n}_{redshift}.png')

    pipe.run()

main()