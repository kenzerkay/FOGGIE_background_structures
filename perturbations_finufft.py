import matplotlib.pyplot as plt
import numpy as np
from SetUp import *
import yt
from ndustria import Pipeline
import finufft

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
    """
    positions = dictionary['Position']
    density = dictionary['Density']

    density = density - np.mean(density)  # Subtract mean to focus on fluctuations

    # Scale positions to [-pi, pi)
    def scale_to_pi(a):
        return 2*np.pi*(a - a.min())/(a.max() - a.min()) - np.pi

    xs = scale_to_pi(positions[:,0])
    ys = scale_to_pi(positions[:,1])
    zs = scale_to_pi(positions[:,2])

    # Define Fourier grid size
    nx, ny, nz = ks_per_bin * nbins, ks_per_bin * nbins, ks_per_bin * nbins

    # Compute NUFFT
    fk = finufft.nufft3d1(xs, ys, zs, density.astype(np.complex128), (nx, ny, nz), eps=1e-6)

    # Compute power spectrum
    power = np.abs(fk)**2

    # Build k-grid
    Lx = positions[:,0].max() - positions[:,0].min()
    Ly = positions[:,1].max() - positions[:,1].min()
    Lz = positions[:,2].max() - positions[:,2].min()

    kx = 2*np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky = 2*np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    kz = 2*np.pi * np.fft.fftfreq(nz, d=Lz/nz)

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kz = np.fft.fftshift(kz)

    power = np.fft.fftshift(power)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Radial binning
    k_flat = kmag.ravel()
    p_flat = power.ravel()

    k_bins = np.linspace(0, k_flat.max(), nbins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    Pk = np.zeros(nbins)

    for i in range(nbins):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        if np.any(mask):
            Pk[i] = p_flat[mask].mean()
        else:
            Pk[i] = np.nan

    return {'Pk': Pk, 'k_centers': k_centers}

@pipe.AddFunction(rerun = RERUN)
def plot_non_uniform_fft(all_halos, filename='non_uniform_fft.png'):

    fig,ax = plt.subplots(figsize=(8,6))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('k [1/kpc]', fontsize=14)
    ax.set_ylabel('P(k) [units of density^2 * volume]', fontsize=14)

    for h in all_halos:
        for z in all_halos[h]:
            dic = all_halos[h][z]
            k = dic['k_centers']
            dist = 1/k
            Pk = dic['Pk']
            ax.plot(dist, Pk/np.nanmax(Pk), label=f"Halo {h} at {z}")

    fig.savefig(filename)


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
            dic = non_uniform_fft(dictionary, ks_per_bin=1, nbins=100)
            
            collect_redshifts.append(dic)
        all_redshifts = list_to_dict(collect_redshifts, target_redshifts)
        collect_halos.append(all_redshifts)
    all_halos = list_to_dict(collect_halos, halos)
    plot_non_uniform_fft(all_halos, filename=f'non_uniform_fft.png')


    pipe.run()

main()