"""
This script processes simulation data from the FOGGIE project to analyze perturbations in the circumgalactic medium (CGM). It performs the following steps:
1. Loads simulation datasets and creates a uniform 3D grid offest from the center as to not indlude the disk (only utilizing CGM)
2. Normalizes fields by their respective profiles and means to isolate fluctuations.
3. Computes the 3D Fourier transform of the normalized fields to obtain power spectra.
4. Radially averages the power spectra to analyze the distribution of perturbations across different scales.
5. Plots the radially averaged power spectra for density and temperature across different halos and redshifts.
6. Saves the processed data and plots for further analysis.
"""

import pickle

import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ndustria import Pipeline
from SetUp import get_center, get_dirs, read_halo_c_v

# Initialize Common Variables on Import
pipe = Pipeline(parallel=True)
RERUN = False  # Set to True to rerun all steps

@pipe.AddFunction(rerun = RERUN)
def pull_data(name, df, halo_n, z_dir, fields, gridsize=[100, 100, 100], left_edge_kpc=[20, 20, 20], right_edge_kpc=[100,100,100]):

    # Load dataset and get center and virial quantities
    print(f"Loading dataset: {name}")
    ds = yt.load(name)
    center = get_center(z_dir, df, ds)

    # Create uniform 3D grid
    print("Creating 3D grid...")
    left_edge = center + left_edge_kpc*ds.units.kpc
    right_edge = center + right_edge_kpc*ds.units.kpc
    grid = ds.arbitrary_grid(left_edge, right_edge, dims=gridsize)

    # Create distance grid - distance from center for each cell
    print("Calculating distance grid...")
    x = np.linspace(left_edge[0].value, right_edge[0].value, gridsize[0])
    y = np.linspace(left_edge[1].value, right_edge[1].value, gridsize[1])
    z = np.linspace(left_edge[2].value, right_edge[2].value, gridsize[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    distance_grid = np.sqrt((xx - center[0].value)**2 + (yy - center[1].value)**2 + (zz - center[2].value)**2) *ds.units.kpc

    # # Calculate average density by profile
    # print("Calculating density and temperature profiles...")
    # # Ensure we use the larger distance for outer sphere
    # outer_radius_kpc = max(abs(np.linalg.norm(left_edge_kpc)), abs(np.linalg.norm(right_edge_kpc)))
    # sph = ds.sphere(center, outer_radius_kpc*ds.units.kpc*2) # Spherical region for profiles
    # sph.set_field_parameter("center", center)  # Explicitly set center for radius calculations
    # profile_plots = yt.ProfilePlot(sph, ("index", "radius"), fields, n_bins=128, weight_field=None, x_log = False)

    # print("Generating projection plot for verification...")
    # tghyt = yt.SlicePlot(ds, 'z', ('gas', 'density'), center=center, width=(outer_radius_kpc*4, 'kpc'), data_source=sph)
    # # Draw box with 4 lines
    # tghyt.annotate_line([left_edge[0]-center[0], left_edge[1] - center[1]], [right_edge[0]-center[0], left_edge[1] - center[1]], coord_system='plot')
    # tghyt.annotate_line([right_edge[0]-center[0], left_edge[1] - center[1]], [right_edge[0]-center[0], right_edge[1] - center[1]], coord_system='plot')
    # tghyt.annotate_line([right_edge[0]-center[0], right_edge[1] - center[1]], [left_edge[0]-center[0], right_edge[1] - center[1]], coord_system='plot')
    # tghyt.annotate_line([left_edge[0]-center[0], right_edge[1] - center[1]], [left_edge[0]-center[0], left_edge[1] - center[1]], coord_system='plot')
    # tghyt.save(f'../images/full_sphere_slice_halo{halo_n}_{z_dir}.png')

    print("Data extraction complete.")
    return {
            "grid": {field: grid["gas", field] for field in fields},
            "box_size": (right_edge[0] - left_edge[0]).to('kpc'),
            "distance_grid": distance_grid.to('kpc'),
            # "profile_plots": {"radius": profile_plots.profiles[0].x.to("kpc"), **{field: profile_plots.profiles[0]["gas", field] for field in fields}}, 
            "cell_depth": ((right_edge[0] - left_edge[0]) / gridsize[0]).to('cm'),
            "grid_size": gridsize
            }

@pipe.AddFunction(rerun = RERUN)
def normalize_fields(dic):

    # Normalize Data Fields
    print("Normalizing data fields...")
    dic["normalized"] = {}
    for key in dic["grid"]:

        # Normalize by profile
        print(f"Normalizing {key} by profile")
        normalized_field = np.zeros_like(dic["grid"][key].flatten())
        for n, (dist, value) in enumerate(zip(dic["distance_grid"].flatten(), dic["grid"][key].flatten())):
            idx = (np.abs(dic["profile_plots"]["radius"].value - dist.value)).argmin() # find closest radius in profile
            rho_profile = dic["profile_plots"][key][idx]
            normalized_field[n] = value / rho_profile
        norm_field = normalized_field.reshape(dic["grid"][key].shape)    # normalize

        # Subtract mean and divide by mean to get fluctuations
        print(f"Normalizing {key} field by mean...")
        field_mean = np.mean(norm_field)
        dic["normalized"][key] = (norm_field - field_mean) / field_mean

    return dic

@pipe.AddFunction(rerun = RERUN)
def compute_power_spectrum(dic):

    # Perform 3D Fourier transform (convert to plain numpy array first)
    print("Computing 3D Fourier transforms...")
    dic['fft_mag'] = {}
    for key in dic["normalized"]:
        print(f"Computing power spectrum for {key}...")
        fft = np.fft.fftn(dic["normalized"][key])
        dic['fft_mag'][key] = np.sqrt(np.abs(fft)**2) # magnitude squared to include both imaginary and real parts)

    # To convert to physical units, divide by box size
    print("Converting FFTs to physical units...")
    kx = np.fft.fftfreq(dic["normalized"][list(dic["normalized"].keys())[0]].shape[0], d=dic["box_size"]/dic["normalized"][list(dic["normalized"].keys())[0]].shape[0])
    ky = np.fft.fftfreq(dic["normalized"][list(dic["normalized"].keys())[0]].shape[1], d=dic["box_size"]/dic["normalized"][list(dic["normalized"].keys())[0]].shape[1])
    kz = np.fft.fftfreq(dic["normalized"][list(dic["normalized"].keys())[0]].shape[2], d=dic["box_size"]/dic["normalized"][list(dic["normalized"].keys())[0]].shape[2])
    lx = 1/kx # DO I NEED THE 2PI FACTOR?  NOOOOO You do not want it
    ly = 1/ky
    lz = 1/kz

    # Distance meshgrid
    lx_3d, ly_3d, lz_3d = np.meshgrid(lx, ly, lz, indexing='ij')
    dic['l_mag'] = np.sqrt(lx_3d**2 + ly_3d**2 + lz_3d**2)

    return dic

@pipe.AddFunction(rerun = RERUN)
def radial_average(dic, num_bins=300):

    print("Performing radial averaging of power spectra...")
    # Flatten arrays
    l_mag_flat = dic["l_mag"].flatten()
    l_mag_flat[np.isinf(l_mag_flat)] = np.nan # Replace infinities with NaNs

    print(np.nanmin(l_mag_flat), np.nanmax(l_mag_flat))
    l_bins = np.linspace(dic["cell_depth"].to('kpc').value*4 , dic["box_size"].to('kpc').value/2, num_bins)  # Define bins
    dic["l_centers"] = 0.5 * (l_bins[:-1] + l_bins[1:])  # Bin centers
    dic['fft_mag_binned'] = {}
    for key in dic['fft_mag']:
        fft_mag_flat = dic['fft_mag'][key].flatten()
        fft_mag_flat[np.isinf(fft_mag_flat)] = np.nan  # Replace infinities with NaNs
        dic["fft_mag_binned"][key] = np.zeros(len(dic["l_centers"]))

        for i in range(len(dic["l_centers"])):
            mask = (l_mag_flat >= l_bins[i]) & (l_mag_flat < l_bins[i+1]) # mask to magnitudes in this bin
            if np.any(mask):
                dic["fft_mag_binned"][key][i] = np.nanmean(fft_mag_flat[mask]) # Get the mean power spectrum for this bin (use nanmean to avoid NaNs)

    return dic

@pipe.AddFunction(rerun = RERUN)
def plot_power_spectrum(dic_list, savefile='radially_averaged_power_spectrum.png'):
    """
    Plot the radially averaged power spectra for density and temperature from the provided dictionary list.

    Parameters:
        dic_list (dict): A dictionary containing the radially averaged power spectra for each halo and redshift. The structure should be:
            {
                halo_n: {
                    redshift: {
                        "l_centers": array of bin centers,
                        "fft_mag_binned": {
                            "density": array of binned power spectrum values for density,
                            "temperature": array of binned power spectrum values for temperature
                        }
                    },
                    ...
                },
                ...            }
        savefile (str): The filename where the plot will be saved. Default is '
    """
    print("Plotting power spectra...")

    fig, ax = plt.subplots(len(dic_list), 2, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle('Density (left) and Temperature (right) Power Spectra', fontsize=18)
    fig.supxlabel('Distance Scale (kpc)', fontsize=16)
    fig.supylabel('Normalized Magnitude', fontsize=16)

    for n, halo_n in enumerate(dic_list):
        dic_halo = dic_list[halo_n]
        for redshift in dic_halo:
            dic = dic_halo[redshift]

            for i, key in enumerate(dic["fft_mag_binned"]):
                ax[n, i].set_title(f'Halo {halo_n}', fontsize=16)
                data = dic["fft_mag_binned"][key]
                print(data)
                # Normalize each spectrum by its own maximum
                normalized_data = data / np.max(data)
                ax[n, i].plot(dic["l_centers"], normalized_data, label=f'(z={redshift})')
                ax[n, i].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(savefile, dpi=150)

@pipe.AddFunction(rerun = RERUN)
def list_to_dict(dicts, names):
    """Combine parallel lists of dictionaries and names into a single dictionary.

    Parameters:
        dicts (list[dict]): List of dictionaries to be combined. 
        names (list[str]): List of names/keys to assign to e ach dictionary in list

    Returns:
        dict: A dictionary mapping each name from ``names`` to the corresponding dictionary from ``dicts``.
    """
    overarching_dict = {}
    for n, d in zip(names, dicts):
        overarching_dict[n] = d
    return dict(overarching_dict)

@pipe.AddFunction(rerun = RERUN)
def pickle_data(dic, filename):
    """
    Pickle a dictionary and save it to a file.

    Parameters:
        dic (dict): The dictionary to be pickled and saved to a file.
        filename (str): The name of the file where the pickled dictionary will be saved
    """
    with open(filename, 'wb') as f:
        pickle.dump(dic, f)
    return

def main():
    """
    Main function to process simulation data, compute power spectra, and generate plots.
    """
    # Define simulation dataset info
    fields = ['density']
    target_redshifts = ["RD0016", "RD0020", "RD0027", "RD0032", "RD0042"]
    halos = ["002392", "002878", "004123", "005016", "005036", "008508"]
    NUM_BINS = 200

    collect_halos = []
    for halo_n in halos:
        z_dirs           = get_dirs(halo_n)
        df               = read_halo_c_v(z_dirs, halo_n)
        collect_redshifts = []
        for redshift in target_redshifts:
            name = f"/mnt/research/turbulence/FOGGIE/halo_{halo_n}/nref11c_nref9f/{redshift}/{redshift}"
            dic = pull_data(name, df, halo_n, redshift, fields, gridsize = [75, 75, 75], left_edge_kpc=[-100, -100, 20], right_edge_kpc=[100, 100, 220])
            dic2 = normalize_fields(dic)
            dic3 = compute_power_spectrum(dic2)
            dic4 = radial_average(dic3, num_bins=NUM_BINS)
            collect_redshifts.append(dic4)
        all_redshifts = list_to_dict(collect_redshifts, target_redshifts)
        collect_halos.append(all_redshifts)
    all_halos = list_to_dict(collect_halos, halos)

    pickle_data(all_halos, 'radially_averaged_power_spectrum.pkl')
    plot_power_spectrum(all_halos, savefile='radially_averaged_power_spectrum_total.png')
    pipe.run()

main()
