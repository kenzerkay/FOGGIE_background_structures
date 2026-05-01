import matplotlib.pyplot as plt
import numpy as np
from SetUp import *
import yt
import matplotlib as mpl
from ndustria import Pipeline

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

    # Evautate density cutoff for every cell 
    rho_cutoff = density_cutoff(radius_dat)
    keep = density_dat < rho_cutoff

    # Define which data points to keep (below the cutoff) and which to cut (above the cutoff).
    # This is so we can keep out large satellite galaxies and focus on the diffuse CGM (our galaxies will be isolated).
    radius_data = radius_dat[keep]
    density_data = density_dat[keep]
    pressure_data = pressure_dat[keep]

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
            'Cut_Radius': cut_radius_data, 
            'Cut_Density': cut_data,
            'Cut_Pressure': pressure_cut_data,
            'Weight': weight_data}

@pipe.AddFunction(rerun = RERUN)
def calculate_profiles(dic):
    """Calculate median and mean density profiles from the histogram data.

    Parameters:
        dic (dict): Dictionary containing the extracted simulation data.
        redshift (str): Redshift identifier for labeling.
        halo_n (str): Halo identifier for labeling.
    Returns:
        dict: A dictionary containing the median profile, mean profile, and radius bin centers.
    """

    # Define radius bins for the profile plot.
    radius_bins = np.linspace(5.0, 250.0, 200)

    # Bin the radius data and calculate the log10 of the density data for the histogram.
    bin_indices = np.digitize(dic['Radius'], radius_bins)
    density_data = np.log10(dic['Density'])
    pressure_data = np.log10(dic['Pressure'])

    # Calculate the median and mean density profiles for each radius bin.
    median_density = np.full(len(radius_bins) - 1, np.nan, dtype=float)
    mean_density = np.full(len(radius_bins) - 1, np.nan, dtype=float)
    median_pressure = np.full(len(radius_bins) - 1, np.nan, dtype=float)
    mean_pressure = np.full(len(radius_bins) - 1, np.nan, dtype=float)

    for i in range(1, len(radius_bins)):
        mask = (bin_indices == i)
        if not np.any(mask):
            continue
        values = 10.0 ** density_data[mask]
        pressure_values = 10.0 ** pressure_data[mask]
        mean_density[i-1] = np.mean(values)
        median_density[i-1] = np.median(values)
        mean_pressure[i-1] = np.mean(pressure_values)
        median_pressure[i-1] = np.median(pressure_values)

    radius_bin_centers = 0.5*np.diff(radius_bins)+radius_bins[1:]

    return dic | {"Radius_bins": radius_bins, 
                  "bin_indices": bin_indices, 
                  "median_density": median_density, 
                  "mean_density": mean_density, 
                  "median_pressure": median_pressure,
                  "mean_pressure": mean_pressure,
                  "radius_bin_centers": radius_bin_centers}

@pipe.AddFunction(rerun = RERUN)
def plot_histogram_profile(dic):

    fig, axes = plt.subplots(len(dic), 2, figsize=(12, 20), dpi=300)
    for i, halo in enumerate(dic.keys()):
        for redshift in dic[halo].keys():
            
            data = dic[halo][redshift]
            density_bins = np.linspace(-32, -21, 100)
            pressure_bins = np.linspace(-19, -11, 100)

            ax = axes[i, 0]
            ax.set_title(f"Halo {halo} at Redshift {redshift}", fontsize=16)
            ax.hist2d(data['Radius'], np.log10(data['Density']), weights=data['Weight'], bins=(data['Radius_bins'], density_bins), cmin= np.min(np.array(data['Weight'])[np.nonzero(data['Weight'])[0]]), cmap=plt.cm.BuPu, norm=mpl.colors.LogNorm())
            ax.hist2d(data['Cut_Radius'], np.log10(data['Cut_Density']), bins=(data['Radius_bins'], density_bins), cmap=plt.cm.Greys, norm=mpl.colors.LogNorm(), alpha=0.5)
            ax.plot(data['radius_bin_centers'], np.log10(density_cutoff(data['radius_bin_centers'])), 'r-', lw=2, label='Density Cutoff')
            ax.plot(data['radius_bin_centers'], np.log10(np.array(data['median_density'])), 'k-', lw=2, label='Median')
            ax.plot(data['radius_bin_centers'], np.log10(np.array(data['mean_density'])), 'k--', lw=2, label='Mean')
            ax.axis([data['Radius_bins'].min(), data['Radius_bins'].max(), -32, -21])
            ax.legend(loc='upper center',fontsize=16, frameon=False, ncol=2)

            ax = axes[i, 1]
            ax.set_title(f"Halo {halo} at Redshift {redshift}", fontsize=16)
            ax.hist2d(data['Radius'], np.log10(data['Pressure']), weights=data['Weight'], bins=(data['Radius_bins'], pressure_bins), cmin= np.min(np.array(data['Weight'])[np.nonzero(data['Weight'])[0]]), cmap=plt.cm.BuPu, norm=mpl.colors.LogNorm())
            ax.hist2d(data['Cut_Radius'], np.log10(data['Cut_Pressure']), bins=(data['Radius_bins'], pressure_bins), cmap=plt.cm.Greys, norm=mpl.colors.LogNorm(), alpha=0.5)
            ax.plot(data['radius_bin_centers'], np.log10(np.array(data['median_pressure'])), 'k-', lw=2, label='Median')
            ax.plot(data['radius_bin_centers'], np.log10(np.array(data['mean_pressure'])), 'k--', lw=2, label='Mean')
            ax.axis([data['Radius_bins'].min(), data['Radius_bins'].max(), -19, -11])
            ax.legend(loc='upper center',fontsize=16, frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(f"density_and_pressure_profiles.png")

@pipe.AddFunction(rerun = RERUN)
def collect_profiles_plot(dic, red):
    """Collect median and mean profiles across halos and redshifts for comparison.
    """

    # Average the median profiles across all halos for the target redshift.
    average_median_density = np.zeros_like(dic[list(dic.keys())[0]][red]["median_density"])
    average_median_pressure = np.zeros_like(dic[list(dic.keys())[0]][red]["median_pressure"])
    for i, halo in enumerate(dic.keys()):
        profile        = dic[halo][red]
        median_density = profile["median_density"]
        average_median_density += np.array(median_density)
        average_median_pressure += np.array(profile["median_pressure"])
    average_median_density /= len(dic)
    average_median_pressure /= len(dic)

    # Save the average median profile to a text file.
    f = open(f"average_median_profile_{red}.txt", 'w')
    f.write('# radius (kpc)  Average Median Density (g/cm**3)  Average Median Pressure (erg/cm**3)\n')
    for i in range(len(dic[list(dic.keys())[0]][red]["median_density"])):
        f.write('%.3f             %.3e             %.3e\n' % (dic[list(dic.keys())[0]][red]["radius_bin_centers"][i], average_median_density[i], average_median_pressure[i]))
    f.close()
    
    # Plot the median profiles for each halo and the average median profile for the target redshift.
    fig, ax = plt.subplots(2, 1, figsize=(10,8), dpi=300)  
    ax[0].set_xlim(5, 250)
    ax[0].set_xlabel('$R$ [kpc]', fontsize=20)
    for i, halo in enumerate(dic.keys()):
        ax[0].plot(dic[halo][red]["radius_bin_centers"], dic[halo][red]["median_density"], color='red', alpha=0.7, lw=2, label=f'{halo} Median') 
    ax[0].plot(dic[list(dic.keys())[0]][red]["radius_bin_centers"], average_median_density, color='black', lw=2, label='Average Median')

    ax[1].set_xlim(5, 250)
    ax[1].set_xlabel('$R$ [kpc]', fontsize=20)
    for i, halo in enumerate(dic.keys()):
        ax[1].plot(dic[halo][red]["radius_bin_centers"], dic[halo][red]["median_pressure"], color='blue', alpha=0.7, lw=2, label=f'{halo} Mean')
    ax[1].plot(dic[list(dic.keys())[0]][red]["radius_bin_centers"], average_median_pressure, color='black', lw=2, label='Average Median')
    fig.savefig(f"average_density_profiles_{red}.png")

    return 

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
            dictionary = calculate_profiles(dictionary)
            collect_redshifts.append(dictionary)
        redshift_dict = list_to_dict(collect_redshifts, target_redshifts)
        collect_halos.append(redshift_dict)
    all_dictionaries_dict = list_to_dict(collect_halos, halos)
    plot_histogram_profile(all_dictionaries_dict)
    collect_profiles_plot(all_dictionaries_dict, red = "RD0042")

    pipe.run()

main()