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

def calculate_median_and_mean_profiles(dic):
    """Calculate median and mean density profiles from the histogram data.

    Parameters:
        dic (dict): A dictionary containing the radius, density, weights, and bin information.
    Returns:
        tuple: A tuple containing the median profile, mean profile, and radius bin centers.
    """
    bin_indices = np.digitize(dic['Radius'], dic['Radius_Bins'])
    y_data = np.log10(dic['Density'])
    
    median_profile, mean_profile = [], []
    for i in range(1,len(dic['Radius_Bins'])):
        values = 10**y_data[np.where(bin_indices==i)[0]]
        idx = np.where(bin_indices==i)[0]
        if (len(idx)==0):
            mean_profile.append(np.nan)
            median_profile.append(np.nan)
            continue
        mean_profile.append(np.mean(values))
        median_profile.append(np.median(values))
    radius_bin_centers = 0.5*np.diff(dic['Radius_Bins'])+dic['Radius_Bins'][1:]

    return median_profile, mean_profile, radius_bin_centers

@pipe.AddFunction(rerun = RERUN)
def make_histogram_profile_plots(name, df, fields, halo_n, z_dir, weighted=False):

    # Load in data 
    ds = yt.load(name)
    center = get_center(z_dir, df, ds)
    vir_mass, vir_radius, vir_temp = pull_virial_quantities(ds, halo_n, z_dir)
    sphere = ds.sphere(center=center, radius=(300, 'kpc')) # 5*vir_radius
    
    # Define radius bins for the profile plot.
    radius_range = [5.0, 250.0] # kpc
    radius_bins = np.linspace(radius_range[0], radius_range[1], 200)

    # Radius and density data for the profile plot.
    radius_dat = sphere['index','radius'].in_units('kpc').v
    density_dat = sphere['gas','density'].in_units('g/cm**3').v

    # Evautate density cutoff for every cell 
    rho_cutoff = density_cutoff(radius_dat)
    keep = density_dat < rho_cutoff

    # Define which data points to keep (below the cutoff) and which to cut (above the cutoff).
    # This is so we can keep out large satellite galaxies and focus on the diffuse CGM (our galaxies will be isolated).
    radius_data = radius_dat[keep]
    density_data = density_dat[keep]
    cut_radius_data = radius_dat[keep == False]
    cut_data = density_dat[keep == False]

    if weighted:
        weight_data = sphere['gas','cell_mass'].in_units('Msun').v
        weight_label = 'Mass'
    else:
        # Unweighted profile: every cell contributes equally.
        weight_data = np.ones_like(radius_data)
        weight_label = 'Cell Count'
    cmin = np.min(np.array(weight_data)[np.nonzero(weight_data)[0]])

    dictionary = {'Radius': radius_data, 'Density': density_data, 'Cut_Radius': cut_radius_data, 'Cut_Density': cut_data, 'Weight': weight_data, 'cmin': cmin, 'Radius_Bins': radius_bins, 'Radius_Range': radius_range}
    return dictionary

@pipe.AddFunction(rerun = RERUN)
def plot_histogram_profile(dic, redshift, halo_n):

    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    ax.set_xlabel('$R$ [kpc]', fontsize=20)

    bin_indices = np.digitize(dic['Radius'], dic['Radius_Bins'])
    y_data = np.log10(dic['Density'])
    removed_data = np.log10(dic['Cut_Density'])
    y_bins = np.linspace(-32, -21, 200)

    ax.hist2d(dic['Radius'], y_data, weights=dic['Weight'], bins=(dic['Radius_Bins'], y_bins), cmin=dic['cmin'], cmap=plt.cm.BuPu, norm=mpl.colors.LogNorm())
    ax.hist2d(dic['Cut_Radius'], removed_data, bins=(dic['Radius_Bins'], y_bins), cmap=plt.cm.Greys, norm=mpl.colors.LogNorm())
    
    median_profile, mean_profile, radius_bin_centers = calculate_median_and_mean_profiles(dic)

    ax.plot(radius_bin_centers, np.log10(density_cutoff(radius_bin_centers)), 'r-', lw=2, label='Density Cutoff')
    ax.plot(radius_bin_centers, np.log10(np.array(median_profile)), 'k-', lw=2, label='Median')
    ax.plot(radius_bin_centers, np.log10(np.array(mean_profile)), 'k--', lw=2, label='Mean')
    ax.axis([dic['Radius_Range'][0], dic['Radius_Range'][1], -32, -21])
    ax.legend(loc='upper center',fontsize=16, frameon=False, ncol=2)

    fig.savefig(f"density_profiles_{redshift}_{halo_n}.png") 

    return {"median_profile": median_profile, "mean_profile": mean_profile, "radius_bin_centers": radius_bin_centers}

@pipe.AddFunction(rerun = RERUN)
def collect_profiles_plot(profiles, target_redshifts, halos):
    """Collect median and mean profiles across halos and redshifts for comparison.
    """

    Average_median_profile = np.zeros_like(profiles[halos[0]][target_redshifts[0]]["median_profile"])
    for halo in halos:
        profile = profiles[halo][target_redshifts[0]]
        median_profile = profile["median_profile"]
        Average_median_profile += np.array(median_profile)
    Average_median_profile /= len(halos)


    f = open(f"average_median_profile_{target_redshifts[0]}.txt", 'w')
    f.write('# radius (kpc)  Average Median Density (g/cm**3)\n')
    for i in range(len(profiles[halos[0]][target_redshifts[0]]["median_profile"])):
        f.write('%.3f             %.3e\n' % (profiles[halos[0]][target_redshifts[0]]["radius_bin_centers"][i], Average_median_profile[i]))
    f.close()

    fig, ax = plt.subplots(figsize=(10,8), dpi=300)  
    ax.set_xlim(5, 250)
    ax.set_xlabel('$R$ [kpc]', fontsize=20)
    ax.plot(profiles[halos[0]][target_redshifts[0]]["radius_bin_centers"], profiles[halos[0]][target_redshifts[0]]["median_profile"], color='magenta', alpha=0.7, lw=2, label=f'{halos[0]} Median')
    ax.plot(profiles[halos[1]][target_redshifts[0]]["radius_bin_centers"], profiles[halos[1]][target_redshifts[0]]["median_profile"], color='magenta', alpha=0.7, lw=2, label=f'{halos[1]} Median')
    ax.plot(profiles[halos[2]][target_redshifts[0]]["radius_bin_centers"], profiles[halos[2]][target_redshifts[0]]["median_profile"], color='magenta', alpha=0.7, lw=2, label=f'{halos[2]} Median')
    ax.plot(profiles[halos[3]][target_redshifts[0]]["radius_bin_centers"], profiles[halos[3]][target_redshifts[0]]["median_profile"], color='magenta', alpha=0.7, lw=2, label=f'{halos[3]} Median')
    ax.plot(profiles[halos[4]][target_redshifts[0]]["radius_bin_centers"], profiles[halos[4]][target_redshifts[0]]["median_profile"], color='magenta', alpha=0.7, lw=2, label=f'{halos[4]} Median')
    ax.plot(profiles[halos[5]][target_redshifts[0]]["radius_bin_centers"], profiles[halos[5]][target_redshifts[0]]["median_profile"], color='magenta', alpha=0.7, lw=2, label=f'{halos[5]} Median')    
    ax.plot(profiles[halos[0]][target_redshifts[0]]["radius_bin_centers"], Average_median_profile, color='black', lw=2, label='Average Median')
    fig.savefig(f"average_density_profiles_{target_redshifts[0]}.png")

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
    fields = ['density', 'pressure']
    target_redshifts = ["RD0042"] #, "RD0020", "RD0027", "RD0032", "RD0042"]
    halos = ["002392", "002878", "004123", "005016", "005036", "008508"]
    NUM_BINS = 200

    collect_halos = []
    for halo_n in halos:
        z_dirs           = get_dirs(halo_n)
        df               = read_halo_c_v(z_dirs, halo_n)
        collect_profiles = []
        for redshift in target_redshifts:
            name = f"/mnt/research/turbulence/FOGGIE/halo_{halo_n}/nref11c_nref9f/{redshift}/{redshift}"
            dictionary = make_histogram_profile_plots(name, df, fields, halo_n, redshift, weighted=False)
            profile = plot_histogram_profile(dictionary, redshift, halo_n)
            collect_profiles.append(profile)
        profiles_dict = list_to_dict(collect_profiles, target_redshifts)
        collect_halos.append(profiles_dict)
    All_profiles_dict = list_to_dict(collect_halos, halos)
    collect_profiles_plot(All_profiles_dict, target_redshifts, halos)


    pipe.run()

main()






# unit_dict = {'density':'g/cm**3',
#                 'temperature':'/Tvir',
#                 'metallicity':'Zsun',
#                 'pressure':'erg/cm**3',
#                 'entropy':'keV*cm**2',
#                 'radial_velocity':'km/s'}

# y_range_dict = {'density':[-32,-23],
#                 #'temperature':[0.01,100],
#                 'temperature':[4,7],
#                 'metallicity':[-3,2],
#                 'pressure':[-19,-12],
#                 'entropy':[-5,5],
#                 'radial_velocity':[-500,1000]}

# label_dict = {'density':'log Density [g/cm$^3$]',
#             'temperature':'$T/T_\\mathrm{vir}$',
#             'metallicity':'log Metallicity [$Z_\\odot$]',
#             'pressure':'log Pressure [erg/cm$^3$]',
#             'entropy':'log Entropy [keV cm$^2$]',
#             'radial_velocity':'Radial Velocity [km/s]'}



    # ## Save to File
    # f = open("density_profiles.txt", 'w')
    # f.write('# radius (kpc)  Median %s (%s)  Mean %s (%s)\n' % ("density", "g/cm**3", "density", "g/cm**3"))
    # for i in range(len(median_profile)):
    #     f.write('%.3f             %.3e                   %.3e\n' % (radius_bin_centers[i], median_profile[i], mean_profile[i]))
    # f.close()
