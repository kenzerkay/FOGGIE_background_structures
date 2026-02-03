import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ndustria import Pipeline
from SetUp import get_center, get_dirs, read_halo_c_v
import pickle

# Initialize Common Variables on Import
pipe = Pipeline(parallel=True)
RERUN = False  # Set to True to rerun all steps

@pipe.AddFunction(rerun = RERUN)
def read_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print("Data keys:", data.keys())
    return data

@pipe.AddFunction(rerun = RERUN)
def sample_perturbations(data, grid_size=[200,200,200]):
    """
    Sample a density field from the power spectrum with an underlying density profile.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing l_centers and fft_mag_binned
    box_size : float
        Physical size of the box (in code units)
    grid_size : list of int
        Number of grid cells per dimension
    """
    l_centers = data["l_centers"]
    fft_mag_density = data["fft_mag_binned"]["density"]

    # Create 3D k-space grid
    lx = 1/np.fft.fftfreq(grid_size[0], d=data["box_size"]/grid_size[0]) 
    ly = 1/np.fft.fftfreq(grid_size[1], d=data["box_size"]/grid_size[1]) 
    lz = 1/np.fft.fftfreq(grid_size[2], d=data["box_size"]/grid_size[2]) 
    lx_3d, ly_3d, lz_3d = np.meshgrid(lx, ly, lz, indexing='ij')
    L = np.sqrt(lx_3d**2 + ly_3d**2 + lz_3d**2)

    power_spectrum = np.interp(L.flatten(), l_centers, fft_mag_density)
    power_spectrum = power_spectrum.reshape(L.shape)  

    # Generate random phases (uniform in [0, 2π])
    phases = np.random.uniform(0, 2*np.pi, size=L.shape)
    
    # Create complex Fourier coefficients
    amplitude = np.sqrt(power_spectrum / 2)
    fft_field = amplitude * (np.cos(phases) + 1j * np.sin(phases))
    
    data["fft_field"] = fft_field
    data["grid_size"] = grid_size
    
    return data

@pipe.AddFunction(rerun = RERUN)
def create_filamentary_structure(data, smoothing_scale=20.0):
    """
    Create filamentary structure using Hessian eigenvalues.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing fft_field and grid_size
    smoothing_scale : float
        Smoothing scale in grid cells. Larger values (10-20) create more elongated 
        filaments, smaller values (1-3) create more clumpy structure.
    
    Returns:
    --------
    data : dict
        Dictionary with added perturbations field
    """

    print("Applying Gaussian smoothing...")
    
    # Set up k-space grid
    kx = np.fft.fftfreq(data["grid_size"][0])
    ky = np.fft.fftfreq(data["grid_size"][1])
    kz = np.fft.fftfreq(data["grid_size"][2])
    kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = kx_3d**2 + ky_3d**2 + kz_3d**2

    # Apply Gaussian smoothing in Fourier space (k2 is squared above)
    fft_smoothed = data["fft_field"] * np.exp(-k2 * smoothing_scale**2 / 2) 

    print("Compute Hessian of the density field...")

    # Compute gradients in Fourier space for derivatives
    kx_deriv = 1j * kx_3d # 2j * np.pi * kx_3d
    ky_deriv = 1j * ky_3d # 2j * np.pi * ky_3d
    kz_deriv = 1j * kz_3d # 2j * np.pi * kz_3d
    
    # Compute second derivatives (Hessian components)
    H_xx = np.fft.ifftn(kx_deriv * kx_deriv * fft_smoothed).real
    H_yy = np.fft.ifftn(ky_deriv * ky_deriv * fft_smoothed).real
    H_zz = np.fft.ifftn(kz_deriv * kz_deriv * fft_smoothed).real
    H_xy = np.fft.ifftn(kx_deriv * ky_deriv * fft_smoothed).real
    H_xz = np.fft.ifftn(kx_deriv * kz_deriv * fft_smoothed).real
    H_yz = np.fft.ifftn(ky_deriv * kz_deriv * fft_smoothed).real
    

    print("Computing eigenvalues of Hessian matrix at each grid point...")
    # Diagonalize Hessian to get eigenvalues at each point
    # This allows T-web classification: filaments, sheets, nodes, voids
    
    # Reshape Hessian components for efficient eigenvalue computation
    shape = H_xx.shape
    n_points = np.prod(shape)
    
    # Construct symmetric 3x3 Hessian matrices for all points
    # Shape: (n_points, 3, 3)
    hessian_matrices = np.zeros((n_points, 3, 3))
    hessian_matrices[:, 0, 0] = H_xx.flatten()
    hessian_matrices[:, 1, 1] = H_yy.flatten()
    hessian_matrices[:, 2, 2] = H_zz.flatten()
    hessian_matrices[:, 0, 1] = hessian_matrices[:, 1, 0] = H_xy.flatten()
    hessian_matrices[:, 0, 2] = hessian_matrices[:, 2, 0] = H_xz.flatten()
    hessian_matrices[:, 1, 2] = hessian_matrices[:, 2, 1] = H_yz.flatten()
    
    # Compute eigenvalues for all points (sorted in ascending order)
    eigenvalues = np.linalg.eigvalsh(hessian_matrices)  # Shape: (n_points, 3)
    
    # Reshape back to 3D grid, with eigenvalues sorted: λ₃ ≤ λ₂ ≤ λ₁
    lambda_3 = eigenvalues[:, 0].reshape(shape) 
    lambda_2 = eigenvalues[:, 1].reshape(shape)  
    lambda_1 = eigenvalues[:, 2].reshape(shape)  
    
    # T-web classification based on number of eigenvalues above threshold
    # Threshold = 0 (using sign of eigenvalues for cosmic web morphology)
    # Filament: λ₁ > 0, λ₂ > 0, λ₃ < 0 (collapse in 2 directions, expansion in 1)
    # Sheet: λ₁ > 0, λ₂ < 0, λ₃ < 0 (collapse in 1 direction, expansion in 2)
    # Node: λ₁ > 0, λ₂ > 0, λ₃ > 0 (collapse in all directions)
    # Void: λ₁ < 0, λ₂ < 0, λ₃ < 0 (expansion in all directions)

    print("Determining cosmic web classification for each cell...")
     
    n_positive = (lambda_1 > 0).astype(int) + (lambda_2 > 0).astype(int) + (lambda_3 > 0).astype(int)

    # Create classification field
    is_void = (n_positive == 0)
    is_sheet = (n_positive == 1)
    is_filament = (n_positive == 2)
    is_node = (n_positive == 3)
    
    print(f"Cosmic web classification:")
    print(f"  Voids:     {is_void.sum()/n_points*100:.1f}%")
    print(f"  Sheets:    {is_sheet.sum()/n_points*100:.1f}%")
    print(f"  Filaments: {is_filament.sum()/n_points*100:.1f}%")
    print(f"  Nodes:     {is_node.sum()/n_points*100:.1f}%")

    print("Tune density field to emphasize certain cosmic web structures...")
    
    # Create weighted field that emphasizes filaments
    delta_field_smooth = np.fft.ifftn(fft_smoothed).real
    
    # Weight by structure type: filaments get highest weight (if desired)
    structure_weight = np.zeros(shape)
    structure_weight[is_void] = 1.0     # Suppress voids
    structure_weight[is_sheet] = 1.0     # Moderate sheets
    structure_weight[is_filament] = 1.0  # Emphasize filaments
    structure_weight[is_node] = 1.0      # Moderate nodes

    print("Combining density field with cosmic web structure...")
    
    # Combine density field with structure classification
    web_field = delta_field_smooth * structure_weight
    
    # Store classification and eigenvalues for later analysis
    data["perturbations"] = web_field
    data["web_classification"] = {"void": is_void, "sheet": is_sheet, "filament": is_filament, "node": is_node}
    data["eigenvalues"] = {"lambda_1": lambda_1, "lambda_2": lambda_2, "lambda_3": lambda_3}
    
    return data

@pipe.AddFunction(rerun = RERUN)
def overlay_nfw_profile(data, grid_size=[200,200,200]):

    # Create 3D spatial grid (distance from center)
    # Handle units if box_size is a unyt array
    box_size_value = float(data["box_size"].value) if hasattr(data["box_size"], 'value') else float(data["box_size"])
    
    cell_size = box_size_value / grid_size[0]  # assuming cubic cells
    x = np.arange(grid_size[0]) * cell_size - box_size_value / 2
    y = np.arange(grid_size[1]) * cell_size - box_size_value / 2
    z = np.arange(grid_size[2]) * cell_size - box_size_value / 2
    x_3d, y_3d, z_3d = np.meshgrid(x, y, z, indexing='ij')
    radius_3d = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
    
    # NFW profile: ρ(r) = ρ_s / ((r/r_s)(1 + r/r_s)^2)
    rho_s = 1e-24  # characteristic density (g/cm^3) (set to align vaguley with one of the FOGGIE halos)
    r_s = 8.0  # scale radius (kpc)
    x = radius_3d / r_s
    background_density = rho_s / (x * (1 + x)**2)
    background_density = np.where(radius_3d < cell_size/2, rho_s, background_density) # Handle the singularity at r=0

    # Apply perturbations: ρ = ρ_background * (1 + δ)
    delta_amplitude = 0.3 
    delta_field = data["perturbations"] / data["perturbations"].std() * delta_amplitude
    delta_field = delta_field - delta_field.mean()
    delta_field = np.clip(delta_field, -0.99, None)
    density_field = background_density * (1 + delta_field)
    
    print(f"Background (NFW) mean: {background_density.mean()}")
    print(f"Delta field stats - Min: {delta_field.min()}, Max: {delta_field.max()}, Mean: {delta_field.mean()}")
    print(f"Density field stats - Min: {density_field.min()}, Max: {density_field.max()}, Mean: {density_field.mean()}")
    
    plt.imshow(density_field[:, :, density_field.shape[2]//2], extent=[-box_size_value/2, box_size_value/2, -box_size_value/2, box_size_value/2], norm=LogNorm())
    plt.colorbar(label='Projected Density')
    plt.title('Density Field Projection (z-axis)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('sampled_density_field.png')

    return data

def main():

    picked_data = read_data('radially_averaged_power_spectrum.pkl')
    data = sample_perturbations(picked_data)
    data = create_filamentary_structure(data)
    data = overlay_nfw_profile(data)
    pipe.run()

main()