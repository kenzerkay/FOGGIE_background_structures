#!/bin/bash --login

#SBATCH --job-name=RerunAllAnalysis                                                                         # Job name
#SBATCH --ntasks=6                                                                                          # Number of tasks   # SLURM defaults to 1 but we specify anyway
#SBATCH --mem=800GB                                                                                         # Memory per node   # Specify "M" or "G" for MB and GB respectively
#SBATCH --time=03:59:00                                                                                     # Wall time         # Format: "minutes", "hours:minutes:seconds",      # "days-hours", or "days-hours:minutes"
#SBATCH --output=/mnt/research/galaxies-REU/ticoras/investigate_perturbations/slurm_outputs/%x-%j-SLURM.out # %x: job name, %j: job ID

# Purge current modules and load those we require
module purge 
source activate /mnt/ffs24/home/scottm59/miniforge3/
conda activate FOGGIE_Background

# Open to directory
directory="/mnt/research/galaxies-REU/ticoras/investigate_perturbations/back"
cd "${directory}"

# mpirun -n 6 python ../FOGGIE_background_structures/smooth_FOGGIE_profiles.py
# mpirun -n 6 python ../FOGGIE_background_structures/pert_NFFT.py # (7607484)at 40% keep_fraction
mpirun -n 6 python ../FOGGIE_background_structures/pert_SF.py # 7607618 at 0.1% keep_fraction
