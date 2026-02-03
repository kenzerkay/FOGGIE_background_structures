#!/bin/bash --login

#SBATCH --job-name=RerunAllAnalysis                                                                         # Job name
#SBATCH --ntasks=30                                                                                         # Number of tasks   # SLURM defaults to 1 but we specify anyway
#SBATCH --mem=800G                                                                                          # Memory per node   # Specify "M" or "G" for MB and GB respectively
#SBATCH --time=03:59:00                                                                                     # Wall time         # Format: "minutes", "hours:minutes:seconds",      # "days-hours", or "days-hours:minutes"
#SBATCH --output=/mnt/research/galaxies-REU/ticoras/investigate_perturbations/slurm_outputs/%x-%j-SLURM.out # %x: job name, %j: job ID

# Purge current modules and load those we require
module purge 
source activate /mnt/ffs24/home/scottm59/miniforge3/
conda activate IONS_ENV

# Open to directory
directory="/mnt/research/galaxies-REU/ticoras/investigate_perturbations"
cd "${directory}"

mpirun -n 30 python perturbations_in_foggie_sims.py    