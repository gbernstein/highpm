#!/usr/bin/zsh
#SBATCH --job-name=PM_retry                                              # Job name
#SBATCH --cpus-per-task=32                                               # CPUs per task
#SBATCH --mem=64gb                                                      # Job memory request
#SBATCH --time=2:00:00                                                   # Time limit hrs:min:sec
#SBATCH --output=/home2/vwetzell/ProperMotion_RetII/pm_auxnew00/out_%a.log     # Output log (must exist before submit)
#SBATCH --error=/home2/vwetzell/ProperMotion_RetII/pm_auxnew00/err_%a.err      # Error log (must exist before submit)
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using bhuv_compute
#SBATCH --array=14                                                       # Just the healpixel(s) that timed out
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion_RetII/pm_auxnew00
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/fake_pipeline.py \
    --config /home2/vwetzell/gitrepos/highpm/config/config.yaml \
    --detections "/home2/vwetzell/ProperMotion_RetII/HealpixDetectionCatalog/cleaned_detections_*.fits" \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion_RetII/retii_healpix.npy \
    --runs real,fake \
    --output-name /home2/vwetzell/ProperMotion_RetII/PMCatalog/

finish
