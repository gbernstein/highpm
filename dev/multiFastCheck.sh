#!/usr/bin/zsh
#SBATCH --job-name=FastCheck                                             # Job name
#SBATCH --cpus-per-task=32                                               # CPUs per task
#SBATCH --mem=128gb                                                      # Job memory request
#SBATCH --time=2:00:00                                                   # Time limit hrs:min:sec
#SBATCH --output=/home2/vwetzell/ProperMotion/pm_auxnew02/fc_out_%a.log  # Output log (must exist before submit)
#SBATCH --error=/home2/vwetzell/ProperMotion/pm_auxnew02/fc_err_%a.err   # Error log (must exist before submit)
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using bhuv_compute
#SBATCH --array=0-8                                                      # Job array range (= len(PMToDo.npy))
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion/pm_auxnew02
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/fastcheck_pipeline.py \
    --config /home2/vwetzell/gitrepos/highpm/config/config.yaml \
    --detections "/home2/vwetzell/ProperMotion/HealpixDetectionCatalog/cleaned_detections_*.fits" \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion/PMToDo.npy \
    --runs real,fake,overlay \
    --output-name /home2/vwetzell/ProperMotion/PMCatalog/

finish
