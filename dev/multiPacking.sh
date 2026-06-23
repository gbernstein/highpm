#!/usr/bin/zsh
#SBATCH --job-name=Packing                                               # Job name
#SBATCH --cpus-per-task=4                                                # CPUs per task
#SBATCH --mem=32gb                                                       # Job memory request
#SBATCH --time=48:00:00                                                  # Time limit hrs:min:sec
#SBATCH --output=/home2/vwetzell/ProperMotion/packing_auxnew00/out_%a.log   # Output log (must exist before submit)
#SBATCH --error=/home2/vwetzell/ProperMotion/packing_auxnew00/err_%a.err    # Error log (must exist before submit)
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using bhuv_compute
#SBATCH --array=0-500                                                    # Job array range (= len(healpixel.npy))
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion/packing_auxnew00
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/detectionPacking.py \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion/healpixel.npy \
    --exposures-file /home2/vwetzell/gitrepos/pixmappy/data/y6a1.exposureinfo.fits \
    --detection-path /home2/vwetzell/ProperMotion/PositionCorrectedExposureCatalog/ \
    --output-path /home2/vwetzell/ProperMotion/HealpixDetectionCatalog/ \
    --overwrite

finish
