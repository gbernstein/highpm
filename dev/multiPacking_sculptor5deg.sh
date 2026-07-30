#!/usr/bin/zsh
#SBATCH --job-name=PackingSculptor
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=/home2/vwetzell/ProperMotion_Sculptor/packing_auxnew00/out_%a.log
#SBATCH --error=/home2/vwetzell/ProperMotion_Sculptor/packing_auxnew00/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-M          # M = len(healpixels.npy) - 1
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion_Sculptor/packing_auxnew00
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/detectionPacking.py \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion_Sculptor/healpixels.npy \
    --exposures-file /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
    --detection-path /home2/vwetzell/ProperMotion_Sculptor/PositionCorrectedExposureCatalog/ \
    --output-path /home2/vwetzell/ProperMotion_Sculptor/HealpixDetectionCatalog/ \
    --overwrite

finish
