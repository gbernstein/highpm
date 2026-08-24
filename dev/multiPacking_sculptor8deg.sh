#!/usr/bin/zsh
#SBATCH --job-name=PackingSculptor8deg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=/home2/vwetzell/ProperMotion_Sculptor/packing_auxnew00_8deg/out_%a.log
#SBATCH --error=/home2/vwetzell/ProperMotion_Sculptor/packing_auxnew00_8deg/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-M          # M = len(sculptor8deg_healpix.npy) - 1
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion_Sculptor/packing_auxnew00_8deg
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/detectionPacking.py \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion_Sculptor/sculptor8deg_healpix.npy \
    --exposures-file /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
    --detection-path /home2/vwetzell/ProperMotion_Sculptor/PositionCorrectedExposureCatalog_8deg/ \
    --output-path /home2/vwetzell/ProperMotion_Sculptor/HealpixDetectionCatalog_8deg/ \
    --overwrite

finish
