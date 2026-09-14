#!/usr/bin/zsh
#SBATCH --job-name=PackingPMSculptor
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=/data8/shared/decampm/PMSculptor/logs/packing/out_%a.log
#SBATCH --error=/data8/shared/decampm/PMSculptor/logs/packing/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-M          # M = len(pmsculptor_healpix.npy) - 1
#SBATCH --exclude=node[01-12]
#SBATCH --requeue

AUXDIR=/data8/shared/decampm/PMSculptor/logs/packing
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/detectionPacking.py \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /data8/shared/decampm/PMSculptor/pmsculptor_healpix.npy \
    --exposures-file /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
    --detection-path /data8/shared/decampm/PMSculptor/PositionCorrectedExposureCatalog/ \
    --output-path /data8/shared/decampm/PMSculptor/HealpixDetectionCatalog/ \
    --overwrite

finish
