#!/usr/bin/zsh
#SBATCH --job-name=PMPMSculptor8deg
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --output=/data8/shared/decampm/PMSculptor_8deg_garyb/logs/pm/out_%a.log
#SBATCH --error=/data8/shared/decampm/PMSculptor_8deg_garyb/logs/pm/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-M          # M = len(pmsculptor8deg_healpix.npy) - 1
#SBATCH --exclude=node[01-12]
#SBATCH --requeue

AUXDIR=/data8/shared/decampm/PMSculptor_8deg_garyb/logs/pm
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

export DES_EXPOSURES=/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5

python /home2/vwetzell/gitrepos/highpm/scripts/fake_pipeline.py \
    --config /home2/vwetzell/gitrepos/highpm/config/config.yaml \
    --detections "/data8/shared/decampm/PMSculptor_8deg_garyb/HealpixDetectionCatalog/cleaned_detections_*.fits" \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /data8/shared/decampm/PMSculptor_8deg_garyb/pmsculptor8deg_healpix.npy \
    --output-name /data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/ \
    --runs real,injection

finish
