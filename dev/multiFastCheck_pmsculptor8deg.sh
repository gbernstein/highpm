#!/usr/bin/zsh
#SBATCH --job-name=FCPMSculptor8deg
#SBATCH --cpus-per-task=1        # fast check is effectively single-threaded (1 vs 4 vs 16 cores: same ~57s on the largest healpix)
#SBATCH --mem=10gb               # measured peak RSS 4.4-5.1GB on the largest healpixels (hp09413/09541/09797, ~20M detections); sacct sometimes reports up to ~9GB, so 10GB
#SBATCH --time=0:30:00           # ~1 min per run (real + injection ~2 min on the largest)
#SBATCH --output=/data8/shared/decampm/PMSculptor_8deg_garyb/logs/fastcheck/out_%a.log
#SBATCH --error=/data8/shared/decampm/PMSculptor_8deg_garyb/logs/fastcheck/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-45%60      # len(pmsculptor8deg_healpix.npy) = 46 (GPR2-complete only)
#SBATCH --exclude=node[01-12]
#SBATCH --requeue
# Quota (qos low): 700 cpu / 1T mem / 200 jobs per user. 60 concurrent x 1cpu x 10GB = 60 cpu / 600GB.
# Healpixels with no real_PM file are skipped by fastcheck_pipeline; ones with no fast_movers
# extension (hp09797) are skipped by fastChecks. Log dir must exist before submitting.

AUXDIR=/data8/shared/decampm/PMSculptor_8deg_garyb/logs/fastcheck
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

export DES_EXPOSURES=/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5

python /home2/vwetzell/gitrepos/highpm/scripts/fastcheck_pipeline.py \
    --config /home2/vwetzell/gitrepos/highpm/config/config.yaml \
    --detections "/data8/shared/decampm/PMSculptor_8deg_garyb/HealpixDetectionCatalog/cleaned_detections_*.fits" \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /data8/shared/decampm/PMSculptor_8deg_garyb/pmsculptor8deg_healpix.npy \
    --output-name /data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/ \
    --cores ${SLURM_CPUS_PER_TASK} \
    --runs real,injection

finish
