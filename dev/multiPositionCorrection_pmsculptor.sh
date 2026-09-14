#!/usr/bin/zsh
#SBATCH --job-name=PosCorrPMSculptor
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb
#SBATCH --time=48:00:00
#SBATCH --output=/data8/shared/decampm/PMSculptor/logs/poscorr/out_%a.log
#SBATCH --error=/data8/shared/decampm/PMSculptor/logs/poscorr/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-N          # N = ceil(len(pmsculptor_exposures.npy) / 2) - 1
#SBATCH --exclude=node[01-12]
#SBATCH --requeue

AUXDIR=/data8/shared/decampm/PMSculptor/logs/poscorr
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/positionCorrection.py \
    --skims-path "/data8/shared/decampm/[GRIZ]/" \
    --gpr-path "/data8/shared/decampm/GPR2/CAT/[griz]/" \
    --output-path /data8/shared/decampm/PMSculptor/PositionCorrectedExposureCatalog/ \
    --des-exposures /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
    --processes 1 \
    --exposures-npy /data8/shared/decampm/PMSculptor/pmsculptor_exposures.npy \
    --index ${SLURM_ARRAY_TASK_ID} \
    --chunk-size 2

finish
