#!/usr/bin/zsh
#SBATCH --job-name=PosCorrSculptor8deg
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb
#SBATCH --time=48:00:00
#SBATCH --output=/home2/vwetzell/ProperMotion_Sculptor/auxnew00_8deg/out_%a.log
#SBATCH --error=/home2/vwetzell/ProperMotion_Sculptor/auxnew00_8deg/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-N          # N = ceil(len(sculptor8deg_exposures.npy) / 2) - 1
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion_Sculptor/auxnew00_8deg
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/positionCorrection.py \
    --skims-path "/data8/shared/decampm/[GRIZ]/" \
    --gpr-path "/data8/shared/decampm/prev_des/[griz]/" \
    --output-path /home2/vwetzell/ProperMotion_Sculptor/PositionCorrectedExposureCatalog_8deg/ \
    --des-exposures /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
    --processes 4 \
    --exposures-npy /home2/vwetzell/ProperMotion_Sculptor/sculptor8deg_exposures.npy \
    --index ${SLURM_ARRAY_TASK_ID} \
    --chunk-size 2

finish
