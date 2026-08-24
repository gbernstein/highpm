#!/usr/bin/zsh
#SBATCH --job-name=PMSculptor8deg
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=2:00:00
#SBATCH --output=/home2/vwetzell/ProperMotion_Sculptor/pm_auxnew00_8deg/out_%a.log
#SBATCH --error=/home2/vwetzell/ProperMotion_Sculptor/pm_auxnew00_8deg/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-M          # M = len(sculptor8deg_healpix.npy) - 1
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion_Sculptor/pm_auxnew00_8deg
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/fake_pipeline.py \
    --config /home2/vwetzell/gitrepos/highpm/config/config.yaml \
    --detections "/home2/vwetzell/ProperMotion_Sculptor/HealpixDetectionCatalog_8deg/cleaned_detections_*.fits" \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion_Sculptor/sculptor8deg_healpix.npy \
    --runs real,fake \
    --output-name /home2/vwetzell/ProperMotion_Sculptor/PMCatalog_8deg/

finish
