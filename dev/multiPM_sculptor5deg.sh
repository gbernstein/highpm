#!/usr/bin/zsh
#SBATCH --job-name=PMSculptor
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=2:00:00
#SBATCH --output=/home2/vwetzell/ProperMotion_Sculptor/pm_auxnew00/out_%a.log
#SBATCH --error=/home2/vwetzell/ProperMotion_Sculptor/pm_auxnew00/err_%a.err
#SBATCH -p low
#SBATCH -q low
#SBATCH --array=0-M          # M = len(healpixels.npy) - 1
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion_Sculptor/pm_auxnew00
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/fake_pipeline.py \
    --config /home2/vwetzell/gitrepos/highpm/config/config.yaml \
    --detections "/home2/vwetzell/ProperMotion_Sculptor/HealpixDetectionCatalog/cleaned_detections_*.fits" \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion_Sculptor/healpixels.npy \
    --runs real,fake \
    --output-name /home2/vwetzell/ProperMotion_Sculptor/PMCatalog/

finish
