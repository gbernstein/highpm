#!/usr/bin/zsh
#SBATCH --job-name=Packing                                               # Job name
#SBATCH --cpus-per-task=4                                                # CPUs par task
#SBATCH --mem=32gb                                                        # Job memory request
#SBATCH --time=48:00:00                                                  # Time limit hrs:min:>
#SBATCH --output=/home2/vwetzell/ProperMotion/packing_auxnew00/out_%a.log          # Output log (only >
#SBATCH --error=/home2/vwetzell/ProperMotion/packing_auxnew00/err_%a.err           # Error log (only t>
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using bhu>
#SBATCH --array=0-500                                                   # Job array range
#SBATCH --exclude=node[01-12]

echo "Running on host: $(hostname) at $(date)"

# Record start time
start_time=$(date +%s)

# Activate conda environment
source /home2/vwetzell/.bashrc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate pm

# Run the Python script
echo "Processing healpix $HEALPIX"
python /home2/vwetzell/gitrepos/highpm/scripts/detectionPacking.py \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion/healpixel.npy \
    --exposures-file /home2/vwetzell/gitrepos/pixmappy/data/y6a1.exposureinfo.fits \
    --detection-path /home2/vwetzell/ProperMotion/PositionCorrectedExposureCatalog/ \
    --output-path /home2/vwetzell/ProperMotion/HealpixDetectionCatalog/ \
    --overwrite

# Record end time and compute elapsed time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# Ensure aux directory exists
mkdir -p /home2/vwetzell/ProperMotion/packing_auxnew00

# Append timing info to a shared timing file
echo "Index ${SLURM_ARRAY_TASK_ID} finished at $(date), total time: ${elapsed}s" \
    >> /home2/vwetzell/ProperMotion/packing_auxnew00/time_all.txt

echo "Task ${SLURM_ARRAY_TASK_ID} completed at $(date), runtime: ${elapsed}s"
