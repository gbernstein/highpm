#!/usr/bin/zsh
#SBATCH --job-name=PM                                                    # Job name
#SBATCH --cpus-per-task=32                                               # CPUs par task
#SBATCH --mem=128gb                                                       # Job memory requ>
#SBATCH --time=2:00:00                                                  # Time limit hrs:m>
#SBATCH --output=/home2/vwetzell/ProperMotion/pm_auxnew02/out_%a.log          # Output>
#SBATCH --error=/home2/vwetzell/ProperMotion/pm_auxnew02/err_%a.err           # Error >
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using >
#SBATCH --array=0-8                                                     # Job array range
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
echo "Processing index ${SLURM_ARRAY_TASK_ID}"
python /home2/vwetzell/gitrepos/highpm/scripts/PM.py \
    --config /home2/vwetzell/gitrepos/highpm/config/config.yaml \
    --detections "/home2/vwetzell/ProperMotion/HealpixDetectionCatalog/cleaned_detections_*.fits" \
    --index ${SLURM_ARRAY_TASK_ID} \
    --healpix-npy /home2/vwetzell/ProperMotion/PMToDo.npy \
    --output-name /home2/vwetzell/ProperMotion/PMCatalog/

# Record end time and compute elapsed time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# Ensure aux directory exists
mkdir -p /home2/vwetzell/ProperMotion/pm_auxnew02

# Append timing info to a shared timing file
echo "Index ${SLURM_ARRAY_TASK_ID} finished at $(date), total time: ${elapsed}s" \
    >> /home2/vwetzell/ProperMotion/pm_auxnew02/time_all.txt

echo "Task ${SLURM_ARRAY_TASK_ID} completed at $(date), runtime: ${elapsed}s"


