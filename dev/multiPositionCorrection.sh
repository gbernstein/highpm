#!/usr/bin/zsh
#SBATCH --job-name=PositionCorrection_z                                  # Job name
#SBATCH --cpus-per-task=4                                                # CPUs par task
#SBATCH --mem=4gb                                                        # Job memory request
#SBATCH --time=48:00:00                                                  # Time limit hrs:min:sec
#SBATCH --output=/home2/vwetzell/ProperMotion/auxnew04/out_%a.log          # Output log (only task ID)
#SBATCH --error=/home2/vwetzell/ProperMotion/auxnew04/err_%a.err           # Error log (only task ID)
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using bhuv_compute
#SBATCH --array=0-1199                                                   # Job array range
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
python /home2/vwetzell/gitrepos/highpm/scripts/positionCorrection.py \
	--skims-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/des_skim/z/ \
	--gpr-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/full_des_y6_run/cat/z/ \
	--coadd-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/coadd_skims/ \
	--output-path /home2/vwetzell/ProperMotion/PositionCorrectedExposureCatalog/z/ \
	--des-exposures /home2/vwetzell/gitrepos/pixmappy/data/y6a1.exposureinfo.fits \
	--processes 1 \
	--exposures-npy /home2/vwetzell/ProperMotion/PositionCorrectedExposureCatalog/z_exposures.npy \
	--index ${SLURM_ARRAY_TASK_ID}

# Record end time and compute elapsed time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# Ensure aux directory exists
mkdir -p /home2/vwetzell/ProperMotion/auxnew04

# Append timing info to a shared timing file
echo "Index ${SLURM_ARRAY_TASK_ID} finished at $(date), total time: ${elapsed}s" \
    >> /home2/vwetzell/ProperMotion/auxnew04/time_all.txt

echo "Task ${SLURM_ARRAY_TASK_ID} completed at $(date), runtime: ${elapsed}s"
