#!/usr/bin/zsh
#SBATCH --job-name=PositionCorrection_z                                  # Job name
#SBATCH --cpus-per-task=4                                                # CPUs per task
#SBATCH --mem=4gb                                                        # Job memory request
#SBATCH --time=48:00:00                                                  # Time limit hrs:min:sec
#SBATCH --output=/home2/vwetzell/ProperMotion_v3/auxnew00/out_%a.log        # Output log (must exist before submit)
#SBATCH --error=/home2/vwetzell/ProperMotion_v3/auxnew00/err_%a.err         # Error log (must exist before submit)
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using bhuv_compute
#SBATCH --array=0-99                                                     # Job array range (= ceil(len(z_exposures.npy) / CHUNK_SIZE) - 1)
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion_v3/auxnew00
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/positionCorrection.py \
	--skims-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/delve_skims/cat/z/ \
	--gpr-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/sculptor_nwcs/cat/z/ \
	--coadd-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/coadd_skims/ \
	--output-path /home2/vwetzell/ProperMotion_v3/PositionCorrectedExposureCatalog/z/ \
	--des-exposures /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
	--processes 1 \
	--exposures-npy /home2/vwetzell/ProperMotion_v3/PositionCorrectedExposureCatalog/z_exposures.npy \
	--index ${SLURM_ARRAY_TASK_ID} \
	--chunk-size 4

finish
