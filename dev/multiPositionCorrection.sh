#!/usr/bin/zsh
#SBATCH --job-name=PositionCorrection_z                                  # Job name
#SBATCH --cpus-per-task=4                                                # CPUs per task
#SBATCH --mem=4gb                                                        # Job memory request
#SBATCH --time=48:00:00                                                  # Time limit hrs:min:sec
#SBATCH --output=/home2/vwetzell/ProperMotion/auxnew04/out_%a.log        # Output log (must exist before submit)
#SBATCH --error=/home2/vwetzell/ProperMotion/auxnew04/err_%a.err         # Error log (must exist before submit)
#SBATCH -p low                                                           # Partition
#SBATCH -q low                                                           # Remove if using bhuv_compute
#SBATCH --array=0-1199                                                   # Job array range (= len(z_exposures.npy))
#SBATCH --exclude=node[01-12]

AUXDIR=/home2/vwetzell/ProperMotion/auxnew04
source /home2/vwetzell/gitrepos/highpm/dev/slurm_common.sh

python /home2/vwetzell/gitrepos/highpm/scripts/positionCorrection.py \
	--skims-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/des_skim/z/ \
	--gpr-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/full_des_y6_run/cat/z/ \
	--coadd-path /home2/dchgomes/Turbulence_GPR/DES_DELVE/coadd_skims/ \
	--output-path /home2/vwetzell/ProperMotion/PositionCorrectedExposureCatalog/z/ \
	--des-exposures /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
	--processes 1 \
	--exposures-npy /home2/vwetzell/ProperMotion/PositionCorrectedExposureCatalog/z_exposures.npy \
	--index ${SLURM_ARRAY_TASK_ID}

finish
