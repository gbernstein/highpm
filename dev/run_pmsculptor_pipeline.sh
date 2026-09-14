#!/usr/bin/env bash
# Runs the whole PMSculptor pipeline (reprocessed GPR2 Sculptor exposures) end
# to end: builds the exposure/healpix .npy lists, then submits and waits on
# each SLURM stage in turn, sizing --array from what was just built instead of
# a hand-edited placeholder.
#
# Run on the HPC login node (needs /data8 + sbatch):
#     dev/run_pmsculptor_pipeline.sh
set -euo pipefail

REPO=/home2/vwetzell/gitrepos/highpm
BASE=/data8/shared/decampm/PMSculptor
SKIMS_PATH="/data8/shared/decampm/[GRIZ]/"
GPR_PATH="/data8/shared/decampm/GPR2/CAT/[griz]/"
EXPOSURES_NPY="$BASE/pmsculptor_exposures.npy"
HEALPIX_NPY="$BASE/pmsculptor_healpix.npy"
POSCORR_OUT="$BASE/PositionCorrectedExposureCatalog/"
POSCORR_CHUNK=2
DES_EXPOSURES="$REPO/../pixmappy/pixmappy/data/delveExposures.hdf5"

MAX_OOM_RETRIES=3

source /home2/vwetzell/.bashrc
conda activate pm

mkdir -p "$BASE" "$BASE/logs/poscorr" "$BASE/logs/packing" "$BASE/logs/pm" "$POSCORR_OUT"

# Submits an array job and waits on it (--requeue in the .sh handles SLURM
# preemption on its own). If any task OOM'd, resubmit just those tasks at
# double the memory, up to MAX_OOM_RETRIES times.
submit_stage() {
    local script="$1" array_spec="$2" mem_gb="$3"
    local attempt=0 jobid oom_tasks
    while true; do
        echo "Submitting $(basename "$script") (array=$array_spec, mem=${mem_gb}gb)"
        set +e
        jobid=$(sbatch --parsable --wait --array="$array_spec" --mem="${mem_gb}gb" "$script")
        set -e
        oom_tasks=$(sacct -j "$jobid" --noheader --parsable2 --format=JobID,State \
            | awk -F'|' '$2 ~ /^OUT_OF_MEM/ {print $1}' \
            | sed -E 's/^[0-9]+_([0-9]+).*/\1/' \
            | sort -un | paste -sd, -)
        if [ -z "$oom_tasks" ]; then
            return 0
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_OOM_RETRIES" ]; then
            echo "ERROR: $(basename "$script") tasks [$oom_tasks] still OOMing after $MAX_OOM_RETRIES retries at ${mem_gb}gb" >&2
            return 1
        fi
        mem_gb=$((mem_gb * 2))
        array_spec="$oom_tasks"
        echo "Tasks [$array_spec] OOM'd; retrying at ${mem_gb}gb"
    done
}

echo "== Stage 0: build exposure list =="
python "$REPO/dev/build_pmsculptor_exposures.py" \
    --skims-path "$SKIMS_PATH" \
    --gpr-path "$GPR_PATH" \
    --output-file "$EXPOSURES_NPY"

n_exposures=$(python -c "import numpy as np; print(len(np.load('$EXPOSURES_NPY')))")
n_pc=$(( (n_exposures + POSCORR_CHUNK - 1) / POSCORR_CHUNK ))
echo "$n_exposures exposures -> $n_pc position-correction array tasks"

echo "== Stage 1: position correction =="
submit_stage "$REPO/dev/multiPositionCorrection_pmsculptor.sh" "0-$((n_pc - 1))" 4

echo "== Stage 2: build healpix list from position-corrected exposures =="
python "$REPO/dev/healpixFromExposures.py" \
    --exposure-dir "$POSCORR_OUT" \
    --exposure-table "$DES_EXPOSURES" \
    --output-file "$HEALPIX_NPY"

n_healpix=$(python -c "import numpy as np; print(len(np.load('$HEALPIX_NPY')))")
echo "$n_healpix healpixels -> packing/PM array tasks"

echo "== Stage 3: detection packing =="
submit_stage "$REPO/dev/multiPacking_pmsculptor.sh" "0-$((n_healpix - 1))" 32

echo "== Stage 4: proper motion fit =="
submit_stage "$REPO/dev/multiPM_pmsculptor.sh" "0-$((n_healpix - 1))" 32

echo "Done. Outputs under $BASE"
