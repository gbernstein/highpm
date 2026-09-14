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
DES_EXPOSURES="$REPO/../pixmappy/pixmappy/data/delveExposures.hdf5"

MAX_OOM_RETRIES=3
MAX_ARRAY_TASKS=1200  # QOSMaxSubmitJobPerUserLimit on this cluster

source /home2/vwetzell/.bashrc
conda activate pm

mkdir -p "$BASE" "$BASE/logs/poscorr" "$BASE/logs/packing" "$BASE/logs/pm" "$POSCORR_OUT"

# Submits one array job (<=MAX_ARRAY_TASKS tasks) and waits on it (--requeue
# in the .sh handles SLURM preemption on its own). If any task OOM'd,
# resubmit just those tasks at double the memory, up to MAX_OOM_RETRIES times.
submit_batch() {
    local script="$1" array_spec="$2" mem_gb="$3"
    local attempt=0 jobid oom_tasks
    while true; do
        echo "Submitting $(basename "$script") (array=$array_spec, mem=${mem_gb}gb)"
        set +e
        jobid=$(sbatch --parsable --wait --array="$array_spec" --mem="${mem_gb}gb" \
            --export="ALL,POSCORR_CHUNK=${POSCORR_CHUNK:-}" "$script")
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

# Runs the full 0..n_tasks-1 range as successive submit_batch calls, each
# <=MAX_ARRAY_TASKS tasks (the cluster's QOSMaxSubmitJobPerUserLimit).
submit_stage() {
    local script="$1" n_tasks="$2" mem_gb="$3"
    local start=0 end
    while [ "$start" -lt "$n_tasks" ]; do
        end=$(( start + MAX_ARRAY_TASKS - 1 ))
        if [ "$end" -ge "$n_tasks" ]; then
            end=$(( n_tasks - 1 ))
        fi
        submit_batch "$script" "$start-$end" "$mem_gb"
        start=$(( end + 1 ))
    done
}

echo "== Stage 0: build exposure list =="
python "$REPO/dev/build_pmsculptor_exposures.py" \
    --skims-path "$SKIMS_PATH" \
    --gpr-path "$GPR_PATH" \
    --output-file "$EXPOSURES_NPY"

n_exposures=$(python -c "import numpy as np; print(len(np.load('$EXPOSURES_NPY')))")
# Chunk size must be big enough that ceil(n_exposures/chunk) fits in one
# stage's worth of array batches (MAX_ARRAY_TASKS each); multiPositionCorrection
# reads --chunk-size exposures per task via --index/--chunk-size.
POSCORR_CHUNK=$(( (n_exposures + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
n_pc=$(( (n_exposures + POSCORR_CHUNK - 1) / POSCORR_CHUNK ))
echo "$n_exposures exposures -> $n_pc position-correction array tasks (chunk=$POSCORR_CHUNK)"

echo "== Stage 1: position correction =="
submit_stage "$REPO/dev/multiPositionCorrection_pmsculptor.sh" "$n_pc" 4

echo "== Stage 2: build healpix list from position-corrected exposures =="
python "$REPO/dev/healpixFromExposures.py" \
    --exposure-dir "$POSCORR_OUT" \
    --exposure-table "$DES_EXPOSURES" \
    --output-file "$HEALPIX_NPY"

n_healpix=$(python -c "import numpy as np; print(len(np.load('$HEALPIX_NPY')))")
echo "$n_healpix healpixels -> packing/PM array tasks"

echo "== Stage 3: detection packing =="
submit_stage "$REPO/dev/multiPacking_pmsculptor.sh" "$n_healpix" 32

echo "== Stage 4: proper motion fit =="
submit_stage "$REPO/dev/multiPM_pmsculptor.sh" "$n_healpix" 32

echo "Done. Outputs under $BASE"
