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

# Compresses a sorted comma-separated int list into SLURM array range syntax
# (e.g. "0,1,2,3" -> "0-3"), since sbatch rejects a submission whose --array
# string is too long and task lists here are usually contiguous.
compress_array_spec() {
    awk -v RS=',' '
    { n=$1+0; if (first=="") { start=prev=n; first=1; next }
      if (n==prev+1) { prev=n; next }
      printf "%s%s", (out?",":""), (start==prev?start:start"-"prev); out=1
      start=prev=n }
    END { if (first!="") printf "%s%s", (out?",":""), (start==prev?start:start"-"prev); print "" }
    ' <<<"$1"
}

# Submits one array job (<=MAX_ARRAY_TASKS tasks) and waits on it (--requeue
# in the .sh handles SLURM preemption on its own). If any task OOM'd,
# resubmit just those tasks at double the memory, up to MAX_OOM_RETRIES times.
submit_batch() {
    local script="$1" array_spec="$2" mem_gb="$3"
    local attempt=0 jobid oom_tasks bad_tasks spec
    while true; do
        spec=$(compress_array_spec "$array_spec")
        echo "Submitting $(basename "$script") (array=$spec, mem=${mem_gb}gb)"
        set +e
        jobid=$(sbatch --parsable --wait --array="$spec" --mem="${mem_gb}gb" \
            --export="ALL,POSCORR_CHUNK=${POSCORR_CHUNK:-},POSCORR_EXPOSURES_NPY=${POSCORR_EXPOSURES_NPY:-}" "$script")
        set -e
        # Only look at the array-task rows (12345_3), not their .batch/.extern steps.
        oom_tasks=$(sacct -j "$jobid" --noheader --parsable2 --format=JobID,State \
            | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ && $2 ~ /^OUT_OF_MEM/ {print $1}' \
            | sed -E 's/^[0-9]+_([0-9]+).*/\1/' \
            | sort -un | paste -sd, -)
        bad_tasks=$(sacct -j "$jobid" --noheader --parsable2 --format=JobID,State \
            | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ && $2 !~ /^(COMPLETED|OUT_OF_MEM)/ {print}')
        if [ -n "$bad_tasks" ]; then
            echo "ERROR: $(basename "$script") tasks finished in a non-COMPLETED, non-OOM state:" >&2
            echo "$bad_tasks" >&2
            return 1
        fi
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

# Runs a comma-separated list of pending task indices as successive
# submit_batch calls, each <=MAX_ARRAY_TASKS tasks (the cluster's
# QOSMaxSubmitJobPerUserLimit). Empty list = nothing pending, skip entirely.
submit_stage() {
    local script="$1" tasks_csv="$2" mem_gb="$3"
    if [ -z "$tasks_csv" ]; then
        echo "  all tasks already done, skipping $(basename "$script")"
        return 0
    fi
    local -a tasks=(${tasks_csv//,/ })
    local n=${#tasks[@]} start=0 count
    while [ "$start" -lt "$n" ]; do
        count=$MAX_ARRAY_TASKS
        if [ "$((start + count))" -gt "$n" ]; then
            count=$(( n - start ))
        fi
        submit_batch "$script" "$(IFS=,; echo "${tasks[*]:$start:$count}")" "$mem_gb"
        start=$(( start + count ))
    done
}

# Writes exposures still missing position-correction output to
# POSCORR_EXPOSURES_NPY and prints how many there are.
write_pending_exposures() {
    python -c "
import glob, re
import numpy as np
exposures = np.load('$EXPOSURES_NPY')
done = {int(re.search(r'(\d+)\.fits\$', f).group(1))
        for f in glob.glob('$POSCORR_OUT/position_corrected_*.fits')}
pending = np.array([e for e in exposures if int(e) not in done])
np.save('$POSCORR_EXPOSURES_NPY', pending)
print(len(pending))
"
}

# Prints comma-separated healpix-array indices (0..n_healpix-1) missing any
# of the given output filename patterns ({hp} = healpix number, e.g.
# ".../cleaned_detections_hp{hp:05d}.fits").
pending_healpix_tasks() {
    python -c "
import os, sys
import numpy as np
healpix = np.load('$HEALPIX_NPY')
patterns = sys.argv[1:]
pending = [i for i, hp in enumerate(healpix)
           if not all(os.path.exists(p.format(hp=int(hp))) for p in patterns)]
print(','.join(map(str, pending)))
" "$@"
}

echo "== Stage 0: build exposure list =="
python "$REPO/dev/build_pmsculptor_exposures.py" \
    --skims-path "$SKIMS_PATH" \
    --gpr-path "$GPR_PATH" \
    --output-file "$EXPOSURES_NPY"

n_exposures=$(python -c "import numpy as np; print(len(np.load('$EXPOSURES_NPY')))")

echo "== Stage 1: position correction =="
POSCORR_EXPOSURES_NPY="$BASE/pmsculptor_exposures_pending.npy"
n_pending=$(write_pending_exposures)
if [ "$n_pending" -eq 0 ]; then
    echo "  all $n_exposures exposures already position-corrected, skipping"
    force_repack=0
else
    # Chunk size must be big enough that ceil(n_pending/chunk) fits in one
    # stage's worth of array batches (MAX_ARRAY_TASKS each); multiPositionCorrection
    # reads --chunk-size exposures per task via --index/--chunk-size, indexing
    # into POSCORR_EXPOSURES_NPY (just the pending exposures, not the full list).
    POSCORR_CHUNK=$(( (n_pending + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
    n_pc=$(( (n_pending + POSCORR_CHUNK - 1) / POSCORR_CHUNK ))
    echo "$n_pending / $n_exposures exposures pending -> $n_pc position-correction array tasks (chunk=$POSCORR_CHUNK)"
    submit_stage "$REPO/dev/multiPositionCorrection_pmsculptor.sh" "$(seq -s, 0 $(( n_pc - 1 )))" 4
    # Already-packed healpix may have had incomplete input last time --
    # reprocess all of them rather than trusting per-file "already done"
    # checks for stages 3-4.
    force_repack=1
fi

echo "== Stage 2: build healpix list from position-corrected exposures =="
python "$REPO/dev/healpixFromExposures.py" \
    --exposure-dir "$POSCORR_OUT" \
    --exposure-table "$DES_EXPOSURES" \
    --output-file "$HEALPIX_NPY"

n_healpix=$(python -c "import numpy as np; print(len(np.load('$HEALPIX_NPY')))")
echo "$n_healpix healpixels -> packing/PM array tasks"

echo "== Stage 3: detection packing =="
if [ "$force_repack" -eq 1 ]; then
    echo "  new position-correction output this run; reprocessing all $n_healpix healpix"
    packing_pending="$(seq -s, 0 $(( n_healpix - 1 )))"
else
    # ponytail: a healpix with zero cleaned detections after cuts never
    # produces this file and so is "pending" forever; harmless, it just
    # reruns (and no-ops) on every resume.
    packing_pending="$(pending_healpix_tasks "$BASE/HealpixDetectionCatalog/cleaned_detections_hp{hp:05d}.fits")"
fi
submit_stage "$REPO/dev/multiPacking_pmsculptor.sh" "$packing_pending" 32

echo "== Stage 4: proper motion fit =="
if [ "$force_repack" -eq 1 ] || [ -n "$packing_pending" ]; then
    echo "  packing changed this run; reprocessing all $n_healpix healpix"
    pm_pending="$(seq -s, 0 $(( n_healpix - 1 )))"
else
    pm_pending="$(pending_healpix_tasks \
        "$BASE/PMCatalog/real_PM_hp{hp:05d}.fits" \
        "$BASE/PMCatalog/injection_PM_hp{hp:05d}.fits")"
fi
submit_stage "$REPO/dev/multiPM_pmsculptor.sh" "$pm_pending" 32

echo "Done. Outputs under $BASE"
