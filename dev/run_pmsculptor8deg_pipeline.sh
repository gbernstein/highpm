#!/usr/bin/env bash
# Runs the whole PMSculptor8deg pipeline end to end: builds the exposure list
# restricted to healpix within REGION_RADIUS_DEG of Sculptor, using the GPR2
# reprocessing with --complete-only (only healpix whose exposures are all in
# GPR2/CAT, and only the exposures those healpix need), then submits and waits
# on each SLURM stage in turn, sizing --array
# from what was just built instead of a hand-edited placeholder.
#
# Unlike run_pmsculptor_pipeline.sh (which processes the *entire* skim/GPR
# overlap), the healpix list here is fixed up front from the Sculptor region
# rather than rebuilt from whatever exposures happened to get processed, so
# there's no separate "build healpix list" stage.
#
# Run on the HPC login node (needs /data8 + sbatch):
#     dev/run_pmsculptor8deg_pipeline.sh
set -euo pipefail

REPO=/home2/vwetzell/gitrepos/highpm
BASE=/data8/shared/decampm/PMSculptor_8deg_garyb
SKIMS_PATH="/data8/shared/decampm/[GRIZ]/"
GPR_PATH="/data8/shared/decampm/GPR2/CAT/[griz]/"
EXPOSURES_NPY="$BASE/pmsculptor8deg_exposures.npy"
HEALPIX_NPY="$BASE/pmsculptor8deg_healpix.npy"
POSCORR_OUT="$BASE/PositionCorrectedExposureCatalog/"
DES_EXPOSURES="$REPO/../pixmappy/pixmappy/data/delveExposures.hdf5"

SCULPTOR_RA=15.038750
SCULPTOR_DEC=-33.709000
REGION_RADIUS_DEG=8.0  # disc radius around Sculptor for the healpix list

MAX_OOM_RETRIES=3
MAX_PREEMPT_RETRIES=5
MAX_ARRAY_TASKS=1200  # QOSMaxSubmitJobPerUserLimit on this cluster
NSIDE=32              # must match build_pmsculptor8deg_exposures.py's default
DISC_RADIUS_DEG=1.1   # ditto -- ~DES focal plane radius, used for touched-healpix lookup

source /home2/vwetzell/.bashrc
conda activate pm

mkdir -p "$BASE" "$BASE/logs/poscorr" "$BASE/logs/packing" "$BASE/logs/pm" "$BASE/logs/fastcheck" "$POSCORR_OUT" "$BASE/PMCatalog"

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

# Prints the comma-separated array-task indices of $1 (a job id) whose State
# matches the regex $2. Only looks at the array-task rows (12345_3), not
# their .batch/.extern steps.
tasks_in_state() {
    sacct -j "$1" --noheader --parsable2 --format=JobID,State \
        | awk -F'|' -v re="$2" '$1 ~ /^[0-9]+_[0-9]+$/ && $2 ~ re {print $1}' \
        | sed -E 's/^[0-9]+_([0-9]+).*/\1/' \
        | sort -un | paste -sd, -
}

# Submits one array job (<=MAX_ARRAY_TASKS tasks) and waits on it. If any
# task OOM'd, resubmit just those tasks at double the memory, up to
# MAX_OOM_RETRIES times. The low partition is PreemptMode=CANCEL, so
# --requeue does nothing there: preempted (or node-failed) tasks are
# resubmitted here at the same memory, up to MAX_PREEMPT_RETRIES times. Each
# stage's script overwrites (or first removes) its own outputs, so a rerun
# never builds on a partial file left by the cancelled attempt.
submit_batch() {
    local script="$1" array_spec="$2" mem_gb="$3"
    local oom_attempt=0 preempt_attempt=0 jobid oom_tasks preempted_tasks bad_tasks spec
    while true; do
        spec=$(compress_array_spec "$array_spec")
        echo "Submitting $(basename "$script") (array=$spec, mem=${mem_gb}gb)"
        set +e
        jobid=$(sbatch --parsable --wait --array="$spec" --mem="${mem_gb}gb" \
            --export="ALL,POSCORR_CHUNK=${POSCORR_CHUNK:-},POSCORR_EXPOSURES_NPY=${POSCORR_EXPOSURES_NPY:-}" "$script")
        set -e
        oom_tasks=$(tasks_in_state "$jobid" '^OUT_OF_MEM')
        preempted_tasks=$(tasks_in_state "$jobid" '^(PREEMPTED|NODE_FAIL|BOOT_FAIL)')
        bad_tasks=$(sacct -j "$jobid" --noheader --parsable2 --format=JobID,State \
            | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ && $2 !~ /^(COMPLETED|OUT_OF_MEM|PREEMPTED|NODE_FAIL|BOOT_FAIL)/ {print}')
        if [ -n "$bad_tasks" ]; then
            echo "ERROR: $(basename "$script") tasks finished in a non-COMPLETED, non-retryable state:" >&2
            echo "$bad_tasks" >&2
            return 1
        fi
        if [ -z "$oom_tasks" ] && [ -z "$preempted_tasks" ]; then
            return 0
        fi
        if [ -n "$oom_tasks" ]; then
            oom_attempt=$((oom_attempt + 1))
            if [ "$oom_attempt" -gt "$MAX_OOM_RETRIES" ]; then
                echo "ERROR: $(basename "$script") tasks [$oom_tasks] still OOMing after $MAX_OOM_RETRIES retries at ${mem_gb}gb" >&2
                return 1
            fi
        fi
        if [ -n "$preempted_tasks" ]; then
            preempt_attempt=$((preempt_attempt + 1))
            if [ "$preempt_attempt" -gt "$MAX_PREEMPT_RETRIES" ]; then
                echo "ERROR: $(basename "$script") tasks [$preempted_tasks] still preempted after $MAX_PREEMPT_RETRIES retries" >&2
                return 1
            fi
            echo "Tasks [$preempted_tasks] preempted/node-failed; resubmitting"
        fi
        # OOM'd and preempted tasks go out together in one array; OOM doubles
        # the memory for all of them (harmless over-ask for the preempted ones).
        if [ -n "$oom_tasks" ]; then
            mem_gb=$((mem_gb * 2))
            echo "Tasks [$oom_tasks] OOM'd; retrying at ${mem_gb}gb"
        fi
        array_spec="$(union_csv "$oom_tasks" "$preempted_tasks")"
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

# Prints comma-separated raw healpix ids (at $NSIDE) whose disc overlaps any
# exposure in the given exposures .npy file, using the same query as
# dev/healpixFromExposures.py -- so we can tell exactly which healpix a batch
# of newly position-corrected exposures could affect, instead of assuming
# "any new exposure -> reprocess everything".
touched_healpix_for_exposures() {
    python -c "
import numpy as np
import healpy as hp
from astropy.table import Table
exposures = set(int(e) for e in np.load('$1'))
tab = Table.read('$DES_EXPOSURES')
expnum = np.asarray(tab['expnum'])
pole = np.asarray(tab['pole'])  # (N, 2) = [ra, dec] in degrees
mask = np.isin(expnum, list(exposures))
discs = [
    hp.query_disc($NSIDE, hp.ang2vec(np.radians(90.0 - dec), np.radians(ra)),
                  np.radians($DISC_RADIUS_DEG), inclusive=True)
    for ra, dec in pole[mask]
]
touched = np.unique(np.concatenate(discs)) if discs else np.array([], dtype=int)
print(','.join(map(str, touched)))
"
}

# Maps raw healpix ids (as printed by touched_healpix_for_exposures) to their
# array indices in $HEALPIX_NPY.
healpix_ids_to_indices() {
    python -c "
import sys
import numpy as np
healpix = np.load('$HEALPIX_NPY')
ids = set(int(x) for x in sys.argv[1].split(',') if x)
print(','.join(str(i) for i, hp in enumerate(healpix) if int(hp) in ids))
" "$1"
}

# Prints the sorted union of two comma-separated int lists (either may be empty).
union_csv() {
    python -c "
import sys
a = set(int(x) for x in sys.argv[1].split(',') if x)
b = set(int(x) for x in sys.argv[2].split(',') if x)
print(','.join(map(str, sorted(a | b))))
" "$1" "$2"
}

echo "== Stage 0: build healpix + exposure lists (${REGION_RADIUS_DEG} deg around Sculptor) =="
python "$REPO/dev/build_pmsculptor8deg_exposures.py" \
    --skims-path "$SKIMS_PATH" \
    --gpr-path "$GPR_PATH" \
    --des-exposures "$DES_EXPOSURES" \
    --ra "$SCULPTOR_RA" --dec "$SCULPTOR_DEC" --radius "$REGION_RADIUS_DEG" --nside "$NSIDE" \
    --healpix-output "$HEALPIX_NPY" \
    --exposures-output "$EXPOSURES_NPY" --complete-only

n_exposures=$(python -c "import numpy as np; print(len(np.load('$EXPOSURES_NPY')))")
n_healpix=$(python -c "import numpy as np; print(len(np.load('$HEALPIX_NPY')))")
echo "$n_healpix healpixels, $n_exposures exposures -> packing/PM array tasks"

echo "== Stage 1: position correction =="
POSCORR_EXPOSURES_NPY="$BASE/pmsculptor8deg_exposures_pending.npy"
n_pending=$(write_pending_exposures)
touched_healpix_ids=""
if [ "$n_pending" -eq 0 ]; then
    echo "  all $n_exposures exposures already position-corrected, skipping"
else
    # Chunk size must be big enough that ceil(n_pending/chunk) fits in one
    # stage's worth of array batches (MAX_ARRAY_TASKS each); multiPositionCorrection
    # reads --chunk-size exposures per task via --index/--chunk-size, indexing
    # into POSCORR_EXPOSURES_NPY (just the pending exposures, not the full list).
    POSCORR_CHUNK=$(( (n_pending + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
    n_pc=$(( (n_pending + POSCORR_CHUNK - 1) / POSCORR_CHUNK ))
    echo "$n_pending / $n_exposures exposures pending -> $n_pc position-correction array tasks (chunk=$POSCORR_CHUNK)"
    submit_stage "$REPO/dev/multiPositionCorrection_pmsculptor8deg.sh" "$(seq -s, 0 $(( n_pc - 1 )))" 4
    # A healpix already packed from a prior run may have had incomplete input
    # if any of these newly-corrected exposures overlap it -- reprocess just
    # those healpix, not every healpix in the region.
    touched_healpix_ids="$(touched_healpix_for_exposures "$POSCORR_EXPOSURES_NPY")"
fi

echo "== Stage 2: detection packing =="
# ponytail: a healpix with zero cleaned detections after cuts never produces
# this file and so is "pending" forever; harmless, it just reruns (and
# no-ops) on every resume.
packing_pending="$(pending_healpix_tasks "$BASE/HealpixDetectionCatalog/cleaned_detections_hp{hp:05d}.fits")"
if [ -n "$touched_healpix_ids" ]; then
    touched_idx="$(healpix_ids_to_indices "$touched_healpix_ids")"
    echo "  reprocessing healpix touched by newly position-corrected exposures too"
    packing_pending="$(union_csv "$packing_pending" "$touched_idx")"
fi
submit_stage "$REPO/dev/multiPacking_pmsculptor8deg.sh" "$packing_pending" 32

echo "== Stage 3: proper motion fit =="
pm_pending="$(pending_healpix_tasks \
    "$BASE/PMCatalog/real_PM_hp{hp:05d}.fits" \
    "$BASE/PMCatalog/injection_PM_hp{hp:05d}.fits")"
# Any healpix (re)packed just now has fresh input for the PM fit too.
pm_pending="$(union_csv "$pm_pending" "$packing_pending")"
submit_stage "$REPO/dev/multiPM_pmsculptor8deg.sh" "$pm_pending" 32

echo "== Stage 4: fast check =="
# ponytail: a healpix with no fast-check movers found never produces this
# file either (fastChecks.run_fast_checks only writes when pm_arr is
# non-empty), so same caveat as Stage 2 -- harmless, just reruns.
fastcheck_pending="$(pending_healpix_tasks \
    "$BASE/PMCatalog/real_hp{hp:05d}_fastcheck_movers.fits" \
    "$BASE/PMCatalog/injection_hp{hp:05d}_fastcheck_movers.fits")"
# Any healpix (re)fit just now has fresh input for the fast check too.
fastcheck_pending="$(union_csv "$fastcheck_pending" "$pm_pending")"
submit_stage "$REPO/dev/multiFastCheck_pmsculptor8deg.sh" "$fastcheck_pending" 10

echo "Done. Outputs under $BASE"
