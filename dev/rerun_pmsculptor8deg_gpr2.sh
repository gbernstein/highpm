#!/usr/bin/env bash
#SBATCH --job-name=RerunPMSculptor8deg
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=3-00:00:00
#SBATCH --output=/data8/shared/decampm/PMSculptor_8deg_garyb_rerun_%j.log
#SBATCH -p bhuv_compute
#SBATCH -q bhuv
# Orchestrator for rerunning PMSculptor8deg from scratch after a GPR2
# reprocessing. Runs on bhuv_compute (PreemptMode=OFF) only because it has to
# outlive the whole pipeline -- it's 1 idle CPU sitting in sbatch --wait; all
# actual pipeline work is submitted by run_pmsculptor8deg_pipeline.sh to low.
#
# Submit with a dependency on the GPR job so it only starts once that's gone:
#     sbatch --dependency=afterany:<gpr_jobid> dev/rerun_pmsculptor8deg_gpr2.sh
# (afterany, not afterok: a failed GPR task would leave afterok pending
# forever; the completeness gate below catches missing GPR output instead.)
#
# Steps:
#   1. Gate: rebuild the --complete-only healpix list against the current
#      GPR2/CAT and require it to cover every healpix of the existing run.
#      If not, stop -- nothing archived, nothing submitted.
#   2. Archive the existing run by renaming $BASE -> $ARCHIVE.
#   3. Record git/config/GPR provenance in $BASE/run_info.txt.
#   4. Run run_pmsculptor8deg_pipeline.sh end to end (from the repo root:
#      config.yaml's exclusion_regions_file is relative to the cwd, which
#      the stage jobs inherit).
set -euo pipefail

REPO=/home2/vwetzell/gitrepos/highpm
BASE=/data8/shared/decampm/PMSculptor_8deg_garyb
ARCHIVE=${ARCHIVE:-/data8/shared/decampm/PMSculptor_8deg_garyb_20260923}
GPR_CAT=/data8/shared/decampm/GPR2/CAT
DES_EXPOSURES="$REPO/../pixmappy/pixmappy/data/delveExposures.hdf5"

set +u
source /home2/vwetzell/.bashrc
conda activate pm
set -u

# Stage jobs are submitted from inside this job; don't let this allocation's
# SLURM_* environment leak into their sbatch calls.
ORCH_JOB_ID=${SLURM_JOB_ID:-none}
ORCH_HOST=$(hostname)
unset "${!SLURM_@}"

cd "$REPO"
echo "Orchestrator job $ORCH_JOB_ID on $ORCH_HOST started $(date)"

echo "== Gate: GPR2 completeness for the existing run's healpix =="
if [ -e "$ARCHIVE" ]; then
    echo "ERROR: archive target $ARCHIVE already exists; refusing to overwrite" >&2
    exit 1
fi
if [ ! -f "$BASE/pmsculptor8deg_healpix.npy" ]; then
    echo "ERROR: no existing run at $BASE to archive/compare against" >&2
    exit 1
fi
GATE_DIR=$(mktemp -d)
trap 'rm -rf "$GATE_DIR"' EXIT
python "$REPO/dev/build_pmsculptor8deg_exposures.py" \
    --skims-path "/data8/shared/decampm/[GRIZ]/" \
    --gpr-path "$GPR_CAT/[griz]/" \
    --des-exposures "$DES_EXPOSURES" \
    --radius 8.0 --nside 32 \
    --healpix-output "$GATE_DIR/hp.npy" \
    --exposures-output "$GATE_DIR/exp.npy" --complete-only
python - "$BASE/pmsculptor8deg_healpix.npy" "$GATE_DIR/hp.npy" <<'EOF'
import sys
import numpy as np
old = set(np.load(sys.argv[1]).tolist())
new = set(np.load(sys.argv[2]).tolist())
missing = sorted(old - new)
print(f"previous run: {len(old)} healpix; GPR2-complete now: {len(new)} healpix")
if missing:
    sys.exit(f"ERROR: {len(missing)} healpix from the previous run are not GPR2-complete: {missing}")
print(f"all {len(old)} previous healpix complete" + (f"; {len(new - old)} new: {sorted(new - old)}" if new - old else ""))
EOF

echo "== Archive: $BASE -> $ARCHIVE =="
mv "$BASE" "$ARCHIVE"
mkdir -p "$BASE"

{
    echo "started: $(date)"
    echo "orchestrator job: $ORCH_JOB_ID on $ORCH_HOST"
    echo "previous run archived to: $ARCHIVE"
    echo "git HEAD: $(git -C "$REPO" rev-parse HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
    echo "git status (tracked):"
    git -C "$REPO" status --short --untracked-files=no | sed 's/^/    /'
    echo "GPR2 CAT files per band:"
    for b in g r i z; do
        echo "    $b: $(find "$GPR_CAT/$b" -name '*.fits' | wc -l)"
    done
    echo "GPR2 CAT mtime range: $(find "$GPR_CAT" -name '*.fits' -printf '%TY-%Tm-%Td %TH:%TM\n' | sort | sed -n '1p;$p' | paste -sd'|' | sed 's/|/ -> /')"
    echo "config ($REPO/config/config.yaml):"
    sed 's/^/    /' "$REPO/config/config.yaml"
} > "$BASE/run_info.txt"

echo "== Pipeline =="
bash "$REPO/dev/run_pmsculptor8deg_pipeline.sh" 2>&1 | tee -a "$BASE/pipeline_run.log"
echo "Orchestrator finished $(date)"
