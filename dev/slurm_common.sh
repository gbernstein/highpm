# Sourced by the multi*.sh SLURM array scripts. Caller sets $AUXDIR first.
# Activates the conda env, pins thread counts to the allocation, starts the
# timer, and defines finish() to log elapsed time. Call finish at script end.

source /home2/vwetzell/.bashrc
conda activate pm

for lib in OMP OPENBLAS MKL NUMEXPR BLIS; do
    export ${lib}_NUM_THREADS=$SLURM_CPUS_PER_TASK
done

echo "Running on host: $(hostname) at $(date)"
start_time=$(date +%s)

finish() {
    local elapsed=$(( $(date +%s) - start_time ))
    mkdir -p "$AUXDIR"
    echo "Index ${SLURM_ARRAY_TASK_ID} finished at $(date), total time: ${elapsed}s" \
        >> "$AUXDIR/time_all.txt"
    echo "Task ${SLURM_ARRAY_TASK_ID} completed at $(date), runtime: ${elapsed}s"
}
