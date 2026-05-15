#!/bin/bash
# =============================================================================
# Self-contained setup + submit for WTA-ICL retraining on MIT Engaging / ORCD.
#
# Run this ON an Engaging login node, from anywhere inside the repo:
#     bash ICL/engaging/run_on_engaging.sh              # build env + submit ONE test job
#     bash ICL/engaging/run_on_engaging.sh --sweep      # build env + submit the sweep
#     bash ICL/engaging/run_on_engaging.sh --setup-only # just build the env
#
# It auto-detects your SLURM account and builds its own venv with torch,
# so you do not need to supply any account name or env path by hand.
# =============================================================================
set -uo pipefail

# --- locate the repo / ICL dir relative to this script -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$ICL_DIR/.." && pwd)"

# --- configuration (override via env vars if you like) -----------------------
ENV_DIR="${WTA_ENV_DIR:-$HOME/wta-icl-env}"          # where the venv is built
RESULTS_BASE="${WTA_RESULTS_BASE:-$ICL_DIR/results/wta_n_nodes_rhoall_seed}"
PARTITION="${WTA_PARTITION:-mit_normal}"
TIME_LIMIT="${WTA_TIME:-04:00:00}"

echo "Repo:        $REPO_DIR"
echo "ICL dir:     $ICL_DIR"
echo "Env dir:     $ENV_DIR"
echo "Results dir: $RESULTS_BASE"
echo "Partition:   $PARTITION"
echo

# --- 1. detect SLURM account -------------------------------------------------
ACCOUNT="$(sacctmgr -nP show assoc where user="$USER" format=account 2>/dev/null \
            | sort -u | grep -v '^$' | head -1)"
if [ -n "$ACCOUNT" ]; then
    echo "SLURM account (auto-detected): $ACCOUNT"
    ACCOUNT_LINE="#SBATCH --account=$ACCOUNT"
else
    echo "SLURM account: none detected -> omitting --account (SLURM default applies)"
    ACCOUNT_LINE="# (no --account flag; SLURM default association used)"
fi
echo

# --- 2. build a venv with torch (once) ---------------------------------------
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "Building Python venv at $ENV_DIR ..."
    # try to load a module-provided python; harmless if none of these exist
    module load python3 2>/dev/null || module load python 2>/dev/null \
        || module load anaconda 2>/dev/null || module load miniforge 2>/dev/null || true
    PYBIN="$(command -v python3 || command -v python)"
    echo "  base python: $PYBIN ($("$PYBIN" --version 2>&1))"
    "$PYBIN" -m venv "$ENV_DIR" || { echo "ERROR: venv creation failed"; exit 1; }
    "$ENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    echo "  installing numpy + scipy ..."
    "$ENV_DIR/bin/python" -m pip install --quiet numpy scipy \
        || { echo "ERROR: numpy/scipy install failed"; exit 1; }
    echo "  installing torch (CPU build) ..."
    "$ENV_DIR/bin/python" -m pip install --quiet torch \
        --index-url https://download.pytorch.org/whl/cpu \
        || { echo "ERROR: torch install failed"; exit 1; }
else
    echo "Reusing existing venv at $ENV_DIR"
fi
"$ENV_DIR/bin/python" -c \
    "import torch,numpy,scipy; print('  env OK: torch',torch.__version__)" \
    || { echo "ERROR: env check failed"; exit 1; }
echo

if [ "${1:-}" = "--setup-only" ]; then
    echo "Setup complete. Re-run without --setup-only to submit jobs."
    exit 0
fi

# --- 3. choose which jobs to run ---------------------------------------------
# Each entry is "n_nodes rho_all seed".
if [ "${1:-}" = "--sweep" ]; then
    JOBS=( "8 1.0 30" "12 1.0 20" )      # <-- edit/extend for a wider sweep
    echo "Mode: sweep (${#JOBS[@]} jobs)"
else
    JOBS=( "8 1.0 30" )                  # default: single test job
    echo "Mode: single test job (pass --sweep for the full set)"
fi
echo

# --- 4. write + submit one SLURM script per job ------------------------------
for triple in "${JOBS[@]}"; do
    read -r N R S <<< "$triple"
    OUT="$RESULTS_BASE/${N}_${R}_${S}"
    mkdir -p "$OUT"
    JOB="$OUT/job_${N}_${R}_${S}.sh"
    cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --job-name=wta_${N}_${R}_${S}
#SBATCH --partition=$PARTITION
$ACCOUNT_LINE
#SBATCH --output=$OUT/training_batch.out
#SBATCH --error=$OUT/training_batch.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
export OMP_NUM_THREADS=4
source $ENV_DIR/bin/activate
cd $ICL_DIR
python run_icl_wta.py --param1 $N --param2 $R --param3 $S --output $OUT
EOF
    echo "Submitting $JOB"
    sbatch "$JOB"
done

echo
echo "============================================================"
echo "Submitted. Monitor with:   squeue -u $USER"
echo "Logs stream to:            $RESULTS_BASE/<n>_<r>_<s>/training_batch.out"
echo
echo "When a run finishes, verify it with:"
echo "  $ENV_DIR/bin/python $ICL_DIR/verify_checkpoints.py \\"
echo "      $RESULTS_BASE/8_1.0_30"
echo "============================================================"
