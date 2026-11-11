#!/bin/bash
#SBATCH --job-name=hosvd_llama
#SBATCH --partition=cpu-redhat            # Amarel's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=8          # CPU cores for data loading             # request 1 GPU
#SBATCH --mem=64G                  # RAM (not VRAM — system memory)
#SBATCH --time=2:00:00            # 12 hours, adjust based on your estimate
#SBATCH --output=out.txt   # stdout — %j is the job ID
#SBATCH --error=err.txt


set -euo pipefail
 
# Under sbatch the script is copied to a spool dir, so use the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
 
SCRIPT="${measure_loss.py}"       
MATRIX="${1:-q_proj}"
RANKS="${2:-16,512,512}"
EXTRA="${3:-}"
OUT_JSON="results_${SLURM_JOB_ID:-local}.json"
 
# ---- Environment (CHANGE to match your setup) ----
# module purge
# module load python/3.11 cuda/12.1
# source ~/venvs/hosvd/bin/activate       # needs torch, safetensors, huggingface_hub, requests
# export HF_TOKEN="hf_xxx"                # or run `huggingface-cli login` once
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lora_train
export HF_HOME=/scratch/kz498/.cache/huggingface
mkdir -p "$HF_HOME" logs
 
echo "Host:   $(hostname)"
echo "Start:  $(date)"
echo "Matrix: $MATRIX | Ranks: $RANKS"
 
ARGS=(--matrix "$MATRIX" --save "$OUT_JSON")
if [[ "$RANKS" == "sweep" ]]; then
    ARGS+=(--sweep)
else
    ARGS+=(--ranks "$RANKS")
fi
[[ "$EXTRA" == "--no-direct" ]] && ARGS+=(--no-direct)
 
python -u "$SCRIPT" "${ARGS[@]}"
 
echo "Done:   $(date)"