#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

HF_DATASET_REPO="${HF_DATASET_REPO:-skboy/eyebench-processed-folds}"
DATA_TASK="${DATA_TASK:-PoTeC_RC}"
BASELINE_MODELS="${BASELINE_MODELS:-MAG}"
FOLDS="${FOLDS:-0 1 2 3}"
PRECISION="${PRECISION:-THIRTY_TWO_TRUE}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_ENTITY="${WANDB_ENTITY:-EyeRead}"
WANDB_PROJECT="${WANDB_PROJECT:-EyeBench_${DATA_TASK}_baseline_$(date +%Y%m%d_%H%M)}"
RUN_TAG="${RUN_TAG:-baseline_${DATA_TASK}_$(echo "$BASELINE_MODELS" | tr ' ' '_' | tr ',' '_')_$(date +%Y%m%d_%H%M)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/outputs/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$ROOT_DIR/results/raw"
echo "$RUN_ROOT" | tee "$LOG_DIR/latest_${DATA_TASK}_baseline_run_dir.txt"

python run_commands/utils/download_processed_folds_from_hf.py \
  "$HF_DATASET_REPO" \
  --local-dir "$ROOT_DIR" \
  --datasets PoTeC

for MODEL in $BASELINE_MODELS; do
  MODEL_RUN_NAME="${RUN_TAG}_${MODEL}"
  MODEL_RUN_DIR="$RUN_ROOT/$MODEL_RUN_NAME"
  mkdir -p "$MODEL_RUN_DIR"

  for FOLD in $FOLDS; do
    FOLD_DIR="$MODEL_RUN_DIR/fold_index=${FOLD}"
    echo "Training model=$MODEL data=$DATA_TASK fold=$FOLD output=$FOLD_DIR"

    python src/run/single_run/train.py \
      +trainer=TrainerDL \
      "+model=${MODEL}" \
      "+data=${DATA_TASK}" \
      "data.fold_index=${FOLD}" \
      trainer.devices=1 \
      "trainer.precision=${PRECISION}" \
      "trainer.num_workers=${NUM_WORKERS}" \
      "trainer.wandb_entity=${WANDB_ENTITY}" \
      "trainer.wandb_project=${WANDB_PROJECT}" \
      "trainer.wandb_job_type=${MODEL}_${DATA_TASK}_fold${FOLD}" \
      "hydra.run.dir='${FOLD_DIR}'" \
      2>&1 | tee "$LOG_DIR/train_${MODEL}_${DATA_TASK}_fold${FOLD}.log"
  done

  echo "Evaluating model=$MODEL eval_path=$MODEL_RUN_DIR"
  python src/run/single_run/test_dl.py \
    "eval_path=${MODEL_RUN_DIR}" \
    2>&1 | tee "$LOG_DIR/eval_${MODEL}_${DATA_TASK}.log"
done

ARCHIVE="${RUN_TAG}_results.tar.gz"
ARCHIVE_PATH="$ROOT_DIR/$ARCHIVE"
python - "$ROOT_DIR" "$RUN_ROOT" "$RUN_TAG" "$ARCHIVE_PATH" <<'PY'
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
run_tag = sys.argv[3]
archive_path = Path(sys.argv[4])

paths = [root / "logs", run_root]
raw_root = root / "results" / "raw"
if raw_root.exists():
    paths.extend(sorted(raw_root.glob(f"{run_tag}_*")))

with tarfile.open(archive_path, "w:gz") as tar:
    for path in paths:
        if path.exists():
            tar.add(path, arcname=str(path.relative_to(root)))
PY

echo "Done."
echo "Run root: $RUN_ROOT"
echo "Results archive: $ARCHIVE_PATH"
