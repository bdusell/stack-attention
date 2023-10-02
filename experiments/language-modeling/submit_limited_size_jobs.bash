set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

dataset=dyer-ptb

for trial_no in "${TRIALS[@]}"; do
  for model in "${MODELS[@]}"; do
    bash experiments/submit-job.bash \
      limited-lm-random+"$model"+"$trial_no" \
      "$LOG_DIR"/outputs \
      gpu \
      bash language_modeling/train_and_evaluate_language_model.bash \
        "$LOG_DIR"/"$model"/"$trial_no" \
        "$LM_DATA_DIR" \
        "$dataset" \
        "$model" \
        --no-progress
  done
done
