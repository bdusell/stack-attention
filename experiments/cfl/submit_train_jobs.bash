set -e
set -u
set -o pipefail

. "$(dirname "$BASH_SOURCE")"/include.bash

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ $model = *nondeterministic* ]]; then
      device=gpu
    else
      device=cpu
    fi
    for trial_no in "${TRIALS[@]}"; do
      bash experiments/submit-job.bash \
        "$task+$model+$trial_no" \
        "$LOG_DIR"/outputs \
        "$device" \
        bash cfl_language_modeling/train_model_on_task.bash \
          "$LOG_DIR" \
          "$model" \
          "$task" \
          "$trial_no" \
          --no-progress
    done
  done
done
