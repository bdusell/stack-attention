set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

dataset=europarl-v7-de-en/max-length/150/sample-size/100000

for trial_no in {1..5}; do
  for model_size in large medium small; do
    for model_type in transformer superposition nondeterministic; do
      bash experiments/submit-job.bash \
        subsampled-mt+"$model_size"+"$model_type"+"$trial_no" \
        "$LOG_DIR"/outputs \
        gpu \
        bash machine_translation/train_and_evaluate_mt_model.bash \
          "$LOG_DIR"/"$model_size"/"$model_type"/"$trial_no" \
          "$MT_DATA_DIR" \
          "$dataset" \
          shared \
          "$model_size"-"$model_type" \
          num-tokens \
          --no-progress
    done
  done
done
