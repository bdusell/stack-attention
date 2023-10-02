set -e
set -u
set -o pipefail

. cfl_language_modeling/functions.bash

usage() {
  echo "$0 <output-dir> <task> <symbol-types> <trial-no> ...

Arguments:

  <output-dir>    Output directory for logs and model parameters.
  <task>          Name of the task.
  <symbol-types>  Number of symbol types to use in the task, where applicable.
  <trial-no>      Number for this random restart.

Extra options:
  --device <device>
  --no-progress
  ...             All arguments for initializing and running the model, except
                  --hidden-units and --layers.
"
}

output_dir=${1-}
task=${2-}
symbol_types=${3-}
trial_no=${4-}
if ! shift 4; then
  usage >&2
  exit 1
fi
extra_args=("$@")

learning_rate=$(python utils/random_sample.py --log 0.0005 0.01)

get_cfl_task_args "$task" "$symbol_types" task_args

python cfl_language_modeling/train.py \
  --output "$output_dir"/"$trial_no" \
  --save-model \
  --train-length-range 40:80 \
  --train-data-size 10000 \
  --batch-size 10 \
  --valid-length-range 40:80 \
  --valid-data-size 1000 \
  --valid-batch-size 10 \
  "${task_args[@]}" \
  --hidden-units 20 \
  --init-scale 0.1 \
  --optimizer Adam \
  --learning-rate "$learning_rate" \
  --learning-rate-patience 5 \
  --learning-rate-decay 0.9 \
  --gradient-clipping 5 \
  --epochs 200 \
  --early-stopping-patience 10 \
  "${extra_args[@]}"
