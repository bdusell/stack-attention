set -e
set -u
set -o pipefail

. "$(dirname "$BASH_SOURCE")"/include.bash

cd src

printf 'Task'
for model in "${MODELS[@]}"; do
  printf ' & %s' "$(format_model_name "$model" | sed 's/ (Ours)//')"
done
echo ' \\'
echo '\midrule'
for task in "${TASKS[@]}"; do
  printf '%s' "$(format_task_name "$task")"
  for model in "${MODELS[@]}"; do
    model_dir=$(get_output_directory "$task" "$model" 1)
    count=$(python cfl_language_modeling/print_model_parameter_count.py --load-model "$model_dir")
    printf ' & %s' "$count"
  done
  echo ' \\'
done
