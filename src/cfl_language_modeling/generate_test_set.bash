set -e
set -u
set -o pipefail

. cfl_language_modeling/functions.bash

usage() {
  echo "$0 <output-file> <task> ...

Arguments:

  <output-file>   Output file.
  <task>          Name of the task.
"
}

output_file=${1-}
task=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

case $task in
  marked-reversal|unmarked-reversal|padded-reversal|dyck|marked-reverse-and-copy) symbol_types=2 ;;
  *) symbol_types=x ;;
esac
get_cfl_task_args "$task" "$symbol_types" task_args
poetry run python cfl_language_modeling/generate_test_data.py \
  --test-length-range 40:100 \
  --test-data-size 100 \
  --test-batch-size 100 \
  --test-data-seed 0 \
  "${task_args[@]}" \
  --output "$output_file"
