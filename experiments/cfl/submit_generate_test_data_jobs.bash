set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

output_dir=$LOG_DIR/test-sets
mkdir -p "$output_dir"

for task in "${TASKS[@]}"; do
  bash experiments/submit-job.bash \
    "generate-test-set-$task" \
    "$LOG_DIR"/outputs \
    cpu \
    bash cfl_language_modeling/generate_test_set.bash \
      "$(get_test_data_file "$task")" \
      "$task"
done
