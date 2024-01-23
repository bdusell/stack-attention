set -e
set -u
set -o pipefail

. "$(dirname "$BASH_SOURCE")"/include.bash

cd "$ROOT_DIR"/src

output_dir=$FIGURES_DIR/tex/heatmaps
mkdir -p "$output_dir"

model=transformer-32-2.superposition-32.2
for task in marked-reversal dyck; do
  model_dirs=()
  for trial_no in "${TRIALS[@]}"; do
    model_dirs+=("$(get_output_directory "$task" "$model" "$trial_no")")
  done
  bash cfl_language_modeling/plot_stack_attention_heatmap_for_task.bash \
    "$output_dir" \
    "$task" \
    superposition \
    "${model_dirs[@]}"
done
