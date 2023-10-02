set -e
set -u
set -o pipefail

. "$(dirname "$BASH_SOURCE")"/include.bash

cd "$ROOT_DIR"/src

format_model_name() {
  local size=$1
  local type=$2
  local result
  case $type in
    *superposition*) result='\LabelTransformerSuperposition{}' ;;
    *nondeterministic*) result='\LabelTransformerNondeterministic{}' ;;
    *) result='\LabelTransformerBaseline{}' ;;
  esac
  printf '%s' "$result"
}

for model_size in small medium large; do
  model_args=()
  for model_type in transformer superposition nondeterministic; do
    model_args+=(--label "$(format_model_name "$model_size" "$model_type")" --inputs)
    for trial_no in "${TRIALS[@]}"; do
      model_args+=("$LOG_DIR"/"$model_size"/"$model_type"/"$trial_no")
    done
  done
  python machine_translation/print_table.py "${model_args[@]}"
done
