ROOT_DIR=$(cd "$(dirname "$BASH_SOURCE")"/../.. && pwd)
. "$ROOT_DIR"/experiments/include.bash
LOG_DIR=$HOME/Private/logs/2023-06-20/limited-language-modeling-random-batch-size
LM_DATA_DIR=$DATA_DIR/language-modeling

MODELS=( \
  5 \
  2.superposition-511.2 \
  2.nondeterministic-3-3-10.2 \
)
TRIALS=({1..20})

get_output_name() {
  local model=$1
  local trial_no=$2
  local result=$LOG_DIR/$model/$trial_no
  printf '%s' "$result"
}

format_model_name() {
  local model=$1
  local result
  case $model in
    *superposition*) result='\LabelTransformerSuperposition{}' ;;
    *nondeterministic*) result='\LabelTransformerNondeterministic{}' ;;
    *) result='\LabelTransformerBaseline{}' ;;
  esac
  printf '%s' "$result"
}
