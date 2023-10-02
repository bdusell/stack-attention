ROOT_DIR=$(cd "$(dirname "$BASH_SOURCE")"/../.. && pwd)
. "$ROOT_DIR"/experiments/include.bash
LOG_DIR=$HOME/Private/logs/2023-09-22/machine-translation
MT_DATA_DIR=$DATA_DIR/machine-translation
TRIALS=({1..5})

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
