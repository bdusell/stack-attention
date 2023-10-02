ROOT_DIR="$(dirname "$BASH_SOURCE")"/../..
. "$ROOT_DIR"/experiments/include.bash
LOG_DIR=$HOME/Private/logs/2023-09-22/cfl
TASKS=( \
  marked-reversal \
  unmarked-reversal \
  padded-reversal \
  dyck \
  hardest-cfl \
)
MODELS=( \
  lstm-100 \
  lstm-93-superposition-10 \
  lstm-64-nondeterministic-x-x-5 \
  transformer-32-5 \
  transformer-32-2.superposition-32.2 \
  transformer-28-2.nondeterministic-x-x-5.2 \
)
TRIALS=({1..10})

format_model_name() {
  local name=$1
  local result
  if [[ $name =~ ^lstm-([0-9]+)$ ]]; then
    result='\LabelLSTM{}'
  elif [[ $name =~ ^lstm-([0-9]+)-superposition-([0-9]+)$ ]]; then
    result='\LabelLSTMSuperposition{}'
  elif [[ $name =~ ^lstm-([0-9]+)-nondeterministic-(.+)$ ]]; then
    result='\LabelLSTMNondeterministic{}'
  elif [[ $name =~ ^transformer-([0-9]+)-(.+)$ ]]; then
    local layers=${BASH_REMATCH[2]}
    if [[ $layers =~ ^[0-9]+$ ]]; then
      result='\LabelTransformerBaseline{}'
    elif [[ $layers =~ .*superposition.* ]]; then
      result='\LabelTransformerSuperposition{} (Ours)'
    elif [[ $layers =~ .*nondeterministic.* ]]; then
      result='\LabelTransformerNondeterministic{} (Ours)'
    else
      return 1
    fi
  else
    return 1
  fi
  echo -n "$result"
}

get_output_directory() {
  local task=$1
  local model=$2
  local trial_no=$3
  local result=$LOG_DIR/$model/$task/$trial_no
  echo -n "$result"
}

format_task_name() {
  local name=$1
  local result
  case $name in
    marked-reversal) result='\MarkedReversal{}' ;;
    unmarked-reversal) result='\UnmarkedReversal{}' ;;
    padded-reversal) result='\PaddedReversal{}' ;;
    dyck) result='Dyck' ;;
    hardest-cfl) result='Hardest CFL' ;;
    *) return 1 ;;
  esac
  echo -n "$result"
}

get_test_data_file() {
  local task=$1
  local result=$HOME/Private/logs/2022-12-12/test-sets/$task-test-data.pt
  echo -n "$result"
}
