set -e
set -u
set -o pipefail

usage() {
  echo "$0 <data-file> <model-name> <model-dirs>... -- <test-args>...

  <data-file>       The .pt file containing the test data.
  <model-name>      The name of the model.
  <model-dirs>...   Trained model directories, and any other arguments to
                    print_best.py.
  <test-args>...    Any arguments to test.py.
"
}

data_file=${1-}
model_name=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi
print_best_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --) shift; break ;;
    *) print_best_args+=("$1")
  esac
  shift
done
test_args=("$@")

best_model=$(python utils/print_best.py "${print_best_args[@]}" | cut -f 1)
if [[ $best_model ]]; then
  echo "best model: $best_model"
  bash cfl_language_modeling/test_model.bash "$data_file" "$best_model" "${test_args[@]}"
else
  echo 'error: no models to test'
  exit 1
fi
