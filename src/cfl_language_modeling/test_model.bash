set -e
set -u
set -o pipefail

usage() {
  echo "$0 <data-file> <model-dir> ...

  <data-file>   The .pt file containing the test data.
  <model-dir>   The directory containing the trained model to evaluate.
  ...           Any other arguments to pass to test.py.
"
}

data_file=${1-}
model_dir=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi
extra_args=("$@")

python cfl_language_modeling/test.py \
  --data "$data_file" \
  --load-model "$model_dir" \
  --output "$model_dir"/test \
  "${extra_args[@]}"
