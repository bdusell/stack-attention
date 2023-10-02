set -e
set -u
set -o pipefail

. machine_translation/functions.bash

usage() {
  echo "$0 <data-dir> <train-dataset> <test-dataset> <model-dir> ..."
}

data_dir=${1-}
train_dataset=${2-}
test_dataset=${3-}
model_dir=${4-}
if ! shift 4; then
  usage >&2
  exit 1
fi
extra_args=("$@")

train_dir=$data_dir/$train_dataset

result_dir=$model_dir/eval/$test_dataset
mkdir -p "$result_dir"
python language_modeling/evaluate.py \
  --input "$train_dir"/datasets/"$test_dataset"/main.prepared \
  --vocabulary "$train_dir"/main.vocab \
  --load-model "$model_dir" \
  "${extra_args[@]}" \
  | tee "$result_dir"/scores.json
