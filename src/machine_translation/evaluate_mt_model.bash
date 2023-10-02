set -e
set -u
set -o pipefail

. machine_translation/functions.bash

usage() {
  echo "$0 <data-dir> <train-dataset> <test-dataset> <vocab-type> <model-dir> ..."
}

data_dir=${1-}
train_dataset=${2-}
test_dataset=${3-}
vocab_type=${4-}
model_dir=${5-}
if ! shift 5; then
  usage >&2
  exit 1
fi
extra_args=("$@")

train_dir=$data_dir/$train_dataset
get_vocab_args "$train_dir" "$vocab_type" vocab_args

result_dir=$model_dir/eval/$test_dataset
mkdir -p "$result_dir"
python sequence_to_sequence/translate.py \
  --input "$train_dir"/datasets/"$test_dataset"/source."$vocab_type".prepared \
  --beam-size 4 \
  --max-target-length 256 \
  "${vocab_args[@]}" \
  --load-model "$model_dir" \
  "${extra_args[@]}" \
  | tee "$result_dir"/translations.seg \
  | spm_decode --model "$train_dir"/sentencepiece.model \
  | tee "$result_dir"/translations.detok \
  | sacrebleu "$train_dir"/datasets/"$test_dataset"/target.filtered \
      --metrics bleu chrf \
      --width 8 \
  | tee "$result_dir"/scores.json
