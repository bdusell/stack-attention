set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 [options] <train-dir> <valid-dir> [<other-dir> ...]

  --max-length <int>
    If given, filter the validation data and other datasets so that all source
    and target sequences have a maximum length of <int> in normalized Unicode
    characters.
"
}

max_length=
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-length) shift; max_length=$1 ;;
    *) break ;;
  esac
  shift
done

train_dir=${1-}
valid_dir=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi
other_dirs=("$@")

get_dataset_subdir() {
  local dataset_dir=$1
  local name=$(basename "$dataset_dir")
  local result=$train_dir/datasets/$name
  printf '%s' "$result"
}

# Create sub-directories for each non-training dataset.
for dir in "$valid_dir" "${other_dirs[@]}"; do
  mkdir -p "$(get_dataset_subdir "$dir")"
done

filter_args=()
if [[ $max_length ]]; then
  filter_args+=(--max-length "$max_length")
fi

# Normalize Unicode in the validation and test sets. It seems that sacrebleu
# does *not* automatically apply Unicode normalization.
# The normalized files could be stored in the original dataset directories
# since they do not change based on the training data, but we store them under
# the training data directory instead. This can create redundant copies, but
# it eliminates the possibility of race conditions when multiple preprocessing
# jobs are running concurrently.
for dir in "$valid_dir" "${other_dirs[@]}"; do
  python machine_translation/filter.py \
    --inputs \
      <(python machine_translation/normalize_unicode.py < "$dir"/source.raw) \
      <(python machine_translation/normalize_unicode.py < "$dir"/target.raw) \
    --outputs "$(get_dataset_subdir "$dir")"/{source,target}.filtered \
    "${filter_args[@]}"
done

# Train the BPE model.
num_pairs=$(wc -l "$train_dir"/source.filtered | cut -d ' ' -f 1)
if [[ $((2 * $num_pairs)) -gt 10000000 ]]; then
  echo "error: sentencepiece isn\'t configured to handle more than 10M lines" >&2
  return 1
fi
# The data has already been normalized, so don't normalize it again. In fact,
# sentencepiece doesn't fully implement Unicode normalization anyway.
# See https://github.com/google/sentencepiece/blob/master/doc/normalization.md#use-pre-defined-normalization-rule
# With --hard_vocab_limit=false, SentencePiece will not throw an error if
# there are fewer possible tokens than --vocab_size.
spm_train \
  --input="$train_dir"/source.filtered,"$train_dir"/target.filtered \
  --model_prefix="$train_dir"/sentencepiece \
  --model_type=bpe \
  --vocab_size=32000 \
  --hard_vocab_limit=false \
  --character_coverage=1 \
  --num_threads "$(nproc)" \
  --unk_id=0 \
  --bos_id=-1 \
  --eos_id=-1 \
  --pad_id=-1 \
  --unk_piece='<unk>' \
  --normalization_rule_name=identity \
  --minloglevel 1

SPM_VOCAB_THRESHOLD=50

for side in source target; do
  # Tokenize the training data using learned BPE model.
  # Although the BPE model is trained on the union of both languages, when
  # segmenting each language, make sure to only output segments that occur in
  # that language.
  # See https://github.com/google/sentencepiece#vocabulary-restriction
  # First, generate the vocabulary for the training data of only this side.
  spm_encode \
    --model="$train_dir"/sentencepiece.model \
    --generate_vocabulary \
    --input="$train_dir"/"$side".filtered \
    --output="$train_dir"/"$side".spmvocab
  # Now tokenize the training data for this side, using the vocabulary just
  # generated.
  spm_encode \
    --model="$train_dir"/sentencepiece.model \
    --vocabulary="$train_dir"/"$side".spmvocab \
    --vocabulary_threshold="$SPM_VOCAB_THRESHOLD" \
    --input="$train_dir"/"$side".filtered \
    --output="$train_dir"/"$side".seg
  rm "$train_dir"/"$side".filtered
  # Now tokenize additional files, such as the validation and test data. The
  # target side of the test set does not need to be preprocessed.
  dirs=("$valid_dir")
  if [[ $side = source ]]; then
    dirs+=("${other_dirs[@]}")
  fi
  for dir in "${dirs[@]}"; do
    dest_dir=$(get_dataset_subdir "$dir")
    spm_encode \
      --model="$train_dir"/sentencepiece.model \
      --vocabulary="$train_dir"/"$side".spmvocab \
      --vocabulary_threshold="$SPM_VOCAB_THRESHOLD" \
      --input="$dest_dir"/"$side".filtered \
      --output="$dest_dir"/"$side".seg
  done
  rm "$train_dir"/"$side".spmvocab
done

prepare_source_args=()
for dir in "${other_dirs[@]}"; do
  prepare_source_args+=(--prepare-source "$(get_dataset_subdir "$dir")")
done

bash sequence_to_sequence/prepare_data.bash \
  --training-data "$train_dir" \
  --prepare-both "$(get_dataset_subdir "$valid_dir")" \
  "${prepare_source_args[@]}" \
  --always-allow-unk \
  --shared-vocabulary
