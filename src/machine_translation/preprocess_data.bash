set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 [options] <train-dir> <valid-dir> [<other-dir> ...]

  Options:
    --max-length <int>
      Limit all datasets to examples where both the source and target sequences
      are no longer than this many normalized Unicode characters.
    --sample-size <int>
      Randomly sub-sample this many examples for the training set.
"
}

max_length=
sample_size=
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-length) shift; max_length=$1 ;;
    --sample-size) shift; sample_size=$1 ;;
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

# Check that all of the .raw files exist.
for dir in "$train_dir" "$valid_dir" "${other_dirs[@]}"; do
  for side in source target; do
    file=$dir/$side.raw
    if [[ ! -f $file ]]; then
      echo "file $file is missing" >&2
      exit 1
    fi
  done
done

output_train_dir=$train_dir
filter_args=()
preprocess_args=()
if [[ $max_length ]]; then
  output_train_dir+=/max-length/$max_length
  filter_args+=(--max-length "$max_length")
  preprocess_args+=(--max-length "$max_length")
fi
if [[ $sample_size ]]; then
  output_train_dir+=/sample-size/$sample_size
  filter_args+=(--sample-size "$sample_size")
fi
mkdir -p "$output_train_dir"

# Normalize Unicode and filter out bad examples in the training data.
python machine_translation/filter.py \
  --inputs \
    <(python machine_translation/normalize_unicode.py < "$train_dir"/source.raw) \
    <(python machine_translation/normalize_unicode.py < "$train_dir"/target.raw) \
  --outputs "$output_train_dir"/{source,target}.filtered \
  --max-length-ratio 4 \
  "${filter_args[@]}"

# Run BPE and prepare all data.
bash machine_translation/preprocess_filtered_data.bash \
  "${preprocess_args[@]}" \
  "$output_train_dir" \
  "$valid_dir" \
  "${other_dirs[@]}"
