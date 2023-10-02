set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 [options]

  Options:

  --training-data <dir>
    Directory of data used as training data.
  --prepare-both <dir>
    Directory of data where both source and target data should be prepared.
    Can be passed multiple times.
  --prepare-source <dir>
    Directory of data where only source data should be prepared. Can be passed
    multiple times.
  --prepare-target <dir>
    Directory of data where only target data should be prepared. Can be passed
    multiple times.
  --always-allow-unk

  --shared-vocabulary
    Generate files where the vocabulary of the source and target is shared.
  --separate-vocabulary
    Generate files where the vocabularies of the source and target are
    separate.

  Note that both --shared-vocabulary and --separate-vocabulary can be used at
  the same time; they generate different sets of files.
"
}

training_data_dir=
prepare_source_dirs=()
prepare_target_dirs=()
always_allow_unk=false
generate_shared_vocabulary=false
generate_separate_vocabulary=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --training-data) shift; training_data_dir=$1 ;;
    --prepare-both)
      shift
      prepare_source_dirs+=("$1")
      prepare_target_dirs+=("$1")
      ;;
    --prepare-source) shift; prepare_source_dirs+=("$1") ;;
    --prepare-target) shift; prepare_target_dirs+=("$1") ;;
    --always-allow-unk) always_allow_unk=true ;;
    --shared-vocabulary) generate_shared_vocabulary=true ;;
    --separate-vocabulary) generate_separate_vocabulary=true ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ ! $training_data_dir ]]; then
  usage >&2
  exit 1
fi

if ! { $generate_shared_vocabulary || $generate_separate_vocabulary; }; then
  usage >&2
  echo 'neither --shared-vocabulary nor --separate-vocabulary was passed' >&2
  exit 1
fi

if $generate_shared_vocabulary; then
  args=()
  for side in source target; do
    args+=(--"$side"-training-files "$training_data_dir"/"$side".{seg,shared.prepared})
    if [[ $side = source ]]; then
      dirs=("${prepare_source_dirs[@]}")
    else
      dirs=("${prepare_target_dirs[@]}")
    fi
    for dir in "${dirs[@]}"; do
      args+=(--more-"$side"-files "$dir"/"$side".{seg,shared.prepared})
    done
  done
  if $always_allow_unk; then
    args+=(--always-allow-unk)
  fi
  python sequence_to_sequence/prepare_data_shared.py \
    --vocabulary-output "$training_data_dir"/both.vocab \
    "${args[@]}"
fi

if $generate_separate_vocabulary; then
  for side in source target; do
    args=()
    if [[ $side = source ]]; then
      dirs=("${prepare_source_dirs[@]}")
    else
      dirs=("${prepare_target_dirs[@]}")
    fi
    for dir in "${dirs[@]}"; do
      args+=(--more-files "$dir"/"$side".{seg,separate.prepared})
    done
    if $always_allow_unk; then
      args+=(--always-allow-unk)
    fi
    python sequence_to_sequence/prepare_data_separate.py \
      --training-files "$training_data_dir"/"$side".{seg,separate.prepared} \
      --vocabulary-output "$training_data_dir"/"$side".vocab \
      "${args[@]}"
  done
fi
