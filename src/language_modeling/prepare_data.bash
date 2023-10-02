set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 [options]

  Options:

  --training-data <dir>
    Directory of data used as training data.
  --prepare <dir>
    Additional directory of data that should be prepared. Can be passed
    multiple times.
  --always-allow-unk
  --never-allow-unk
"
}

training_data_dir=
prepare_dirs=()
other_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --training-data) shift; training_data_dir=$1 ;;
    --prepare) shift; prepare_dirs+=("$1") ;;
    --always-allow-unk|--never-allow-unk) other_args+=("$1") ;;
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

more_files_args=()
for dir in "${prepare_dirs[@]}"; do
  more_files_args+=(--more-files "$dir"/main.{seg,prepared})
done
python language_modeling/prepare_data.py \
  --training-files "$training_data_dir"/main.{seg,prepared} \
  --vocabulary-output "$training_data_dir"/main.vocab \
  "${more_files_args[@]}" \
  "${other_args[@]}"
