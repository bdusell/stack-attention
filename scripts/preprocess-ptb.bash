set -e
set -u
set -o pipefail

BASE_DIR=$(cd $(dirname "$BASH_SOURCE")/.. && pwd)
DEFAULT_SECTIONS_DIR=$BASE_DIR/data/ptb/dist/treebank_3/parsed/mrg/wsj
OUTPUT_DIR=$BASE_DIR/data/language-modeling

usage() {
  echo "Usage: $0 [<sections-dir>]"
}

sections_dir=${1-"$DEFAULT_SECTIONS_DIR"}
if [[ ! -d $sections_dir ]]; then
  echo "\
error: The directory $sections_dir does not exist. Please ensure that it
  contains the files under dist/treebank_3/parsed/mrg/wsj from the original
  Penn Treebank distribution." >&2
  exit 1
fi

cd src
bash dyer_ptb/preprocess_ptb.bash "$sections_dir" "$OUTPUT_DIR"
