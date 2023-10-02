set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <sections-dir> <output-dir>"
}

sections_dir=${1-}
output_dir=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

cat_sections() {
  for d in "$@"; do
    if [[ ! -d $d ]]; then
      echo "error: $d is not a directory of .mrg files" >&2
      return 1
    fi
  done
  # It looks like the file 24/wsj_2423.mrg has been mistakenly replaced with
  # 24/wsj_2424.mrg in Dyer et al. (2016)'s version of the PTB, so repeat that
  # mistake here for consistency.
  cat $( \
    find "$@" -name '*.mrg' \
      | sort \
      | sed 's|/24/wsj_2423.mrg$|/24/wsj_2424.mrg|' \
  )
}

generate_file() {
  local train_trees=$1
  local output_dir=$2
  local split=$3
  shift 3
  local sections=("$@")
  local trees_file=$output_dir/main.trees
  local seg_file=$output_dir/main.seg
  echo "writing $trees_file"
  cat_sections "${sections[@]}" \
    | python dyer_ptb/mrg_to_trees.py \
    | python dyer_ptb/clean_trees.py \
    > "$trees_file"
  if [[ $split = train ]]; then
    # See https://github.com/clab/rnng/tree/439950aa30ce1190f948c74ae4ef2fd17ae7a8d3#removing-sentences-that-start-with--from-the-oracle
    sed -i '33571d' "$trees_file"
  fi
  echo "writing $seg_file"
  PYTHONPATH=dyer_ptb/rnng python2 dyer_ptb/rnng/trees_to_tokens.py \
    "$train_trees" \
    "$trees_file" \
    > "$seg_file"
}

mkdir -p "$output_dir"/dyer-ptb-train/datasets/dyer-ptb-{valid,test}
# See https://aclanthology.org/N16-1024.pdf
# Training: sections 2-21
# Validation: section 24
# Evaluation: section 23
# Note that this is different from the Mikolov et al. (2011) version.
train_dir=$output_dir/dyer-ptb-train
train_trees=$train_dir/main.trees
generate_file "$train_trees" "$train_dir" train "$sections_dir"/{02..21}
generate_file "$train_trees" "$train_dir"/datasets/dyer-ptb-valid valid "$sections_dir"/24
generate_file "$train_trees" "$train_dir"/datasets/dyer-ptb-test test "$sections_dir"/23
rm "$output_dir"/dyer-ptb-train{,/datasets/dyer-ptb-{valid,test}}/main.trees
