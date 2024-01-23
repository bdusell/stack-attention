set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <model> <task>"
}

model=${1-}
task=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

python cfl_language_modeling/print_stack_attention_viterbi.py \
  --load-model ../ignore/iclr2024/cfl/transformer-*-2."$model"-*.2/"$task"/* \
  --input-string <( \
    python cfl_language_modeling/print_data.py \
      --data-seed 123 \
      --length-range 40:41 \
      --data-size 1 \
      --task "$task" \
  ) \
  --task "$task"
