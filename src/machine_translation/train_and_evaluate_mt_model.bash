set -e
set -u
set -o pipefail

. machine_translation/functions.bash

usage() {
  echo "$0 <output-dir> <data-dir> <dataset-str> <vocab-type> <model-str> <batching-str> ..."
}

output_dir=${1-}
data_dir=${2-}
dataset_str=${3-}
vocab_type=${4-}
model_str=${5-}
batching_str=${6-}
if ! shift 6; then
  usage >&2
  exit 1
fi
extra_args=("$@")

if [[ $dataset_str =~ ^([^/]+)(/.+)?$ ]]; then
  dataset_name=${BASH_REMATCH[1]}
  dataset_suffix=${BASH_REMATCH[2]}
else
  echo "invalid dataset string: $dataset_str" >&2
  exit 1
fi

if [[ $dataset_name = europarl-v7-de-en ]]; then
  train_dataset=europarl-v7-de-en
  valid_dataset=newstest2016-de-en
  test_dataset=newstest2017-de-en
elif [[ $dataset_name = toy ]]; then
  train_dataset=toy-train
  valid_dataset=toy-valid
  test_dataset=toy-test
  extra_args=(--checkpoint-interval-sequences 5000 "${extra_args[@]}")
else
  echo "unknown dataset: $dataset_name" >&2
  exit 1
fi

train_dataset=$train_dataset$dataset_suffix

train_dir=$data_dir/$train_dataset
valid_dir=$train_dir/datasets/$valid_dataset

learning_rate=$(python utils/random_sample.py --log 0.00001 0.001)

get_optimizer_args optimizer_args
get_parameter_update_args parameter_update_args
get_vocab_args "$train_dir" "$vocab_type" vocab_args
if [[ $model_str =~ ^(small|medium|large)-(transformer|superposition|nondeterministic)$ ]]; then
  size=${BASH_REMATCH[1]}
  model_type=${BASH_REMATCH[2]}
  num_heads=8
  case $size in
    small) model_size=20 ;;
    medium) model_size=30 ;;
    large) model_size=45 ;;
  esac
  dmodel=$(( model_size * num_heads ))
  case $model_type in
    transformer) layers=5 ;;
    superposition) layers=2.superposition-$dmodel.2 ;;
    nondeterministic) layers=2.nondeterministic-3-3-5.2 ;;
  esac
  model_args=( \
    --d-model "$dmodel" \
    --num-heads "$num_heads" \
    --feedforward-size "$(( 4 * dmodel ))" \
    --dropout 0.1 \
    --init-scale 0.01 \
    --encoder-layers "$layers" \
    --decoder-layers "$layers" \
  )
else
  get_model_args "$model_str" model_args
fi
get_batching_args "$train_dir" "$vocab_type" "$model_str" "$batching_str" batching_args

common_args=( \
  "${batching_args[@]}" \
  --einsum-block-size 32 \
)
python sequence_to_sequence/train.py \
  --training-data-source "$train_dir"/source."$vocab_type".prepared \
  --training-data-target "$train_dir"/target."$vocab_type".prepared \
  --validation-data-source "$valid_dir"/source."$vocab_type".prepared \
  --validation-data-target "$valid_dir"/target."$vocab_type".prepared \
  "${vocab_args[@]}" \
  --output "$output_dir" \
  "${model_args[@]}" \
  --epochs 100 \
  "${optimizer_args[@]}" \
  --learning-rate "$learning_rate" \
  "${parameter_update_args[@]}" \
  --early-stopping-patience 4 \
  --learning-rate-patience 2 \
  --learning-rate-decay-factor 0.5 \
  --checkpoint-interval-sequences 50000 \
  "${common_args[@]}" \
  "${extra_args[@]}"
bash machine_translation/evaluate_mt_model.bash \
  "$data_dir" \
  "$train_dataset" \
  "$test_dataset" \
  "$vocab_type" \
  "$output_dir" \
  "${common_args[@]}"
# TODO It might be necessary to decrease max memory here to account for beam size
