set -e
set -u
set -o pipefail

usage() {
  echo "$0 <output-dir> <data-dir> <dataset-str> <model-str> ..."
}

output_dir=${1-}
data_dir=${2-}
dataset_str=${3-}
model_str=${4-}
if ! shift 4; then
  usage >&2
  exit 1
fi
extra_args=("$@")

if [[ $dataset_str = dyer-ptb ]]; then
  train_dataset=dyer-ptb-train
  valid_dataset=dyer-ptb-valid
  test_dataset=dyer-ptb-test
else
  echo "unknown dataset: $dataset_str" >&2
  exit 1
fi

train_dir=$data_dir/$train_dataset
valid_dir=$train_dir/datasets/$valid_dataset

num_heads=8

if [[ $model_str =~ ^(small|medium|large)-(transformer|superposition|nondeterministic)$ ]]; then
  size=${BASH_REMATCH[1]}
  model_type=${BASH_REMATCH[2]}
  case $size in
    small)
      case $model_type in
        transformer) model_size=12 ;;
        superposition) model_size=12 ;;
        nondeterministic) model_size=11 ;;
      esac
      ;;
    medium)
      case $model_type in
        transformer) model_size=21 ;;
        superposition) model_size=21 ;;
        nondeterministic) model_size=20 ;;
      esac
      ;;
    large)
      case $model_type in
        transformer) model_size=36 ;;
        superposition) model_size=36 ;;
        nondeterministic) model_size=35 ;;
      esac
      ;;
  esac
  dmodel=$(( model_size * num_heads ))
  case $model_type in
    transformer) layers=5 ;;
    superposition) layers=2.superposition-$dmodel.2 ;;
    nondeterministic) layers=2.nondeterministic-3-3-5.2 ;;
  esac
else
  dmodel=256
  layers=$model_str
fi

learning_rate=$(python utils/random_sample.py --log 1e-6 1e-4)
train_batching_max_tokens=$(python utils/random_sample.py --int 128 512)

common_args=( \
  --einsum-block-size 10 \
)
python language_modeling/train.py \
  --training-data "$train_dir"/main.prepared \
  --validation-data "$valid_dir"/main.prepared \
  --vocabulary "$train_dir"/main.vocab \
  --output "$output_dir" \
  --batching-max-tokens "$train_batching_max_tokens" \
  --d-model "$dmodel" \
  --num-heads "$num_heads" \
  --feedforward-size "$(( 4 * dmodel ))" \
  --dropout 0.1 \
  --init-scale 0.01 \
  --layers "$layers" \
  --epochs 10000 \
  --optimizer Adam \
  --learning-rate "$learning_rate" \
  --gradient-clipping-threshold 5 \
  --early-stopping-patience 4 \
  --learning-rate-patience 2 \
  --learning-rate-decay-factor 0.5 \
  --checkpoint-interval-sequences 20000 \
  "${common_args[@]}" \
  "${extra_args[@]}"
bash language_modeling/evaluate_language_model.bash \
  "$data_dir" \
  "$train_dataset" \
  "$test_dataset" \
  "$output_dir" \
  "${common_args[@]}" \
  --batching-max-tokens 2048
