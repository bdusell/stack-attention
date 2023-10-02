get_optimizer_args() {
  declare -n _result=$1
  _result=( \
    --optimizer Adam \
  )
}

get_parameter_update_args() {
  declare -n _result=$1
  _result=( \
    --label-smoothing 0.1 \
    --gradient-clipping-threshold 5 \
  )
}

get_vocab_args() {
  local train_dir=$1
  local vocab_type=$2
  declare -n _result=$3
  case $vocab_type in
    shared)
      vocab_args=( \
        --shared-vocabulary "$train_dir"/both.vocab \
      )
      ;;
    separate)
      vocab_args=( \
        --source-vocabulary "$train_dir"/source.vocab \
        --target-vocabulary "$train_dir"/target.vocab \
      )
      ;;
    *)
      echo "unknown vocabulary type: $vocab_type" &>2
      return 1
      ;;
  esac
}

get_model_args() {
  local model_str=$1
  declare -n _result=$2
  _result=( \
    --d-model 256 \
    --num-heads 8 \
    --feedforward-size 1024 \
    --dropout 0.1 \
    --init-scale 0.01 \
    --encoder-layers "$model_str" \
    --decoder-layers "$model_str" \
  )
}

get_batching_args() {
  local train_dir=$1
  local vocab_type=$2
  local model_str=$3
  local batching_str=$4
  declare -n _result=$5
  case $batching_str in
    num-tokens)
      _result=( \
        --batching-max-tokens 2048 \
      )
      ;;
    polynomial)
      _result=( \
        --batching-space-coefficients=$(< "$train_dir"/cost-data/"$vocab_type"/"$model_str"/coefficients.txt) \
        --batching-max-memory 8GB \
      )
      ;;
    precomputed)
      _result=( \
        --batching-precomputed-cost "$train_dir"/cost-data/"$vocab_type"/"$model_str"/precomputed-cost.tsv \
        --batching-max-memory 8GB \
      )
      ;;
    *)
      echo "unknown batching type: $batching_str" >&2
      return 1
      ;;
  esac
}
