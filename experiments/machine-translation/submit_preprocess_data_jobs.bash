set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

for dataset in europarl-v7-de-en; do
  case $dataset in
    europarl-v7-de-en) args=({europarl-v7,newstest{2016,2017}}-de-en) ;;
    toy) args=(toy-{train,valid,test}) ;;
    *) exit 1 ;;
  esac
  data_dirs=()
  for arg in "${args[@]}"; do
    data_dirs+=("$MT_DATA_DIR"/"$arg")
  done
  bash experiments/submit-job.bash \
    preprocess+"$dataset" \
    "$LOG_DIR"/outputs \
    cpu \
    bash machine_translation/preprocess_data.bash \
      --max-length 150 \
      --sample-size 100000 \
      "${data_dirs[@]}"
done
