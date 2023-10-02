set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

for dataset in europarl-v7 newstest2016 newstest2017; do
  bash experiments/submit-job.bash \
    download+"$dataset" \
    "$LOG_DIR"/outputs \
    cpu \
    bash -c "cd .. && bash scripts/download-mt.bash $(printf %q "$MT_DATA_DIR") $(printf %q "$dataset")"
done
