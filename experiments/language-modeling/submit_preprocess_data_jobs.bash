set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

bash experiments/submit-job.bash \
  preprocess+dyer-ptb \
  "$LOG_DIR"/outputs \
  cpu \
  bash language_modeling/prepare_data.bash \
    --training-data "$LM_DATA_DIR"/dyer-ptb-train \
    --prepare "$LM_DATA_DIR"/dyer-ptb-train/datasets/dyer-ptb-valid \
    --prepare "$LM_DATA_DIR"/dyer-ptb-train/datasets/dyer-ptb-test \
    --never-allow-unk
