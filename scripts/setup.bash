set -e
bash scripts/poetry-install.bash
poetry run bash scripts/preprocess-ptb.bash
