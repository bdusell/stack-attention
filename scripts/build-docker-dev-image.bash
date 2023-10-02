set -e
set -u

. scripts/variables.bash

DOCKER_BUILDKIT=1 docker build "$@" -t "$DEV_IMAGE":latest -f Dockerfile-dev .
