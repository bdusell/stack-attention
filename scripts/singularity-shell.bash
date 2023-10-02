set -e

. scripts/variables.bash

singularity shell --nv "$IMAGE".sif
