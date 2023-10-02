#!/bin/bash
set -e
set -u
set -o pipefail

. scripts/variables.bash

usage() {
  echo "Usage: <device> <command>...

  <device>    One of: cpu, gpu
"
}

device=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi

case $device in
  cpu) flags='' ;;
  gpu) flags='--nv' ;;
  *)
    usage >&2
    exit 1
    ;;
esac

exec singularity exec \
  $flags \
  "$IMAGE".sif \
  "$@"
