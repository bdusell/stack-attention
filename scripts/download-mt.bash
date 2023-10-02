set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <dir> <dataset-name>..."
}

BASE_DIR=$(cd "$(dirname "$BASH_SOURCE")"/.. && pwd)
dir=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi
dataset_names=("$@")

download_to_file() {
  local url=$1
  local output_file=$2
  echo "downloading $url to $output_file" >&2
  if ! curl -o "$output_file" "$url"; then
    rm -f "$output_file"
    return 1
  fi
}

download_to_dir() {
  local url=$1
  local output_dir=$2
  if [[ ! ( $url =~ /([^/]+)$ ) ]]; then
    echo "URL $url does not end in a file name" >&2
    return 1
  fi
  local output_name=${BASH_REMATCH[1]}
  download_to_file "$url" "$output_dir"/"$output_name"
}

mkdir -p "$dir"
for dataset_name in "${dataset_names[@]}"; do
  . "$BASE_DIR"/scripts/download-mt/"$dataset_name".bash
done
