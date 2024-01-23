set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <model> <task>"
}

model=${1-}
task=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

output_dir=../ignore/iclr2024/figures
mkdir -p "$output_dir"
output_name=$output_dir/$model-$task-heatmap
output_file=$output_name.tex
rm -f -- "$output_name"-*.png
python cfl_language_modeling/plot_stack_attention_heatmap.py \
  --load-model ../ignore/iclr2024/cfl/transformer-*-2."$model"-*.2/"$task"/* \
  --input-string <( \
    python cfl_language_modeling/print_data.py \
      --data-seed 123 \
      --length-range 40:41 \
      --data-size 1 \
      --task "$task" \
  ) \
  --task "$task" \
  --separate-legend \
  --show
  --pgfplots-output "$output_file"
sed -i '
  s/BOS/\\bos{}/g;
  s/EOS/\\eos{}/g;
  s/\\mathtt/\\sym/g;
  s|colormap/blackwhite,|colormap={whiteblack}{gray(0cm)=(1); gray(1cm)=(0)},|;
  s|\(\\addplot graphics .*{\)|\1figures/04-analysis/|;
  /\\end{axis}/q
' "$output_file"
echo '\end{tikzpicture}' >> "$output_file"
rm -f -- "$output_name"-001.png
