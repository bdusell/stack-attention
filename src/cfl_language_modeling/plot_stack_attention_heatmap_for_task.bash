set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <output-dir> <task> <stack-attention-type> <model-dirs>..."
}

output_dir=${1-}
task=${2-}
stack_attention_type=${3-}
if ! shift 3; then
  usage >&2
  exit 1
fi
model_dirs=("$@")

best_model=$(python utils/print_best.py "${model_dirs[@]}" | cut -f 1)
echo "best model: $best_model"
mkdir -p "$output_dir"
output_name=$output_dir/$stack_attention_type-$task-heatmap
output_file=$output_name.tex
rm -f -- "$output_name"-*.png
python cfl_language_modeling/plot_stack_attention_heatmap.py \
  --load-model "$best_model" \
  --input-string <( \
    python cfl_language_modeling/print_data.py \
      --data-seed 123 \
      --length-range 40:41 \
      --data-size 1 \
      --task "$task" \
  ) \
  --task "$task" \
  --separate-legend \
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
