set -e
set -u
set -o pipefail

. "$(dirname "$BASH_SOURCE")"/include.bash

print_table() {
  local TEX_PATH='ICLR 2024/figures/01-cfls'
  for task in "$@"; do
    echo "\\scalebox{0.8}{\\input{$TEX_PATH/train/$task}}"
    echo "&\\scalebox{0.8}{\\input{$TEX_PATH/test/$task}}"
    echo "\\\\"
  done
}

write_table() {
  local output=$1
  shift 1
  echo "writing $output"
  print_table "$@" > "$output"
}

fix_line_thickness() {
  local tex_file=$1
  sed -i '
    s/\\addplot \[semithick, color\([0-9]\+\)\]/\\addplot [cedline, color\1]/;
    s/\\addplot \[semithick, .* dashed\]/\\addplot [dashedline]/
  ' "$tex_file"
}

cd "$ROOT_DIR"/src

mkdir -p "$FIGURES_DIR"/{png,tex}/cfl/{train,test}

table_output=$FIGURES_DIR/tex/cfl/table.tex

write_table "$table_output" "${TASKS[@]}"

legend_png_output=$FIGURES_DIR/png/cfl/legend.png
legend_tex_output=$FIGURES_DIR/tex/cfl/legend.tex
wrote_legend=false

write_plots() {
  local tasks=("$@")
  local last_task=${tasks[${#tasks[@]} - 1]}
  for task in "${tasks[@]}"; do
    plot_args=( \
      --title "$(format_task_name "$task")" \
      --target-runs "${#TRIALS[@]}" \
      --separate-legend \
      --width 3.4 \
      --height 1.575 \
    )
    plot_train_args=()
    plot_test_args=()
    for model in "${MODELS[@]}"; do
      trial_args=()
      for trial_no in "${TRIALS[@]}"; do
        trial_args+=("$(get_output_directory "$task" "$model" "$trial_no")")
      done
      output=$(python utils/print_best.py "${trial_args[@]}")
      best_model=$(cut -f 1 <<<"$output")
      num_trials=$(cut -f 2 <<<"$output")
      plot_train_args+=(--input)
      plot_test_args+=(--input)
      if [[ $best_model ]]; then
        plot_train_args+=("$best_model")
        local test_dir=$best_model/test
        if [[ -d $test_dir ]]; then
          plot_test_args+=("$test_dir")
        else
          echo "missing: $test_dir" >&2
        fi
      fi
      plot_args+=(--label "$(format_model_name "$model")" --runs "$num_trials")
    done
    if ! $wrote_legend; then
      echo "writing $legend_png_output"
      echo "writing $legend_tex_output"
      plot_train_args+=( \
        --legend-output "$legend_png_output" \
        --legend-pgfplots-output "$legend_tex_output" \
      )
    fi
    if [[ $task = $last_task ]]; then
      plot_args+=(--show-x-label)
    fi
    train_png_output=$FIGURES_DIR/png/cfl/train/$task.png
    train_tex_output=$FIGURES_DIR/tex/cfl/train/$task.tex
    echo "writing $train_png_output"
    echo "writing $train_tex_output"
    python cfl_language_modeling/plot_train.py \
      --output "$train_png_output" \
      --pgfplots-output "$train_tex_output" \
      "${plot_args[@]}" \
      "${plot_train_args[@]}"
    fix_line_thickness "$train_tex_output"
    test_png_output=$FIGURES_DIR/png/cfl/test/$task.png
    test_tex_output=$FIGURES_DIR/tex/cfl/test/$task.tex
    echo "writing $test_png_output"
    echo "writing $test_tex_output"
    if ! $wrote_legend; then
      sed -i '
        s/legend columns=-1/legend columns=3/;
        s/\\addlegendimage{/\\addlegendimage{cedline,/
      ' "$legend_tex_output"
      wrote_legend=true
    fi
    python cfl_language_modeling/plot_test.py \
      --output "$test_png_output" \
      --pgfplots-output "$test_tex_output" \
      "${plot_args[@]}" \
      "${plot_test_args[@]}"
    fix_line_thickness "$test_tex_output"
  done
}

write_plots "${TASKS[@]}"
