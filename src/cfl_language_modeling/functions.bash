get_cfl_task_args() {
  local task=$1
  local symbol_types=$2
  declare -n _result=$3
  _result=(--task "$task")
  case $task in
    marked-reversal|unmarked-reversal)
      _result+=( \
        --symbol-types "$symbol_types" \
        --mean-length 60 \
      )
      ;;
    padded-reversal)
      _result+=( \
        --symbol-types "$symbol_types" \
        --mean-content-length 60 \
        --mean-padding-length 30 \
      )
      ;;
    dyck)
      _result+=( \
        --bracket-types "$symbol_types" \
        --mean-bracket-splits 1 \
        --mean-nesting-depth 40 \
      )
      ;;
    hardest-cfl)
      if [[ $symbol_types != x ]]; then
        echo "variable symbol types not supported for $task" >&2
        return 1
      fi
      _result+=( \
        --mean-num-commas 0.5 \
        --mean-short-filler-length 0.5 \
        --mean-long-filler-length 2 \
        --semicolon-probability 0.25 \
        --mean-bracket-splits 1.5 \
        --mean-nesting-depth 3 \
      )
      ;;
    count-3|marked-copy|unmarked-copy|unmarked-reverse-and-copy|count-and-copy|unmarked-copy-different-alphabets)
      if [[ $symbol_types != x ]]; then
        echo "variable symbol types not supported for $task" >&2
        return 1
      fi
      ;;
    marked-reverse-and-copy)
      _result+=( \
        --symbol-types "$symbol_types" \
      )
      ;;
    *)
      echo "unrecognized task name: $task" >&2
      return 1
      ;;
  esac
}
