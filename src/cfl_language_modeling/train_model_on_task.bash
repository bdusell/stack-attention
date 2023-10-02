set -e
set -u
set -o pipefail

usage() {
  echo "$0 <output-dir> <model-str> <task-str> <trial-no> ...

  <model-str> A string describing the type of architecture to train.
              Choices:
              - lstm-<h>(-<stack>)
                LSTM with <h> hidden units, with optional stack string <stack>.
              - transformer-<dmodel>-<layer1>.<layer2>. ... .<layerk>
                Each layer can be one of:
                  - An integer, indicating that many regular transformer layers.
                  - A stack string <stack>.
              Choices for <stack>:
              - superposition-<m>
                Superposition stack with stack embedding size <m>.
              - nondeterministic-<q>-<s>-<m>
                dVPDA with <q> states, <s> stack symbols, and a stack embedding
                size of <m>.
                If <q>-<s> is the string x-x, then <q> and <s> will be chosen
                automatically based on the task.
  <task-str>  A string describing the task to train on.
              In all cases below, <k> indicates that the task will involve <k>
              symbol types, and the -<k> can be omitted to indicate a default
              value of 2.
              - marked-reversal-<k>
              - unmarked-reversal-<k>
              - padded-reversal-<k>
              - dyck-<k>
              - hardest-cfl
              - count-3
              - marked-copy
              - marked-reverse-and-copy-<k>
              - unmarked-copy
              - unmarked-reverse-and-copy
              - count-and-copy
              - unmarked-copy-different-alphabets
  <trial-no>  A number to assign to this random restart.
  ...         Any extra arguments to pass to train.py.
              Common choices:
              --no-progress
              --device
"
}

output_dir=${1-}
model_str=${2-}
task_str=${3-}
trial_no=${4-}
if ! shift 4; then
  usage >&2
  exit 1
fi
extra_args=("$@")

# Parse the task string.
if [[ $task_str =~ ^(.+)(-([0-9]+))?$ ]]; then
  task_name=${BASH_REMATCH[1]}
  task_symbol_types=${BASH_REMATCH[3]}
  if [[ ! $task_symbol_types ]]; then
    if [[ $task_name =~ ^(marked-reversal|unmarked-reversal|padded-reversal|dyck|marked-reverse-and-copy)$ ]]; then
      task_symbol_types=2
    else
      task_symbol_types=x
    fi
  fi
else
  echo "Unknown task $task_str" >&2
  usage >&2
  exit 1
fi

# Figure out the default |Q| and |\Gamma|.
if [[ $task_name =~ ^(marked-reversal|unmarked-reversal|dyck)$ ]]; then
  default_num_states=2
else
  default_num_states=3
fi
default_stack_alphabet_size=3

model_args=()
if [[ $model_str =~ ^lstm-([0-9]+)(-(.+))?$ ]]; then
  hidden_units=${BASH_REMATCH[1]}
  stack_str=${BASH_REMATCH[3]}
  model_args+=(--hidden-units "$hidden_units")
  if [[ ! $stack_str ]]; then
    model_args+=(--model-type lstm)
  elif [[ $stack_str =~ ^superposition-([0-9]+)$ ]]; then
    stack_embedding_size=${BASH_REMATCH[1]}
    model_args+=(--model-type jm --stack-embedding-size "$stack_embedding_size")
  elif [[ $stack_str =~ ^nondeterministic-x-x-([0-9]+)$ ]]; then
    stack_embedding_size=${BASH_REMATCH[1]}
    model_args+=( \
      --model-type vns \
      --num-states "$default_num_states" \
      --stack-alphabet-size "$default_stack_alphabet_size" \
      --stack-embedding-size "$stack_embedding_size" \
    )
  else
    echo "unrecognized stack string: $stack_str" >&2
    exit 1
  fi
elif [[ $model_str =~ ^transformer-([0-9]+)-(.*)$ ]]; then
  dmodel=${BASH_REMATCH[1]}
  layers=${BASH_REMATCH[2]}
  layers=${layers//nondeterministic-x-x-/nondeterministic-$default_num_states-$default_stack_alphabet_size-}
  model_args=( \
    --model-type transformer \
    --d-model "$dmodel" \
    --num-heads 4 \
    --feedforward-size "$(( 2 * dmodel ))" \
    --transformer-sublayer-dropout 0.1 \
    --transformer-layers "$layers" \
  )
else
  echo "Unknown model $model_str" >&2
  usage >&2
  exit 1
fi

bash cfl_language_modeling/train_on_task.bash \
  "$output_dir"/"$model_str"/"$task_str" \
  "$task_name" \
  "$task_symbol_types" \
  "$trial_no" \
  "${model_args[@]}" \
  "${extra_args[@]}"
