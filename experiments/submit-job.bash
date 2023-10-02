# NOTE: If you intend to run experiments on your own computing cluster, you
# should edit this file to submit batch jobs to your system. See the comment
# near "dummy-submit-batch-job-command" below.

set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <job-name> <output-dir> <device> <command>...

Queues batch jobs.

Arguments:
  <job-name>      Unique name to identify this job.
  <output-dir>    Directory where additional outputs will be written, such as
                  the stdout of the command being run.
  <device>        Whether the job should be run on GPU or CPU.
                  One of: gpu, cpu
  <command>...    The command to be run.
"
}

job_name=${1-}
output_dir=${2-}
device=${3-}
if ! shift 3; then
  usage >&2
  exit 1
fi
args=("$@")

print_only=${PRINT_ONLY-0}
test_locally=${TEST_RUN-0}

case $device in
  gpu)
    # Possibly add different flags
    ;;
  cpu)
    # Possibly add different flags
    ;;
  *)
    echo "unrecognized device: $device" >&2
    exit 1
    ;;
esac

# NOTE To run this on your own computing cluster, replace this with a command
# that submits batch jobs to your system.
submit_job_args=( \
  dummy-submit-batch-job-command \
    --job-name "$job_name" \
    --output "$output_dir"/"$job_name".txt \
    experiments/job-script.bash "$device" "${args[@]}"
)

if [[ $print_only != 0 ]]; then
  echo "${submit_job_args[@]}"
elif [[ $test_locally != 0 ]]; then
  echo "running $job_name with output directory $output_dir"
  bash experiments/job-script.bash cpu "${args[@]}"
  exit 1
else
  echo "submitting $job_name"
  "${submit_job_args[@]}"
fi
