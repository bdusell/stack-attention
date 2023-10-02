set -e
set -u
set -o pipefail

output_file=code.zip
rm -f "$output_file"
zip -r "$output_file" . \
  --include \
    'Dockerfile*' \
    '.docker*' \
    'experiments/*' \
    'poetry.lock' \
    'pyproject.toml' \
    'README.md' \
    'scripts/*' \
    'src/*' \
    'tests/*' \
  --exclude \
    '*/.git*' \
    '*/__pycache__/*' \
    '*.pyc' \
    '*.swp' \
    '*/.pytest_cache/*' \
    '*/.mypy_cache/*'
