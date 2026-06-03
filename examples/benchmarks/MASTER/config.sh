# use existing local environment `py310`
eval "$(conda shell.bash hook)"
conda activate py310

# install benchmark requirements in current env
pip install -r requirements.txt

# install local customizable qlib from this repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QLIB_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$QLIB_ROOT" || exit 1
python -m pip install -e .

# quick check (same effect as `qrun --help`)
python -m qlib.workflow.cli --help