To set up:
zdrive
conda activate mystctm_env
export MSG_DIR="$HOME/Imperial_College/MSci/msg"
export MESASDK_ROOT="$HOME/Imperial_College/MSci/mesasdk-26.6.1"
source "$MESASDK_ROOT/bin/mesasdk_init.sh"

To create grid and check:
python create_GJ486_grid.py

python - <<'PY'
import h5py

with h5py.File("R10000_model_grids/GJ486_pymsg.h5", "r") as f:
    print(f["model_grid"].shape)
PY


To run:
cd ~/Imperial_College/MSci/stctm/example/exotune_analysis/GJ486_JWST
stctm_exotune gj486_jwst.toml