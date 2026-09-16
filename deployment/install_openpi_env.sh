#!/usr/bin/env bash
# Build an isolated OpenPI inference environment from the training checkout's uv.lock.
set -euo pipefail

OPENPI_ROOT=${TATE_OPENPI_ROOT:-/home/qijun/models/TATE/openpi}
APP_ROOT=${TATE_APP_ROOT:-/home/qijun/models/TATE/app}
SDK_PYTHON=${TATE_ROBOT_PYTHON:-/home/qijun/ARX5_beta/.venv/bin/python}
BOOTSTRAP_DIR=${TATE_UV_BOOTSTRAP:-/home/qijun/models/TATE/uv-bootstrap}
INFER_PYTHON=${TATE_OPENPI_PYTHON:-3.11}

if [[ ! -f "$OPENPI_ROOT/pyproject.toml" || ! -f "$OPENPI_ROOT/uv.lock" ]]; then
  echo "OpenPI source and uv.lock are missing from $OPENPI_ROOT" >&2
  exit 2
fi
if [[ ! -x "$SDK_PYTHON" ]]; then
  echo "ARX SDK Python is missing: $SDK_PYTHON" >&2
  exit 2
fi

if [[ ! -x "$BOOTSTRAP_DIR/bin/uv" ]]; then
  "$SDK_PYTHON" -m venv "$BOOTSTRAP_DIR"
  "$BOOTSTRAP_DIR/bin/python" -m pip install --upgrade pip uv
fi

cd "$OPENPI_ROOT"
rm -f "$OPENPI_ROOT/.venv/.tate_inference_ready"
"$BOOTSTRAP_DIR/bin/uv" python install "$INFER_PYTHON"
"$BOOTSTRAP_DIR/bin/uv" sync --frozen --no-dev --python "$INFER_PYTHON"

TRANSFORMERS_SITE=$("$OPENPI_ROOT/.venv/bin/python" -c 'from pathlib import Path; import transformers; print(Path(transformers.__file__).parent)')
cp -a "$OPENPI_ROOT/src/openpi/models_pytorch/transformers_replace/." "$TRANSFORMERS_SITE/"

ln -sfn "$OPENPI_ROOT" "$APP_ROOT/thirdparty/openpi"
"$OPENPI_ROOT/.venv/bin/python" -c 'import torch, openpi; from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(); print("OpenPI ready: torch", torch.__version__, "CUDA available", torch.cuda.is_available())'
touch "$OPENPI_ROOT/.venv/.tate_inference_ready"
echo "OpenPI environment installed at $OPENPI_ROOT/.venv"
