#!/usr/bin/env bash
set -euo pipefail


PYTHON="python"

"$PYTHON" - <<'PY'
import torch
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 8)
print(f"H3 environment OK: torch={torch.__version__}, cuda={torch.version.cuda}, gpus={torch.cuda.device_count()}")
PY

# The H3 integration is not in the public 0.36 wheel. Avoid build isolation so
# the cluster's configured PyPI mirror is not consulted for build dependencies.
if ! "$PYTHON" -c 'from diffusers import MiniMaxH3Transformer3DModel' >/dev/null 2>&1 || [[ "$(readlink -f "$("$PYTHON" -c 'import diffusers; print(diffusers.__file__)')")" == /tmp/* ]]; then
  tmp_dir="$(mktemp -d /tmp/diffusers-h3-install.XXXXXX)"
  trap 'rm -rf "$tmp_dir"' EXIT
  git clone -q https://github.com/huggingface/diffusers.git "$tmp_dir"
  git -C "$tmp_dir" checkout -q abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
  "$PYTHON" -m pip install --no-deps --no-build-isolation "$tmp_dir"
fi
if ! "$PYTHON" -c 'from transformers import Qwen3VLForConditionalGeneration' >/dev/null 2>&1 || [[ "$(readlink -f "$("$PYTHON" -c 'import transformers; print(transformers.__file__)')")" == /tmp/* ]]; then
  tmp_transformers="$(mktemp -d /tmp/transformers-h3-install.XXXXXX)"
  trap 'rm -rf "$tmp_transformers"' EXIT
  git clone -q --depth 1 --branch v4.57.3 https://github.com/huggingface/transformers.git "$tmp_transformers"
  "$PYTHON" -m pip install --no-deps --no-build-isolation "$tmp_transformers"
fi

"$PYTHON" - <<'PY'
from diffusers import MiniMaxH3Transformer3DModel
print("MiniMaxH3Transformer3DModel import OK")
PY
