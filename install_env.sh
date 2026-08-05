#!/usr/bin/env bash
set -euo pipefail


PYTHON="python"

"$PYTHON" - <<'PY'
import torch
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 8)
print(f"H3 environment OK: torch={torch.__version__}, cuda={torch.version.cuda}, gpus={torch.cuda.device_count()}")
PY

cleanup() { rm -rf -- "${tmp_dir:-}" "${tmp_transformers:-}"; }
trap cleanup EXIT

# diffusers@abc5e9bf imports huggingface_hub.get_cached_repo_tree, which only
# exists in hub >= 1.0 (the cluster mirror carries no 0.37-0.99) and its
# metadata requires hub>=1.23. The diffusers install below is --no-deps, so
# bump hub explicitly — and BEFORE the transformers check: with hub >= 1.23 a
# pre-existing 4.57.x transformers fails its import-time version check
# (hub<1.0 required), which is exactly what trips the reinstall branch below.
"$PYTHON" -m pip install -q "huggingface-hub>=1.23,<2"

# The H3 integration is not in the public 0.36 wheel. Avoid build isolation so
# the cluster's configured PyPI mirror is not consulted for build dependencies.
if ! "$PYTHON" -c 'from diffusers import MiniMaxH3Transformer3DModel' >/dev/null 2>&1 || [[ "$(readlink -f "$("$PYTHON" -c 'import diffusers; print(diffusers.__file__)')")" == /tmp/* ]]; then
  tmp_dir="$(mktemp -d /tmp/diffusers-h3-install.XXXXXX)"
  git clone -q https://github.com/huggingface/diffusers.git "$tmp_dir"
  git -C "$tmp_dir" checkout -q abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
  "$PYTHON" -m pip install --no-deps --no-build-isolation "$tmp_dir"
fi
if ! "$PYTHON" -c 'from transformers import Qwen3VLForConditionalGeneration' >/dev/null 2>&1 || [[ "$(readlink -f "$("$PYTHON" -c 'import transformers; print(transformers.__file__)')")" == /tmp/* ]]; then
  # transformers 4.57.x hard-requires huggingface-hub<1.0 at import time, an
  # empty intersection with the hub>=1.23 pin above. 5.0.0 allows hub<2.0 and
  # keeps Qwen3-VL, which the H3 text encoder needs.
  tmp_transformers="$(mktemp -d /tmp/transformers-h3-install.XXXXXX)"
  git clone -q --depth 1 --branch v5.0.0 https://github.com/huggingface/transformers.git "$tmp_transformers"
  "$PYTHON" -m pip install --no-deps --no-build-isolation "$tmp_transformers"
fi
# peft 0.17 imports transformers.HybridCache, removed in transformers 5.0;
# 0.20.0 is the first version verified against the 5.0.0 + hub 1.26 combo.
"$PYTHON" -m pip install -q "peft>=0.20"

"$PYTHON" - <<'PY'
from diffusers import MiniMaxH3Transformer3DModel
from transformers import Qwen3VLForConditionalGeneration
import huggingface_hub, peft, transformers
assert tuple(map(int, huggingface_hub.__version__.split('.')[:2])) >= (1, 23)
print(f"MiniMaxH3Transformer3DModel import OK (hub {huggingface_hub.__version__}, "
      f"transformers {transformers.__version__}, peft {peft.__version__})")
PY
