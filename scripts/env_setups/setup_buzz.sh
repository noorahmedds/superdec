#!/usr/bin/env bash

# Robust, idempotent environment setup for SuperDec.
# - Creates/activates a Conda env if available; falls back to venv.
# - Installs dependencies from requirements.txt.
# - Optionally installs PyTorch (CPU by default) with configurable index URL.
#
# Usage:
#   setup_buzz.sh [ENV_NAME]
#
# Env vars (optional):
#   PYTHON_VERSION   Python version for new env (default: 3.9)
#   TORCH            If set to 1, ensures torch/torchvision are installed
#   TORCH_INDEX_URL  Custom pip index for PyTorch wheels (e.g. CUDA builds)

set -Eeuo pipefail
trap 'echo "[setup_buzz] Error on line $LINENO" >&2' ERR

ENV_NAME="${1:-superdec}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

log()  { printf "[setup_buzz] %s\n" "$*"; }
warn() { printf "[setup_buzz][warn] %s\n" "$*" >&2; }
die()  { printf "[setup_buzz][error] %s\n" "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Try to load conda into the shell if present but not initialized
maybe_init_conda() {
  if have conda; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
  fi
}

create_or_activate_conda_env() {
  local name="$1" py="$2"
  maybe_init_conda
  if ! have conda; then
    return 1
  fi

  # Detect if env exists
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    log "Conda env '$name' exists; activating."
  else
    log "Creating conda env '$name' with Python $py (first run only)."
    conda create -y -n "$name" "python=$py"
  fi

  # Activate the env for subsequent pip installs
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$name"
  return 0
}

create_or_activate_venv() {
  local name="$1" py_bin
  py_bin="$(command -v python3 || command -v python || true)"
  [ -n "$py_bin" ] || die "Python is not installed. Install Python 3.9+ and re-run."

  local venv_dir="$PROJECT_ROOT/.venv-$name"
  if [ ! -d "$venv_dir" ]; then
    log "Creating venv at $venv_dir (first run only)."
    "$py_bin" -m venv "$venv_dir"
  else
    log "Using existing venv at $venv_dir."
  fi
  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"
}

install_requirements() {
  log "Upgrading pip and wheel."
  python -m pip install --upgrade pip wheel

  if [ -f "$REQ_FILE" ]; then
    log "Installing pytorch first"
    pip install torch torchvision

    log "Installing Python packages from requirements.txt one by one"
    cat "$REQ_FILE" | xargs -n 1 pip install
  else
    warn "requirements.txt not found at $REQ_FILE; installing a minimal set."
    pip install h5py omegaconf hydra-core tqdm plyfile ninja wandb \
                trimesh Cython matplotlib open3d viser gdown
  fi

  if [ "${TORCH:-0}" = "1" ]; then
    if python -c 'import torch' 2>/dev/null; then
      log "PyTorch already installed; skipping."
    else
      if [ -n "${TORCH_INDEX_URL:-}" ]; then
        log "Installing PyTorch via custom index: $TORCH_INDEX_URL"
        pip install --index-url "$TORCH_INDEX_URL" torch torchvision
      else
        log "Installing CPU-only PyTorch from PyPI. Set TORCH=1 to enable, TORCH_INDEX_URL for CUDA builds."
        pip install torch torchvision
      fi
    fi
  else
    log "Skipping PyTorch install (set TORCH=1 to enable)."
  fi
}

main() {
  log "Setting up environment '$ENV_NAME' (Python $PYTHON_VERSION)."

  if create_or_activate_conda_env "$ENV_NAME" "$PYTHON_VERSION"; then
    log "Using conda environment '$ENV_NAME'."
  else
    warn "Conda not available; falling back to Python venv."
    create_or_activate_venv "$ENV_NAME"
  fi

  install_requirements

  log "Environment setup complete for '$ENV_NAME'."
  log "Tip: Re-activate later with either:"
  log "  - conda:   conda activate $ENV_NAME (if using conda)"
  log "  - venv:    source .venv-$ENV_NAME/bin/activate (if using venv)"
}

main "$@"
