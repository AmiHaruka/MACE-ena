#!/usr/bin/env bash
# Linux x86_64 / CPython 3.10 optional dependencies only.
set -euo pipefail

mode=${1:-help}
wheel_dir=${2:-wheelhouse-mlmm}
backend=${3:-cuda12}
constraints_source=${4:-}
mlmm_python=${PYTHON:-python}

if [[ "$mode" == help || "$mode" == --help ]]; then
    cat <<'EOF'
Offline target server, metadata only; copy the exported file to the online machine:
  PYTHON=/server/python bash offline.sh constraints > server-cuda-constraints.txt
Online Linux CPython 3.10 machine:
  PYTHON=/office/python bash offline.sh download /path/to/wheelhouse cuda12 /path/to/server-cuda-constraints.txt
  PYTHON=/path/to/python bash offline.sh download /path/to/wheelhouse [cuda12|cpu]
The three-argument download form derives constraints from the local Torch.
Copy wheelhouse and this directory to the offline Linux CPython 3.10 machine:
  PYTHON=/path/to/python bash offline.sh install /path/to/wheelhouse [cuda12|cpu]
  PYTHON=/path/to/python bash offline.sh check /path/to/wheelhouse [cuda12|cpu]
Only optional OpenMM/torch-pme and the selected OpenMM CUDA plugin closure are
downloaded/installed. Existing Torch, PyG and Triton versions are preserved.
EOF
    exit 0
fi
if [[ "$backend" != cuda12 && "$backend" != cpu ]]; then
    echo "Backend must be cuda12 or cpu." >&2
    exit 2
fi
"$mlmm_python" - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 10) or sys.platform != "linux" or platform.machine() != "x86_64":
    raise SystemExit("Use Linux x86_64 with CPython 3.10 for this wheel recipe.")
PY

cuda_constraints() {
    "$mlmm_python" - "${1:-}" <<'PY'
from importlib.metadata import requires
from pathlib import Path
import re
import sys

if sys.argv[1]:
    lines = Path(sys.argv[1]).read_text().splitlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if any(not line.lower().startswith("nvidia-") for line in requirements):
        raise SystemExit("Target constraints must contain NVIDIA requirements only.")
else:
    requirements = [line for line in requires("torch") or [] if line.lower().startswith("nvidia-")]
if not any(line.lower().startswith("nvidia-cuda-nvrtc-cu12") for line in requirements):
    raise SystemExit("CUDA12 wheel preparation requires the target CUDA12 Torch package's NVIDIA requirement metadata.")
runtime = next((line for line in requirements if line.lower().startswith("nvidia-cuda-runtime-cu12")), "")
match = re.search(r"==([0-9]+)\.([0-9]+)", runtime)
if match is None:
    raise SystemExit("Target Torch must declare an exact CUDA12 runtime version.")
if not any(line.lower().startswith("nvidia-cuda-nvcc-cu12") for line in requirements):
    major, minor = map(int, match.groups())
    requirements.append(f"nvidia-cuda-nvcc-cu12>={major}.{minor},<{major}.{minor+1}")
print("\n".join(requirements))
PY
}

if [[ "$mode" == constraints ]]; then
    cuda_constraints
    exit 0
fi

if [[ "$mode" == download ]]; then
    mkdir -p "$wheel_dir"
    if [[ "$backend" == cuda12 ]]; then
        # The fourth argument may have been exported on an offline server with
        # a different Torch/CUDA stack.  No local Torch metadata is used then.
        cuda_constraints "$constraints_source" > "$wheel_dir/torch-nvidia-constraints.txt.tmp"
        mv "$wheel_dir/torch-nvidia-constraints.txt.tmp" "$wheel_dir/torch-nvidia-constraints.txt"
    fi
    # --no-deps keeps the large existing Torch stack out of the wheelhouse.
    "$mlmm_python" -m pip download --only-binary=:all: --no-deps --dest "$wheel_dir" 'OpenMM==8.6.1' 'torch-pme==0.5.0'
    if [[ "$backend" == cuda12 ]]; then
        # Resolve the plugin's own dependency metadata, including CUDA NVRTC.
        # Target Torch constraints prevent a newer incompatible CUDA toolkit.
        "$mlmm_python" -m pip download --only-binary=:all: --constraint "$wheel_dir/torch-nvidia-constraints.txt" --dest "$wheel_dir" 'OpenMM[cuda12]==8.6.1'
    fi
    exit 0
fi

if [[ "$mode" == install ]]; then
    if [[ "$backend" == cuda12 ]]; then
        current_constraints=$(cuda_constraints)
        "$mlmm_python" - "$wheel_dir/torch-nvidia-constraints.txt" "$current_constraints" <<'PY'
from pathlib import Path
import sys
expected = {line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")}
current = set(sys.argv[2].splitlines())
if current != expected:
    raise SystemExit("Wheelhouse CUDA constraints differ from the active Torch; export target-server constraints and prepare its wheels again.")
PY
    fi
    before_stack=$("$mlmm_python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import numpy
import torch
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 3):
    raise SystemExit("torch-pme 0.5.0 requires an existing Torch >=2.3; prepare that environment separately.")
for name in ("torch", "torch-geometric", "triton"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "absent")
PY
)
    "$mlmm_python" -m pip install --no-index --find-links "$wheel_dir" --no-deps 'OpenMM==8.6.1' 'torch-pme==0.5.0'
    if [[ "$backend" == cuda12 ]]; then
        "$mlmm_python" -m pip install --no-index --find-links "$wheel_dir" --constraint "$wheel_dir/torch-nvidia-constraints.txt" 'OpenMM-CUDA-12==8.6.1'
    fi
    after_stack=$("$mlmm_python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "torch-geometric", "triton"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "absent")
PY
)
    if [[ "$before_stack" != "$after_stack" ]]; then
        echo "The existing Torch/PyG/Triton stack changed unexpectedly." >&2
        exit 1
    fi
elif [[ "$mode" != check ]]; then
    echo "Mode must be constraints, download, install or check." >&2
    exit 2
fi

"$mlmm_python" - "$backend" <<'PY'
import ctypes
from importlib.metadata import version
from pathlib import Path
import sys
import openmm
import torch
import torchpme

print("OpenMM", openmm.__version__, "torch-pme", version("torch-pme"))
print("Preserved Torch", torch.__version__, "CUDA runtime", torch.version.cuda)
if not hasattr(openmm, "PythonForce"):
    raise SystemExit("This OpenMM build has no PythonForce.")
platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print("OpenMM platforms:", ", ".join(platforms))
if sys.argv[1] == "cuda12":
    nvrtc = [path for root in sys.path if root for path in Path(root).glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*") if not path.name.startswith("libnvrtc.so.") or path.name.count(".") <= 2]
    if nvrtc:
        ctypes.CDLL(str(nvrtc[0]))
        print("NVRTC:", nvrtc[0])
    else:
        try:
            ctypes.CDLL("libnvrtc.so.12")
            print("NVRTC: system libnvrtc.so.12")
        except OSError as exc:
            raise SystemExit("CUDA NVRTC could not be loaded; check the downloaded plugin dependencies or CUDA library path.") from exc
    if "CUDA" not in platforms:
        print("Plugin errors:", openmm.Platform.getPluginLoadFailures())
        raise SystemExit("OpenMM CUDA platform unavailable; verify CUDA plugin, NVRTC and NVIDIA driver.")
    if not torch.cuda.is_available():
        raise SystemExit("Torch CUDA is unavailable; verify the NVIDIA driver and preserved Torch CUDA build.")
PY
