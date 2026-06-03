#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD="${PYTHON:-python}"
ASSET_DIR="${REPO_ROOT}/source/tritonhumanoid/tritonhumanoid/assets/mujoco"
MJCF_PATH="${ASSET_DIR}/ch_robot_10dof.xml"

mkdir -p "${ASSET_DIR}"

"${PYTHON_CMD}" - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "mujoco": "mujoco==3.9.0",
    "glfw": "glfw",
    "OpenGL": "PyOpenGL",
    "numpy": "numpy",
}
missing = [pip_name for module, pip_name in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("[INFO] MuJoCo Python runtime dependencies already installed.")
PY

if [[ ! -f "${MJCF_PATH}" ]]; then
  cat > "${MJCF_PATH}" <<'XML'
<!-- PLACEHOLDER_MJCF: replace with the canonical ch_robot_10dof.xml before MuJoCo eval. -->
<mujoco model="placeholder_ch_robot">
  <worldbody>
    <body name="placeholder"/>
  </worldbody>
</mujoco>
XML
fi

set +e
PYTHONPATH="${REPO_ROOT}/source/tritonhumanoid:${PYTHONPATH:-}" \
  "${PYTHON_CMD}" "${REPO_ROOT}/scripts/mujoco_eval_locomotion.py" --validate-only
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  cat <<EOF
[WARN] MuJoCo runtime installed, but model validation did not pass.
[WARN] Replace the placeholder/canonical source MJCF here:
       ${MJCF_PATH}

Next checks after replacing the MJCF:
  PYTHONPATH=${REPO_ROOT}/source/tritonhumanoid:\${PYTHONPATH:-} ${PYTHON_CMD} scripts/mujoco_eval_locomotion.py --validate-only
EOF
  exit 0
fi

cat <<EOF
[INFO] MuJoCo locomotion setup is valid.

Useful commands:
  ${PYTHON_CMD} scripts/mujoco_eval_locomotion.py --validate-only
  ${PYTHON_CMD} scripts/mujoco_playback_locomotion.py --trace logs/parity/<trace>.npz --validate-only
EOF

