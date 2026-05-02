#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "$(realpath "${BASH_SOURCE[0]}")" )" && pwd )"

cd "$SCRIPT_DIR"
python deploy_mujoco_debug.py g1.yaml
