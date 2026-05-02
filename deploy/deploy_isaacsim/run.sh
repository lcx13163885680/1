#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "$(realpath "${BASH_SOURCE[0]}")" )" && pwd )"
ISAAC_SIM_DIR="/home/linchenxu/miniconda3/envs/unitree-demo/isaac-sim-4.5.0"
VRAM_THRESHOLD_MB=7200

if [ ! -d "$ISAAC_SIM_DIR" ]; then
    echo "Error: Isaac Sim not found at $ISAAC_SIM_DIR"
    exit 1
fi

# VRAM monitor daemon - kills python if VRAM exceeds threshold
start_vram_monitor() {
    local target_pid=$1
    echo "[VRAM Monitor] Watching PID $target_pid, threshold ${VRAM_THRESHOLD_MB}MB"
    while kill -0 "$target_pid" 2>/dev/null; do
        sleep 2
        if command -v nvidia-smi &> /dev/null; then
            vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
            if [ -n "$vram_used" ] && [ "$vram_used" -gt "$VRAM_THRESHOLD_MB" ]; then
                echo ""
                echo "[VRAM Monitor] CRITICAL: VRAM ${vram_used}MB > ${VRAM_THRESHOLD_MB}MB"
                echo "[VRAM Monitor] Killing python process to prevent system freeze..."
                kill -TERM "$target_pid" 2>/dev/null
                sleep 1
                kill -KILL "$target_pid" 2>/dev/null
                exit 1
            fi
        fi
    done
}

if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda environment detected: $CONDA_PREFIX"

    # Save our script dir before sourcing (setup_conda_env.sh overwrites SCRIPT_DIR)
    DEPLOY_DIR="$SCRIPT_DIR"

    source "$ISAAC_SIM_DIR/setup_conda_env.sh"
    echo "Isaac Sim environment configured."

    # Optimize CUDA memory for RTX 4070 8GB
    export CUDA_MODULE_LOADING=LAZY
    export OMP_NUM_THREADS=8

    echo "Launching G1 policy with motion.pt..."
    echo "Script path: $DEPLOY_DIR/play_g1_isaacsim.py"

    # Start python in background and monitor its VRAM
    python "$DEPLOY_DIR/play_g1_isaacsim.py" "$@" &
    PYTHON_PID=$!

    start_vram_monitor $PYTHON_PID &
    MONITOR_PID=$!

    wait $PYTHON_PID
    PYTHON_EXIT=$?
    kill $MONITOR_PID 2>/dev/null
    exit $PYTHON_EXIT
else
    echo "No conda environment detected. Please run: conda activate unitree-demo"
    exit 1
fi
