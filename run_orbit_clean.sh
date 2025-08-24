#!/bin/bash
# This script runs Isaac Lab / Orbit Surgical safely in a temporary environment

(
    # 1️⃣ Start clean
    unset PYTHONPATH
    unset LD_LIBRARY_PATH
    hash -r

    # 2️⃣ Load Isaac Sim Python environment temporarily
    source ~/isaacsim/setup_python_env.sh

    # 3️⃣ Activate your Orbit Surgical Conda environment
    conda activate orbitsurgical

    # 4️⃣ Run the passed command (like python train.py ...)
    "$@"
)
