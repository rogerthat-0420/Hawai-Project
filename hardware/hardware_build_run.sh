#!/bin/bash

# ==============================================================================
# H2O Hardware Build & Run Script with Logging
# Usage: ./build_and_run.sh <run_name>
# Example: ./build_and_run.sh experiment_01
# ==============================================================================

# 1. Validate Input
if [ -z "$1" ]; then
    echo "Error: No run name provided."
    echo "Usage: $0 <run_name>"
    exit 1
fi

RUN_NAME="$1"
LOG_DIR="output_logs/${RUN_NAME}"

mkdir -p "$LOG_DIR"
echo "============================================================"
echo " Starting Build & Run: ${RUN_NAME}"
echo " Logs will be saved to: ${LOG_DIR}/"
echo "============================================================"

# if [ -z "$XILINX_XRT" ]; then
#     echo "[1/4] Setting up XRT Environment..."
#     if [ -f /opt/xilinx/xrt/setup.sh ]; then
#         source /opt/xilinx/xrt/setup.sh
#     else
#         echo "Error: /opt/xilinx/xrt/setup.sh not found." | tee "${LOG_DIR}/system.log"
#         exit 1
#     fi
# fi

echo "[1/3] Building Host Application..."
# We use '2>&1 | tee' to show output on screen AND save to log
make host > "${LOG_DIR}/host.log" 2>&1

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Host build failed. See ${LOG_DIR}/host.log for details."
    exit 1
fi
echo " - Host built successfully."

# 5. Build Kernel (Synthesis)
echo "[2/3] Building Hardware Kernel (TARGET=hw)..."
echo "      (This takes 2-4 hours. Logs streaming to ${LOG_DIR}/kernel.log)"

# Use 'stdbuf' or unbuffered pipe if possible, but standard redirection is fine.
# We do NOT use 'tee' here to keep the terminal clean for this long process, 
# but you can 'tail -f output_logs/<name>/kernel.log' in another terminal.
make kernel TARGET=hw > "${LOG_DIR}/kernel.log" 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Kernel synthesis failed. See ${LOG_DIR}/kernel.log for details."
    exit 1
fi
echo " - Kernel built successfully."

# # 6. Run on Hardware
# echo "[3/3] Running on Alveo U50..."
# make run TARGET=hw > "${LOG_DIR}/run.log" 2>&1

# if [ $? -ne 0 ]; then
#     echo "ERROR: Execution failed. See ${LOG_DIR}/run.log for details."
#     exit 1
# fi

echo "============================================================"
echo " SUCCESS! Build Complete."
echo " Host Log:   ${LOG_DIR}/host.log"
echo " Kernel Log: ${LOG_DIR}/kernel.log"
# echo " Run Log:    ${LOG_DIR}/run.log"
echo "============================================================"