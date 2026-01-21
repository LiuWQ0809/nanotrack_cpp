#!/bin/bash

# 检查是否已经运行
if pgrep -f "nanotrack_tracker.launch.py" > /dev/null; then
    echo "[WARNING] NanoTrack appears to be already running!"
    echo "PIDs:"
    pgrep -f "nanotrack_tracker.launch.py" -a
    echo "Please stop it first using ./stop.sh"
    exit 1
fi

SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
cd "$SCRIPT_DIR"

echo "Running NanoTrack..."
source install/setup.bash

# Ensure DISPLAY is set for OpenCV GUI
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "[INFO] DISPLAY was not set, exporting DISPLAY=:0"
fi

# 启动 Launch 文件（后台）
LOG_FILE="${SCRIPT_DIR}/nanotrack_tracker.log"
nohup ros2 launch nanotrack_cpp nanotrack_tracker.launch.py > "${LOG_FILE}" 2>&1 &
echo "[INFO] NanoTrack started in background (PID: $!)"
echo "[INFO] Log: ${LOG_FILE}"
