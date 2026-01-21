#!/bin/bash
set -e

# 获取脚本所在目录
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
cd "$SCRIPT_DIR"

echo "============================================="
echo "   Compiling NanoTrack C++ Workspace        "
echo "   WorkDir: $SCRIPT_DIR"
echo "============================================="

# 赋予脚本执行权限
chmod +x scripts/*.py

# 编译
echo "Building recomo_msgs..."
colcon build --packages-select recomo_msgs --paths recomo_msgs --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source 刚刚编译的 msgs，以便 nanotrack_cpp 能找到它
if [ -f install/setup.bash ]; then
    source install/setup.bash
fi

echo "Building nanotrack_cpp..."
colcon build --packages-select nanotrack_cpp --paths . --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ $? -eq 0 ]; then
    echo "============================================="
    echo "   Build Success!                            "
    echo "============================================="
else
    echo "============================================="
    echo "   Build Failed!                             "
    echo "============================================="
    exit 1
fi
