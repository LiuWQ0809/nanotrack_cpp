#!/bin/bash

echo "Stopping NanoTrack Nodes..."

# 杀掉 Launch 进程
pkill -f "nanotrack_tracker.launch.py"

# 杀掉 C++ 节点进程
pkill -f "nanotrack_tracker"

# 杀掉 Python 可视化节点进程
pkill -f "visualizer_node.py"

echo "Cleanup done."
