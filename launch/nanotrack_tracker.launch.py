import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('nanotrack_cpp')
    
    # Check if models are in share directory, else use env
    # The C++ node logic tries to find them.
    
    return LaunchDescription([
        Node(
            package='nanotrack_cpp',
            executable='nanotrack_tracker',
            name='nanotrack_tracker',
            output='screen',
            parameters=[{
                'image_topic': '/recomo/rgb',
                'visualization_topic': '/nanotrack/visualization',
                'target_width': 640,
                'target_height': 480,
                # Engine paths relative to resolved engine_dir
                'backbone_template_engine': 'nanotrack_backbone_127_fp16.engine',
                'backbone_search_engine': 'nanotrack_backbone_255_fp16.engine',
                'head_engine': 'nanotrack_head_fp16.engine',
                'min_confidence': 0.99,
                'track_lost_threshold': 0.8
            }]
        ),
        Node(
            package='nanotrack_cpp',
            executable='visualizer_node.py',
            name='nanotrack_visualizer',
            output='screen',
            parameters=[{
                'visualization_topic': '/nanotrack/visualization',
                'target_roi_topic': '/target_roi'
            }]
        )
    ])
