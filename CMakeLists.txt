cmake_minimum_required(VERSION 3.8)
project(nanotrack_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(CUDA REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(recomo_msgs QUIET)
find_package(ament_index_cpp REQUIRED)

# Setup CUDA
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
enable_language(CUDA)

# Find recomo_msgs
find_package(recomo_msgs QUIET)

# TensorRT is usually in standard search paths on JetPack
find_library(NVINFER_LIB nvinfer)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)
find_library(NVPARSERS_LIB nvparsers)
find_library(NVONNXPARSER_LIB nvonnxparser)

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT not found")
endif()

include_directories(include)

# CUDA Preprocessor Library
cuda_add_library(nanotrack_preprocessor
  src/preprocessor.cu
)
target_include_directories(nanotrack_preprocessor PUBLIC include)

# Main Library
add_library(nanotrack_lib
  src/tensorrt_engine.cpp
  src/nanotrack.cpp
  src/kalman_filter.cpp
)
target_include_directories(nanotrack_lib PUBLIC include)
ament_target_dependencies(nanotrack_lib rclcpp)
target_link_libraries(nanotrack_lib 
  nanotrack_preprocessor
  ${NVINFER_LIB} 
  ${NVINFER_PLUGIN_LIB}
  ${CUDA_LIBRARIES}
)

# ROS 2 Node
add_executable(nanotrack_tracker src/tracker_node.cpp)
ament_target_dependencies(nanotrack_tracker rclcpp sensor_msgs std_msgs ament_index_cpp)
target_link_libraries(nanotrack_tracker nanotrack_lib)

# check for recomo_msgs, if exists link it, else compile define
if (recomo_msgs_FOUND)
  ament_target_dependencies(nanotrack_tracker recomo_msgs)
  target_compile_definitions(nanotrack_tracker PUBLIC HAS_RECOMO_MSGS)
else()
  message(WARNING "recomo_msgs not found. Tracker node will NOT publish tracking message.")
endif()

install(TARGETS nanotrack_lib nanotrack_tracker
  EXPORT export_nanotrack_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Copies
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/visualizer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(nanotrack_lib nanotrack_preprocessor)

ament_package()
