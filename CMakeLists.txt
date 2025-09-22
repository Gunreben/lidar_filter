cmake_minimum_required(VERSION 3.8)
project(lidar_filter)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(PCL REQUIRED COMPONENTS 
  common 
  io 
  filters 
  segmentation
)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Define common dependencies for all nodes
set(LIDAR_FILTER_DEPS
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
  visualization_msgs
  tf2_ros
  tf2_eigen
  PCL
)

# Combined filter node
add_executable(combined_lidar_filter_node src/combined_lidar_filter_node.cpp)
ament_target_dependencies(combined_lidar_filter_node
  ${LIDAR_FILTER_DEPS}
  visualization_msgs
)
target_link_libraries(combined_lidar_filter_node ${PCL_LIBRARIES})

# Install targets
install(TARGETS
  combined_lidar_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()