# LiDAR Filter Package

A comprehensive ROS 2 package for LiDAR point cloud filtering, providing multiple filtering capabilities including ground plane removal, box filtering, and aperture (field-of-view) filtering. The package offers both standalone filter nodes and a combined filter node for flexible processing pipelines.

## Features

- **Ground Plane Filtering**: Removes ground points using RANSAC plane segmentation
- **Box Filtering**: Filters points within or outside a 3D bounding box
- **Aperture Filtering**: Limits point cloud to specific azimuth and elevation angles
- **Combined Filtering**: Single node that applies multiple filters in sequence
- **Dynamic Parameter Reconfiguration**: Real-time parameter updates
- **Visualization Support**: Bounding box markers for RViz visualization
- **Adaptive QoS**: Automatically handles both RELIABLE and BEST_EFFORT publishers using SensorDataQoS

## Package Contents

### Nodes

1. **`lidar_ground_filter_node`** - Standalone ground plane removal
2. **`lidar_box_filter_node`** - Standalone 3D box filtering with visualization
3. **`lidar_aperture_filter_node`** - Standalone field-of-view filtering
4. **`combined_lidar_filter_node`** - All filters in one configurable node

### Dependencies

- ROS 2 Humble
- PCL (Point Cloud Library) 1.12+
- pcl_ros
- pcl_conversions
- sensor_msgs
- visualization_msgs

## Installation

### Prerequisites

Make sure you have ROS 2 Humble installed and sourced:

```bash
source /opt/ros/humble/setup.bash
```

### Install Dependencies

```bash
sudo apt update
sudo apt install -y \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-sensor-msgs \
    ros-humble-visualization-msgs \
    libpcl-dev
```

### Build the Package

```bash
cd ~/ros2_ws/src
git clone <your-repository-url> lidar_filter  # Replace with actual repository URL
cd ~/ros2_ws
colcon build --packages-select lidar_filter
source install/setup.bash
```

## Usage

### Combined Filter Node (Recommended)

The combined filter node provides the most flexibility by allowing you to enable/disable individual filters:

```bash
ros2 run lidar_filter combined_lidar_filter_node
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `/points` | Input point cloud topic |
| `output_topic` | string | `/filtered_points` | Output filtered point cloud topic |
| `enable_aperture_filter` | bool | `true` | Enable/disable aperture filtering |
| `enable_box_filter` | bool | `true` | Enable/disable box filtering |
| `enable_ground_filter` | bool | `true` | Enable/disable ground filtering |

**Aperture Filter Parameters:**
- `min_azimuth_angle` (double, default: -60.0°): Minimum azimuth angle
- `max_azimuth_angle` (double, default: 60.0°): Maximum azimuth angle
- `min_elevation_angle` (double, default: -30.0°): Minimum elevation angle
- `max_elevation_angle` (double, default: 30.0°): Maximum elevation angle

**Box Filter Parameters:**
- `min_point` (float array, default: [-0.5, -1.5, -1.2]): Minimum corner of box
- `max_point` (float array, default: [3.0, 1.5, 1.0]): Maximum corner of box
- `box_filter_negative` (bool, default: true): If true, removes points inside box

**Ground Filter Parameters:**
- `distance_threshold` (double, default: 0.2): RANSAC distance threshold
- `optimize_coefficients` (bool, default: true): Optimize plane coefficients
- `max_iterations` (int, default: 100): Maximum RANSAC iterations
- `angle_threshold` (double, default: 15.0°): Angle threshold for plane normal

### Standalone Nodes

#### Ground Filter

```bash
ros2 run lidar_filter lidar_ground_filter_node \
    --ros-args \
    -p input_topic:=/points \
    -p output_topic:=/no_ground_points \
    -p distance_threshold:=0.1
```

#### Box Filter

```bash
ros2 run lidar_filter lidar_box_filter_node \
    --ros-args \
    -p input_topic:=/no_ground_points \
    -p output_topic:=/box_filtered_points \
    -p min_point:="[-1.0, -1.0, -1.0]" \
    -p max_point:="[1.0, 1.0, 1.0]"
```

#### Aperture Filter

```bash
ros2 run lidar_filter lidar_aperture_filter_node \
    --ros-args \
    -p input_topic:=/box_filtered_points \
    -p output_topic:=/filtered_fov_points \
    -p min_azimuth_angle:=-60.0 \
    -p max_azimuth_angle:=60.0
```

### Processing Pipeline

The typical processing order is:
1. **Aperture Filter**: Reduce data by limiting field of view
2. **Box Filter**: Remove ego-vehicle points or focus on specific regions
3. **Ground Filter**: Remove ground plane for obstacle detection

## Configuration Examples

### Launch File Example

Create a launch file `launch/lidar_filter.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_filter',
            executable='combined_lidar_filter_node',
            name='lidar_filter',
            parameters=[{
                'input_topic': '/os0_128_10hz_512_rev7/points',
                'output_topic': '/filtered_points',
                'enable_aperture_filter': True,
                'enable_box_filter': True,
                'enable_ground_filter': True,
                'min_azimuth_angle': -90.0,
                'max_azimuth_angle': 90.0,
                'min_elevation_angle': -30.0,
                'max_elevation_angle': 30.0,
                'min_point': [-0.5, -2.0, -1.5],
                'max_point': [4.0, 2.0, 1.0],
                'box_filter_negative': True,
                'distance_threshold': 0.15,
                'max_iterations': 150
            }],
            output='screen'
        )
    ])
```

### Parameter File Example

Create `config/lidar_filter_params.yaml`:

```yaml
lidar_filter:
  ros__parameters:
    input_topic: "/points"
    output_topic: "/filtered_points"
    
    # Enable/disable filters
    enable_aperture_filter: true
    enable_box_filter: true
    enable_ground_filter: true
    
    # Aperture filter (angles in degrees)
    min_azimuth_angle: -60.0
    max_azimuth_angle: 60.0
    min_elevation_angle: -20.0
    max_elevation_angle: 20.0
    
    # Box filter (coordinates in meters)
    min_point: [-0.5, -1.5, -1.2]
    max_point: [3.0, 1.5, 1.0]
    box_filter_negative: true
    
    # Ground filter
    distance_threshold: 0.2
    optimize_coefficients: true
    max_iterations: 100
    angle_threshold: 15.0
```

Run with parameter file:
```bash
ros2 run lidar_filter combined_lidar_filter_node --ros-args --params-file config/lidar_filter_params.yaml
```

## Visualization

The box filter publishes visualization markers on the `/filter_box_marker` topic. View in RViz by:

1. Add a "Marker" display
2. Set the topic to `/filter_box_marker`
3. The green semi-transparent box shows the filtering region

## Dynamic Reconfiguration

All parameters can be updated at runtime:

```bash
# Change aperture angles
ros2 param set /combined_lidar_filter_node min_azimuth_angle -45.0
ros2 param set /combined_lidar_filter_node max_azimuth_angle 45.0

# Adjust box filter
ros2 param set /combined_lidar_filter_node min_point "[-1.0, -2.0, -1.0]"

# Toggle filters
ros2 param set /combined_lidar_filter_node enable_ground_filter false
```

## QoS Compatibility

All nodes now use **SensorDataQoS** profile which provides:
- **BEST_EFFORT** reliability (compatible with most LiDAR drivers)
- **VOLATILE** durability (no message persistence)
- **SYSTEM_DEFAULT** history (typically keeps last few messages)

This ensures compatibility with:
- ✅ Ouster LiDAR drivers (BEST_EFFORT)
- ✅ Velodyne LiDAR drivers (BEST_EFFORT) 
- ✅ Standard ROS 2 sensor data publishers
- ✅ Custom publishers using either RELIABLE or BEST_EFFORT

**No more QoS incompatibility warnings!**

## Troubleshooting

### Common Issues

1. **Build fails with "Could not find pcl_ros"**:
   ```bash
   sudo apt install ros-humble-pcl-ros
   ```

2. **No output points**: Check that input topic is publishing and parameters are reasonable

3. **Performance issues**: 
   - Disable unused filters
   - Reduce RANSAC iterations for ground filter
   - Use aperture filter first to reduce point count

4. **QoS Issues (Legacy)**: If you encounter QoS warnings with older versions, all nodes now use SensorDataQoS which is compatible with most sensor data publishers

### Debugging

Enable debug logging:
```bash
ros2 run lidar_filter combined_lidar_filter_node --ros-args --log-level debug
```

Check topics:
```bash
ros2 topic list | grep points
ros2 topic echo /filtered_points --field data
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## License

This package is licensed under the Apache License 2.0. See the LICENSE file for details.

## Maintainer

- **User** (gunreben@wifa.uni-leipzig.de)

## Version History

- **0.1.0**: Initial release with ground, box, and aperture filtering capabilities