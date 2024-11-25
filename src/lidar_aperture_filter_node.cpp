// src/lidar_aperture_filter_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>

class LidarApertureFilterNode : public rclcpp::Node
{
public:
  LidarApertureFilterNode()
  : Node("lidar_aperture_filter_node")
  {
    // Declare and get parameters
    this->declare_parameter<std::string>("input_topic", "/no_ground_points");
    this->declare_parameter<std::string>("output_topic", "/filtered_fov_points");
    this->declare_parameter<double>("min_azimuth_angle", -60.0); // degrees
    this->declare_parameter<double>("max_azimuth_angle", 60.0);  // degrees
    this->declare_parameter<double>("min_elevation_angle", -30.0); // degrees
    this->declare_parameter<double>("max_elevation_angle", 30.0);  // degrees

    input_topic_ = this->get_parameter("input_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    min_azimuth_angle_ = this->get_parameter("min_azimuth_angle").as_double();
    max_azimuth_angle_ = this->get_parameter("max_azimuth_angle").as_double();
    min_elevation_angle_ = this->get_parameter("min_elevation_angle").as_double();
    max_elevation_angle_ = this->get_parameter("max_elevation_angle").as_double();

    // Convert degrees to radians
    min_azimuth_rad_ = min_azimuth_angle_ * M_PI / 180.0;
    max_azimuth_rad_ = max_azimuth_angle_ * M_PI / 180.0;
    min_elevation_rad_ = min_elevation_angle_ * M_PI / 180.0;
    max_elevation_rad_ = max_elevation_angle_ * M_PI / 180.0;

    // Subscriber and Publisher
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&LidarApertureFilterNode::pointCloudCallback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // Convert ROS2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // Iterate over points and filter based on azimuth and elevation angles
    for (const auto& point : cloud->points)
    {
      // Calculate azimuth angle (theta) and elevation angle (phi)
      double azimuth = std::atan2(point.y, point.x); // [-pi, pi]
      double range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      if (range == 0) continue; // Avoid division by zero
      double elevation = std::asin(point.z / range); // [-pi/2, pi/2]

      // Check if the point is within the desired angular ranges
      if (azimuth >= min_azimuth_rad_ && azimuth <= max_azimuth_rad_ &&
          elevation >= min_elevation_rad_ && elevation <= max_elevation_rad_)
      {
        filtered_cloud->points.push_back(point);
      }
    }

    // Set header and publish filtered point cloud
    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*filtered_cloud, output_msg);
    output_msg.header = cloud_msg->header;

    pub_->publish(output_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  std::string input_topic_;
  std::string output_topic_;
  double min_azimuth_angle_;
  double max_azimuth_angle_;
  double min_elevation_angle_;
  double max_elevation_angle_;
  double min_azimuth_rad_;
  double max_azimuth_rad_;
  double min_elevation_rad_;
  double max_elevation_rad_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarApertureFilterNode>());
  rclcpp::shutdown();
  return 0;
}
