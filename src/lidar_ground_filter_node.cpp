// src/lidar_ground_filter_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

class LidarGroundFilterNode : public rclcpp::Node
{
public:
  LidarGroundFilterNode()
  : Node("lidar_ground_filter_node")
  {
    // Parameters
    this->declare_parameter<std::string>("input_topic", "/filtered_points");
    this->declare_parameter<std::string>("output_topic", "/no_ground_points");
    this->declare_parameter<double>("distance_threshold", 0.1);
    this->declare_parameter<bool>("optimize_coefficients", true);
    this->declare_parameter<int>("max_iterations", 100);
    this->declare_parameter<double>("angle_threshold", 10.0); // degrees

    input_topic_ = this->get_parameter("input_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    distance_threshold_ = this->get_parameter("distance_threshold").as_double();
    optimize_coefficients_ = this->get_parameter("optimize_coefficients").as_bool();
    max_iterations_ = this->get_parameter("max_iterations").as_int();
    angle_threshold_ = this->get_parameter("angle_threshold").as_double();

    // Subscriber and Publisher
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&LidarGroundFilterNode::pointCloudCallback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // Convert ROS2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Plane segmentation
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(optimize_coefficients_);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold_);
    seg.setMaxIterations(max_iterations_);

    // Optionally, limit the orientation of the plane (assuming ground is roughly horizontal)
    seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0)); // Z-axis
    seg.setEpsAngle(angle_threshold_ * M_PI / 180.0); // Convert degrees to radians

    seg.setInputCloud(cloud);
    seg.segment(*ground_indices, *coefficients);

    if (ground_indices->indices.empty())
    {
      RCLCPP_WARN(this->get_logger(), "No ground plane found.");
      pub_->publish(*cloud_msg); // Publish the original cloud if no ground found
      return;
    }

    // Extract the non-ground points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(ground_indices);
    extract.setNegative(true); // Extract points that are not ground

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>());
    extract.filter(*cloud_no_ground);

    // Convert back to ROS2 message
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud_no_ground, output);
    output.header = cloud_msg->header;

    pub_->publish(output);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  std::string input_topic_;
  std::string output_topic_;
  double distance_threshold_;
  bool optimize_coefficients_;
  int max_iterations_;
  double angle_threshold_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarGroundFilterNode>());
  rclcpp::shutdown();
  return 0;
}
