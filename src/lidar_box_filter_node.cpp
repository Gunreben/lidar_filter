// src/lidar_filter_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <cmath>


class LidarBoxFilterNode : public rclcpp::Node
{
public:
  LidarBoxFilterNode()
  : Node("lidar_box_filter_node")
  {
    // Parameters for the bounding box (adjust these values to match the cowling's position)
    this->declare_parameter<std::vector<float>>("min_point", std::vector<float>{-1.0, -1.0, -1.0});
    this->declare_parameter<std::vector<float>>("max_point", std::vector<float>{1.0, 1.0, 1.0});
    this->declare_parameter<std::string>("input_topic", "/points");
    this->declare_parameter<std::string>("output_topic", "/filtered_points");

    min_point_ = this->get_parameter("min_point").as_double_array();
    max_point_ = this->get_parameter("max_point").as_double_array();
    input_topic_ = this->get_parameter("input_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();

    // Subscriber and Publisher
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&LidarBoxFilterNode::pointCloudCallback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // Convert ROS2 message to PCL PointCloud
    pcl::PCLPointCloud2::Ptr pcl_pc2(new pcl::PCLPointCloud2());
    pcl_conversions::toPCL(*cloud_msg, *pcl_pc2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromPCLPointCloud2(*pcl_pc2, *temp_cloud);

    // Set up the crop box filter
    pcl::CropBox<pcl::PointXYZ> box_filter;
    box_filter.setMin(Eigen::Vector4f(min_point_[0], min_point_[1], min_point_[2], 1.0));
    box_filter.setMax(Eigen::Vector4f(max_point_[0], max_point_[1], max_point_[2], 1.0));
    box_filter.setNegative(true); // Remove points inside the box

    box_filter.setInputCloud(temp_cloud);
    box_filter.filter(*temp_cloud);

    // Convert back to ROS2 message
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*temp_cloud, output);
    output.header = cloud_msg->header;

    pub_->publish(output);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  std::vector<double> min_point_;
  std::vector<double> max_point_;
  std::string input_topic_;
  std::string output_topic_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarBoxFilterNode>());
  rclcpp::shutdown();
  return 0;
}
