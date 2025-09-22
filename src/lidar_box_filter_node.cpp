#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>

class LidarBoxFilterNode : public rclcpp::Node
{
public:
  LidarBoxFilterNode()
  : Node("lidar_box_filter_node")
  {
    // Parameters for the bounding box
    this->declare_parameter<std::vector<float>>("min_point", std::vector<float>{-1.0, -1.0, -1.0});
    this->declare_parameter<std::vector<float>>("max_point", std::vector<float>{1.0, 1.0, 1.0});
    this->declare_parameter<std::string>("input_topic", "/points");
    this->declare_parameter<std::string>("output_topic", "/filtered_points");

    min_point_ = this->get_parameter("min_point").as_double_array();
    max_point_ = this->get_parameter("max_point").as_double_array();
    input_topic_ = this->get_parameter("input_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();

    // Create publishers with appropriate QoS
    auto sensor_qos = rclcpp::SensorDataQoS();
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, sensor_qos);
    
    // Visualization marker can use default QoS (reliable)
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("filter_box_marker", 10);

    // Create subscription with sensor data QoS
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sensor_qos,
      std::bind(&LidarBoxFilterNode::pointCloudCallback, this, std::placeholders::_1));

    // Publish the initial marker
    //publishMarker();

    // Publish the marker periodically
    timer_ = this->create_wall_timer(
    std::chrono::milliseconds(500),
    std::bind(&LidarBoxFilterNode::publishMarker, this));

    // Add parameter callback
    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&LidarBoxFilterNode::parameterCallback, this, std::placeholders::_1));
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

  rcl_interfaces::msg::SetParametersResult parameterCallback(
    const std::vector<rclcpp::Parameter> &parameters)
  {
    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;
    for (const auto &param : parameters) {
      if (param.get_name() == "min_point") {
        min_point_ = param.as_double_array();
      } else if (param.get_name() == "max_point") {
        max_point_ = param.as_double_array();
      } else {
        result.successful = false;
        result.reason = "Invalid parameter";
        return result;
      }
    }

    // Update the marker
    publishMarker();

    return result;
  }

  void publishMarker()
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "base_link"; // Adjust the frame as necessary
    marker.header.stamp = this->now();
    marker.ns = "filter_box";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Calculate the center of the box
    float x_center = (min_point_[0] + max_point_[0]) / 2.0;
    float y_center = (min_point_[1] + max_point_[1]) / 2.0;
    float z_center = (min_point_[2] + max_point_[2]) / 2.0;

    marker.pose.position.x = x_center;
    marker.pose.position.y = y_center;
    marker.pose.position.z = z_center;

    marker.pose.orientation.w = 1.0;

    // Calculate the scale of the box
    marker.scale.x = std::abs(max_point_[0] - min_point_[0]);
    marker.scale.y = std::abs(max_point_[1] - min_point_[1]);
    marker.scale.z = std::abs(max_point_[2] - min_point_[2]);

    // Set the color
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 0.5; // Semi-transparent

    marker.lifetime = rclcpp::Duration(0, 0); // 0 means forever

    marker_pub_->publish(marker);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
  rclcpp::TimerBase::SharedPtr timer_;

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
