#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>
#include <pcl/common/transforms.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>

class CombinedLidarFilterNode : public rclcpp::Node
{
public:
  CombinedLidarFilterNode()
  : Node("combined_lidar_filter_node")
  {
    // Common parameters
    this->declare_parameter<std::string>("input_topic", "/points");
    this->declare_parameter<std::string>("output_topic", "/filtered_points");
    this->declare_parameter<std::string>("target_frame", "base_link");
    
    // Aperture filter parameters
    this->declare_parameter<bool>("enable_aperture_filter", true);
    this->declare_parameter<double>("min_azimuth_angle", -60.0);
    this->declare_parameter<double>("max_azimuth_angle", 60.0);
    this->declare_parameter<double>("min_elevation_angle", -30.0);
    this->declare_parameter<double>("max_elevation_angle", 30.0);

    // Box filter parameters
    this->declare_parameter<bool>("enable_box_filter", true);
    this->declare_parameter<std::vector<float>>("min_point", std::vector<float>{-0.5, -1.5, -1.2});
    this->declare_parameter<std::vector<float>>("max_point", std::vector<float>{3.0, 1.5, 1.0});
    this->declare_parameter<bool>("box_filter_negative", true);

    // Ground filter parameters
    this->declare_parameter<bool>("enable_ground_filter", true);
    this->declare_parameter<double>("distance_threshold", 0.2);
    this->declare_parameter<bool>("optimize_coefficients", true);
    this->declare_parameter<int>("max_iterations", 100);
    this->declare_parameter<double>("angle_threshold", 15.0);

    // Load parameters
    loadParameters();

    // TF setup
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Initialize publishers with appropriate QoS
    auto sensor_qos = rclcpp::SensorDataQoS();
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, sensor_qos);
    
    if (enable_box_filter_) {
      // Visualization markers can use default QoS (reliable)
      marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("filter_box_marker", 10);
      marker_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&CombinedLidarFilterNode::publishBoxMarker, this));
    }

    // Main subscriber with sensor data QoS
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sensor_qos,
      std::bind(&CombinedLidarFilterNode::pointCloudCallback, this, std::placeholders::_1));

    // Parameter callback
    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&CombinedLidarFilterNode::parameterCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Node initialized with filters: Aperture=%s, Box=%s, Ground=%s",
                enable_aperture_filter_ ? "enabled" : "disabled",
                enable_box_filter_ ? "enabled" : "disabled",
                enable_ground_filter_ ? "enabled" : "disabled");
  }

private:
  void loadParameters() {
    // Load common parameters
    input_topic_ = this->get_parameter("input_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    if (this->has_parameter("target_frame")) {
      target_frame_ = this->get_parameter("target_frame").as_string();
    }
    
    // Load aperture filter parameters
    enable_aperture_filter_ = this->get_parameter("enable_aperture_filter").as_bool();
    if (enable_aperture_filter_) {
      min_azimuth_angle_ = this->get_parameter("min_azimuth_angle").as_double();
      max_azimuth_angle_ = this->get_parameter("max_azimuth_angle").as_double();
      min_elevation_angle_ = this->get_parameter("min_elevation_angle").as_double();
      max_elevation_angle_ = this->get_parameter("max_elevation_angle").as_double();

      // Convert angles to radians
      min_azimuth_rad_ = min_azimuth_angle_ * M_PI / 180.0;
      max_azimuth_rad_ = max_azimuth_angle_ * M_PI / 180.0;
      min_elevation_rad_ = min_elevation_angle_ * M_PI / 180.0;
      max_elevation_rad_ = max_elevation_angle_ * M_PI / 180.0;
    }

    // Load box filter parameters
    enable_box_filter_ = this->get_parameter("enable_box_filter").as_bool();
    if (enable_box_filter_) {
      min_point_ = this->get_parameter("min_point").as_double_array();
      max_point_ = this->get_parameter("max_point").as_double_array();
      box_filter_negative_ = this->get_parameter("box_filter_negative").as_bool();
    }

    // Load ground filter parameters
    enable_ground_filter_ = this->get_parameter("enable_ground_filter").as_bool();
    if (enable_ground_filter_) {
      distance_threshold_ = this->get_parameter("distance_threshold").as_double();
      optimize_coefficients_ = this->get_parameter("optimize_coefficients").as_bool();
      max_iterations_ = this->get_parameter("max_iterations").as_int();
      angle_threshold_ = this->get_parameter("angle_threshold").as_double();
    }
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // Convert ROS2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Transform once to target_frame if needed
    std::string source_frame = cloud_msg->header.frame_id;
    if (!target_frame_.empty() && source_frame != target_frame_) {
      try {
        geometry_msgs::msg::TransformStamped tf_stamped =
          tf_buffer_->lookupTransform(target_frame_, source_frame, tf2::TimePointZero);
        Eigen::Isometry3d T_eig = tf2::transformToEigen(tf_stamped.transform);
        pcl::transformPointCloud(*cloud, *cloud, T_eig.cast<float>().matrix());
        source_frame = target_frame_;
      } catch (const std::exception &ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
          "TF lookup failed (%s -> %s): %s", source_frame.c_str(), target_frame_.c_str(), ex.what());
      }
    }

    // Apply filters sequentially if enabled
    if (enable_aperture_filter_) {
      cloud = applyApertureFilter(cloud);
    }

    if (enable_box_filter_) {
      cloud = applyBoxFilter(cloud);
    }

    if (enable_ground_filter_) {
      cloud = applyGroundFilter(cloud);
    }

    // Publish final filtered cloud
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = cloud_msg->header;
    if (!target_frame_.empty()) {
      output_msg.header.frame_id = source_frame; // target_frame_ if transform succeeded
    }
    pub_->publish(output_msg);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr applyApertureFilter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto& point : input_cloud->points) {
      double azimuth = std::atan2(point.y, point.x);
      double range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      if (range == 0) continue;
      
      double elevation = std::asin(point.z / range);

      if (azimuth >= min_azimuth_rad_ && azimuth <= max_azimuth_rad_ &&
          elevation >= min_elevation_rad_ && elevation <= max_elevation_rad_) {
        filtered_cloud->points.push_back(point);
      }
    }

    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;

    return filtered_cloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr applyBoxFilter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CropBox<pcl::PointXYZ> box_filter;
    
    box_filter.setMin(Eigen::Vector4f(min_point_[0], min_point_[1], min_point_[2], 1.0));
    box_filter.setMax(Eigen::Vector4f(max_point_[0], max_point_[1], max_point_[2], 1.0));
    box_filter.setNegative(box_filter_negative_);
    box_filter.setInputCloud(input_cloud);
    box_filter.filter(*filtered_cloud);

    return filtered_cloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr applyGroundFilter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(optimize_coefficients_);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold_);
    seg.setMaxIterations(max_iterations_);
    seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
    seg.setEpsAngle(angle_threshold_ * M_PI / 180.0);

    seg.setInputCloud(input_cloud);
    seg.segment(*ground_indices, *coefficients);

    if (ground_indices->indices.empty()) {
      RCLCPP_WARN(this->get_logger(), "No ground plane found.");
      return input_cloud;
    }

    // Extract non-ground points
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(ground_indices);
    extract.setNegative(true);
    extract.filter(*filtered_cloud);

    return filtered_cloud;
  }

  void publishBoxMarker()
  {
    if (!enable_box_filter_) return;

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = target_frame_.empty() ? "base_link" : target_frame_;
    marker.header.stamp = this->now();
    marker.ns = "filter_box";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = (min_point_[0] + max_point_[0]) / 2.0;
    marker.pose.position.y = (min_point_[1] + max_point_[1]) / 2.0;
    marker.pose.position.z = (min_point_[2] + max_point_[2]) / 2.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = std::abs(max_point_[0] - min_point_[0]);
    marker.scale.y = std::abs(max_point_[1] - min_point_[1]);
    marker.scale.z = std::abs(max_point_[2] - min_point_[2]);

    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 0.5;
    marker.lifetime = rclcpp::Duration(0, 0);

    marker_pub_->publish(marker);
  }

  rcl_interfaces::msg::SetParametersResult parameterCallback(
    const std::vector<rclcpp::Parameter>& parameters)
  {
    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;

    bool needs_param_reload = false;
    for (const auto& param : parameters) {
      if (param.get_name().find("min_") != std::string::npos ||
          param.get_name().find("max_") != std::string::npos ||
          param.get_name().find("enable_") != std::string::npos ||
          param.get_name() == "target_frame" ||
          param.get_name() == "distance_threshold" ||
          param.get_name() == "optimize_coefficients" ||
          param.get_name() == "max_iterations" ||
          param.get_name() == "angle_threshold") {
        needs_param_reload = true;
      }
    }

    if (needs_param_reload) {
      loadParameters();
      RCLCPP_INFO(this->get_logger(), "Parameters updated - Filters: Aperture=%s, Box=%s, Ground=%s",
                  enable_aperture_filter_ ? "enabled" : "disabled",
                  enable_box_filter_ ? "enabled" : "disabled",
                  enable_ground_filter_ ? "enabled" : "disabled");
    }

    return result;
  }

  // Common members
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  std::string input_topic_;
  std::string output_topic_;
  std::string target_frame_;
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Aperture filter members
  bool enable_aperture_filter_;
  double min_azimuth_angle_, max_azimuth_angle_;
  double min_elevation_angle_, max_elevation_angle_;
  double min_azimuth_rad_, max_azimuth_rad_;
  double min_elevation_rad_, max_elevation_rad_;

  // Box filter members
  bool enable_box_filter_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr marker_timer_;
  std::vector<double> min_point_;
  std::vector<double> max_point_;
  bool box_filter_negative_;

  // Ground filter members
  bool enable_ground_filter_;
  double distance_threshold_;
  bool optimize_coefficients_;
  int max_iterations_;
  double angle_threshold_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CombinedLidarFilterNode>());
  rclcpp::shutdown();
  return 0;
}