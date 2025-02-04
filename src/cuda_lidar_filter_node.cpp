#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>

// --- Include CUDA-based segmentation ---
#include "cuda_runtime.h"
#include "lidar_filter/cudaSegmentation.h"

class CudaLidarFilterNode : public rclcpp::Node
{
public:
  CudaLidarFilterNode()
  : Node("cuda_lidar_filter_node")
  {
    // Common parameters
    this->declare_parameter<std::string>("input_topic", "/points");
    this->declare_parameter<std::string>("output_topic", "/filtered_points");
    
    // Aperture filter parameters
    this->declare_parameter<bool>("enable_aperture_filter", true);
    this->declare_parameter<double>("min_azimuth_angle", -60.0);
    this->declare_parameter<double>("max_azimuth_angle",  60.0);
    this->declare_parameter<double>("min_elevation_angle", -30.0);
    this->declare_parameter<double>("max_elevation_angle", 30.0);

    // Box filter parameters
    this->declare_parameter<bool>("enable_box_filter", true);
    this->declare_parameter<std::vector<float>>("min_point", std::vector<float>{-0.5f, -1.5f, -1.2f});
    this->declare_parameter<std::vector<float>>("max_point", std::vector<float>{ 3.0f,  1.5f,  1.0f});
    this->declare_parameter<bool>("box_filter_negative", true);

    // Ground filter parameters (CUDA-based)
    this->declare_parameter<bool>("enable_ground_filter", true);
    this->declare_parameter<double>("distance_threshold", 0.01);
    this->declare_parameter<bool>("optimize_coefficients", true);
    this->declare_parameter<int>("max_iterations", 50);
    this->declare_parameter<double>("angle_threshold", 15.0); // Not directly used by the CUDA sample, but kept for parity

    // Load parameters
    loadParameters();
    
    // Set QOS to BEST_EFFORT
    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.best_effort();

    // Initialize publishers
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos);
    
    if (enable_box_filter_) {
      marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("filter_box_marker", qos);
      marker_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&CudaLidarFilterNode::publishBoxMarker, this));
    }

    // Main subscriber
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      std::bind(&CudaLidarFilterNode::pointCloudCallback, this, std::placeholders::_1));

    // Parameter callback
    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&CudaLidarFilterNode::parameterCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "CudaLidarFilterNode initialized with filters: Aperture=%s, Box=%s, Ground=%s",
                enable_aperture_filter_ ? "enabled" : "disabled",
                enable_box_filter_ ? "enabled" : "disabled",
                enable_ground_filter_ ? "enabled" : "disabled");
  }

private:
  //-----------------------------------------------------------------------------------
  // Load parameters from the parameter server
  void loadParameters() {
    // Common
    input_topic_ = this->get_parameter("input_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    
    // Aperture filter
    enable_aperture_filter_ = this->get_parameter("enable_aperture_filter").as_bool();
    if (enable_aperture_filter_) {
      min_azimuth_angle_ = this->get_parameter("min_azimuth_angle").as_double();
      max_azimuth_angle_ = this->get_parameter("max_azimuth_angle").as_double();
      min_elevation_angle_ = this->get_parameter("min_elevation_angle").as_double();
      max_elevation_angle_ = this->get_parameter("max_elevation_angle").as_double();

      // Convert angles to radians
      min_azimuth_rad_   = min_azimuth_angle_   * M_PI / 180.0;
      max_azimuth_rad_   = max_azimuth_angle_   * M_PI / 180.0;
      min_elevation_rad_ = min_elevation_angle_ * M_PI / 180.0;
      max_elevation_rad_ = max_elevation_angle_ * M_PI / 180.0;
    }

    // Box filter
    enable_box_filter_ = this->get_parameter("enable_box_filter").as_bool();
    if (enable_box_filter_) {
      min_point_ = this->get_parameter("min_point").as_double_array();
      max_point_ = this->get_parameter("max_point").as_double_array();
      box_filter_negative_ = this->get_parameter("box_filter_negative").as_bool();
    }

    // Ground filter (CUDA-based)
    enable_ground_filter_ = this->get_parameter("enable_ground_filter").as_bool();
    if (enable_ground_filter_) {
      distance_threshold_      = this->get_parameter("distance_threshold").as_double();
      optimize_coefficients_   = this->get_parameter("optimize_coefficients").as_bool();
      max_iterations_          = this->get_parameter("max_iterations").as_int();
      angle_threshold_         = this->get_parameter("angle_threshold").as_double();
    }
  }

  //-----------------------------------------------------------------------------------
  // Subscription callback: apply filters in series
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // Convert ROS2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Aperture Filter
    if (enable_aperture_filter_) {
      cloud = applyApertureFilter(cloud);
    }

    // Box Filter
    if (enable_box_filter_) {
      cloud = applyBoxFilter(cloud);
    }

    // Ground Filter (CUDA-based)
    if (enable_ground_filter_) {
      cloud = applyCudaGroundFilter(cloud);
    }

    // Publish final filtered cloud
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = cloud_msg->header;
    pub_->publish(output_msg);
  }

  //-----------------------------------------------------------------------------------
  // Aperture filter
  pcl::PointCloud<pcl::PointXYZ>::Ptr applyApertureFilter(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    filtered_cloud->reserve(input_cloud->points.size());

    for (const auto& point : input_cloud->points) {
      double azimuth = std::atan2(point.y, point.x);
      double range   = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      if (range == 0) continue;
      
      double elevation = std::asin(point.z / range);

      if (azimuth >= min_azimuth_rad_ && azimuth <= max_azimuth_rad_ &&
          elevation >= min_elevation_rad_ && elevation <= max_elevation_rad_)
      {
        filtered_cloud->points.push_back(point);
      }
    }

    filtered_cloud->width    = filtered_cloud->points.size();
    filtered_cloud->height   = 1;
    filtered_cloud->is_dense = true;
    return filtered_cloud;
  }

  //-----------------------------------------------------------------------------------
  // Box filter
  pcl::PointCloud<pcl::PointXYZ>::Ptr applyBoxFilter(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CropBox<pcl::PointXYZ> box_filter;
    
    box_filter.setMin(Eigen::Vector4f(
      static_cast<float>(min_point_[0]),
      static_cast<float>(min_point_[1]),
      static_cast<float>(min_point_[2]),
      1.0f
    ));
    box_filter.setMax(Eigen::Vector4f(
      static_cast<float>(max_point_[0]),
      static_cast<float>(max_point_[1]),
      static_cast<float>(max_point_[2]),
      1.0f
    ));
    box_filter.setNegative(box_filter_negative_);
    box_filter.setInputCloud(input_cloud);
    box_filter.filter(*filtered_cloud);

    return filtered_cloud;
  }

  //-----------------------------------------------------------------------------------
  // Ground filter using cuPCL's CUDA-based plane segmentation
  //
  // This function takes the input cloud, performs RANSAC plane segmentation on the GPU,
  // and removes the inliers (plane) from the cloud—thus removing ground.
  //
  pcl::PointCloud<pcl::PointXYZ>::Ptr applyCudaGroundFilter(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    // If cuPCL logic fails or no plane is found, we return the original cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>(*input_cloud));

    // Setup CUDA stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // Prepare device memory
    int nCount = input_cloud->width * input_cloud->height;
    if (nCount == 0) {
      RCLCPP_WARN(this->get_logger(), "Empty point cloud provided to CUDA ground filter.");
      cudaStreamDestroy(stream);
      return filtered_cloud;
    }

    // Our pcl::PointXYZ memory layout is x,y,z (plus padding). Each point is typically 4 floats in memory.
    float* input = nullptr;
    cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, input);
    cudaStreamSynchronize(stream);

    // Copy host->device
    std::memcpy(input, input_cloud->points.data(), sizeof(float) * 4 * nCount);

    // Index array to mark inliers
    int* index = nullptr;
    cudaMallocManaged(&index, sizeof(int) * nCount, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, index);
    cudaStreamSynchronize(stream);
    std::memset(index, 0, sizeof(int) * nCount);

    // Model coefficients array
    float* modelCoefficients = nullptr;
    int modelSize = 4; // For plane [a, b, c, d]
    cudaMallocManaged(&modelCoefficients, sizeof(float) * modelSize, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, modelCoefficients);
    cudaStreamSynchronize(stream);
    std::memset(modelCoefficients, 0, sizeof(float) * modelSize);

    // Setup cuPCL segmentation object
    cudaSegmentation cudaSeg(SACMODEL_PLANE, SAC_RANSAC, stream);
    segParam_t segParams;
    segParams.distanceThreshold     = distance_threshold_;
    segParams.maxIterations         = max_iterations_;
    segParams.probability           = 0.99; // Hard-coded in example
    segParams.optimizeCoefficients  = optimize_coefficients_;

    cudaSeg.set(segParams);

    // Run CUDA-based plane segmentation
    cudaSeg.segment(input, nCount, index, modelCoefficients);

    // 'index[i] == 1' means this point is part of the plane (ground)
    // Remove these points from the cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_removed(new pcl::PointCloud<pcl::PointXYZ>());
    ground_removed->reserve(nCount);

    for (int i = 0; i < nCount; ++i) {
      if (index[i] == 1) {  // index=0 means not in plane => keep
        // Rebuild point from the original data
        pcl::PointXYZ pt;
        pt.x = input[i*4 + 0];
        pt.y = input[i*4 + 1];
        pt.z = input[i*4 + 2];
        ground_removed->points.push_back(pt);
      }
    }

    ground_removed->width    = ground_removed->points.size();
    ground_removed->height   = 1;
    ground_removed->is_dense = true;

    // Cleanup
    cudaFree(input);
    cudaFree(index);
    cudaFree(modelCoefficients);
    cudaStreamDestroy(stream);

    // If no plane found, the plane inliers might be empty; or if plane found, we have effectively removed it
    return ground_removed;
  }

  //-----------------------------------------------------------------------------------
  // Visualization marker (box)
  void publishBoxMarker()
  {
    if (!enable_box_filter_) return;

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "os_sensor"; // os0_128_10hz_512_rev7 in simulation, os_sensor real
    marker.header.stamp = this->now();
    marker.ns = "filter_box";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = (min_point_[0] + max_point_[0]) / 2.0;
    marker.pose.position.y = (min_point_[1] + max_point_[1]) / 2.0;
    marker.pose.position.z = (min_point_[2] + max_point_[2]) / 2.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = std::fabs(max_point_[0] - min_point_[0]);
    marker.scale.y = std::fabs(max_point_[1] - min_point_[1]);
    marker.scale.z = std::fabs(max_point_[2] - min_point_[2]);

    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 0.5;
    marker.lifetime = rclcpp::Duration(0, 0);

    marker_pub_->publish(marker);
  }

  //-----------------------------------------------------------------------------------
  // Parameter callback
  rcl_interfaces::msg::SetParametersResult parameterCallback(
    const std::vector<rclcpp::Parameter>& parameters)
  {
    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;

    bool needs_param_reload = false;
    for (const auto& param : parameters) {
      // If any relevant parameters change, we need to reload
      if (param.get_name().find("min_") != std::string::npos ||
          param.get_name().find("max_") != std::string::npos ||
          param.get_name().find("enable_") != std::string::npos ||
          param.get_name() == "distance_threshold" ||
          param.get_name() == "optimize_coefficients" ||
          param.get_name() == "max_iterations" ||
          param.get_name() == "angle_threshold")
      {
        needs_param_reload = true;
      }
    }

    if (needs_param_reload) {
      loadParameters();
      RCLCPP_INFO(this->get_logger(), "Parameters updated - Aperture=%s, Box=%s, Ground=%s",
                  enable_aperture_filter_ ? "enabled" : "disabled",
                  enable_box_filter_ ? "enabled" : "disabled",
                  enable_ground_filter_ ? "enabled" : "disabled");
    }

    return result;
  }

  //-----------------------------------------------------------------------------------
  // Member variables

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  // Box visualization
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr marker_timer_;

  // Topics
  std::string input_topic_;
  std::string output_topic_;

  // Aperture filter
  bool enable_aperture_filter_;
  double min_azimuth_angle_, max_azimuth_angle_;
  double min_elevation_angle_, max_elevation_angle_;
  double min_azimuth_rad_, max_azimuth_rad_;
  double min_elevation_rad_, max_elevation_rad_;

  // Box filter
  bool enable_box_filter_;
  std::vector<double> min_point_;
  std::vector<double> max_point_;
  bool box_filter_negative_;

  // Ground filter (CUDA-based)
  bool enable_ground_filter_;
  double distance_threshold_;
  bool optimize_coefficients_;
  int max_iterations_;
  double angle_threshold_;
};

//-----------------------------------------------------------------------------------
// main
int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CudaLidarFilterNode>());
  rclcpp::shutdown();
  return 0;
}
