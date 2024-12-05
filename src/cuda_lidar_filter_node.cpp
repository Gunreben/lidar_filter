#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <visualization_msgs/msg/marker.hpp>
#include <cuda_runtime.h>
#include <cmath>

// CUDA forward declarations
void cudaGroundSegmentation(float* d_points, int num_points, float* d_coefficients, 
                          int* d_indices, float distance_threshold, int max_iterations);

class CudaLidarFilterNode : public rclcpp::Node
{
public:
  CudaLidarFilterNode()
  : Node("cuda_lidar_filter_node"), cuda_enabled_(false)
  {
    // Check CUDA availability
    initializeCuda();

    // Common parameters
    this->declare_parameter<std::string>("input_topic", "/points");
    this->declare_parameter<std::string>("output_topic", "/filtered_points");
    this->declare_parameter<bool>("force_cpu_processing", false);
    
    // Aperture filter parameters
    this->declare_parameter<bool>("enable_aperture_filter", true);
    this->declare_parameter<double>("min_azimuth_angle", -60.0);
    this->declare_parameter<double>("max_azimuth_angle", 60.0);
    this->declare_parameter<double>("min_elevation_angle", -30.0);
    this->declare_parameter<double>("max_elevation_angle", 30.0);

    // Box filter parameters
    this->declare_parameter<bool>("enable_box_filter", true);
    this->declare_parameter<std::vector<float>>("min_point", std::vector<float>{-1.0, -1.0, -1.0});
    this->declare_parameter<std::vector<float>>("max_point", std::vector<float>{1.0, 1.0, 1.0});
    this->declare_parameter<bool>("box_filter_negative", true);

    // Ground filter parameters
    this->declare_parameter<bool>("enable_ground_filter", true);
    this->declare_parameter<double>("distance_threshold", 0.1);
    this->declare_parameter<bool>("optimize_coefficients", true);
    this->declare_parameter<int>("max_iterations", 100);
    this->declare_parameter<double>("angle_threshold", 10.0);

    loadParameters();

    // Initialize publisher
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
    
    if (enable_box_filter_) {
      marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("filter_box_marker", 10);
      marker_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&CudaLidarFilterNode::publishBoxMarker, this));
    }

    // Main subscriber
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&CudaLidarFilterNode::pointCloudCallback, this, std::placeholders::_1));

    // Parameter callback
    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&CudaLidarFilterNode::parameterCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), 
                "Node initialized with CUDA %s. Filters: Aperture=%s, Box=%s, Ground=%s",
                cuda_enabled_ ? "enabled" : "disabled",
                enable_aperture_filter_ ? "enabled" : "disabled",
                enable_box_filter_ ? "enabled" : "disabled",
                enable_ground_filter_ ? "enabled" : "disabled");
  }

  ~CudaLidarFilterNode()
  {
    if (cuda_enabled_) {
      cudaStreamDestroy(cuda_stream_);
      cudaFree(d_points_);
      cudaFree(d_indices_);
      cudaFree(d_coefficients_);
    }
  }

private:
    void loadParameters() {
      // Load common parameters
      input_topic_ = this->get_parameter("input_topic").as_string();
      output_topic_ = this->get_parameter("output_topic").as_string();
      
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

  void initializeCuda()
  {
      int device_count = 0;
      cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
      
      if (cuda_status != cudaSuccess || device_count == 0) {
          RCLCPP_WARN(this->get_logger(), "No CUDA devices available. Using CPU processing.");
          return;
      }

      cuda_status = cudaStreamCreate(&cuda_stream_);
      if (cuda_status != cudaSuccess) {
          RCLCPP_WARN(this->get_logger(), "Failed to create CUDA stream. Using CPU processing.");
          return;
      }

      // Initialize with a reasonable size, will be resized as needed
      size_t initial_size = 100000; // Adjust based on expected point cloud size
      cuda_status = cudaMallocManaged(&d_points_, sizeof(float) * 4 * initial_size);
      if (cuda_status != cudaSuccess) {
          RCLCPP_WARN(this->get_logger(), "Failed to allocate points memory. Using CPU processing.");
          cudaStreamDestroy(cuda_stream_);
          return;
      }

      cuda_status = cudaMallocManaged(&d_indices_, sizeof(int) * initial_size);
      if (cuda_status != cudaSuccess) {
          RCLCPP_WARN(this->get_logger(), "Failed to allocate indices memory. Using CPU processing.");
          cudaFree(d_points_);
          cudaStreamDestroy(cuda_stream_);
          return;
      }

      cuda_status = cudaMallocManaged(&d_coefficients_, sizeof(float) * 4);
      if (cuda_status != cudaSuccess) {
          RCLCPP_WARN(this->get_logger(), "Failed to allocate coefficients memory. Using CPU processing.");
          cudaFree(d_points_);
          cudaFree(d_indices_);
          cudaStreamDestroy(cuda_stream_);
          return;
      }

      cuda_enabled_ = true;
      allocated_points_ = initial_size;
      RCLCPP_INFO(this->get_logger(), "CUDA initialization successful");
  }

  void resizeGpuBuffers(size_t needed_size)
  {
    if (needed_size <= allocated_points_) return;

    // Free existing buffers
    cudaFree(d_points_);
    cudaFree(d_indices_);

    // Allocate new buffers with more space
    size_t new_size = needed_size * 1.5; // Add some extra space for future growth
    cudaError_t cuda_status = cudaMallocManaged(&d_points_, sizeof(float) * 4 * new_size);
    if (cuda_status != cudaSuccess) {
        RCLCPP_ERROR(this->get_logger(), "Failed to resize points buffer!");
        cuda_enabled_ = false;
        return;
    }

    cuda_status = cudaMallocManaged(&d_indices_, sizeof(int) * new_size);
    if (cuda_status != cudaSuccess) {
        RCLCPP_ERROR(this->get_logger(), "Failed to resize indices buffer!");
        cudaFree(d_points_); // Clean up the first allocation
        cuda_enabled_ = false;
        return;
    }

    allocated_points_ = new_size;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // Convert ROS2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Apply filters sequentially if enabled
    if (enable_aperture_filter_) {
      cloud = applyApertureFilter(cloud);
    }

    if (enable_box_filter_) {
      cloud = applyBoxFilter(cloud);
    }

    if (enable_ground_filter_) {
      cloud = cuda_enabled_ ? applyGroundFilterCuda(cloud) : applyGroundFilterCpu(cloud);
    }

    // Publish final filtered cloud
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = cloud_msg->header;
    pub_->publish(output_msg);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr applyGroundFilterCuda(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    size_t num_points = input_cloud->points.size();
    
    // Resize GPU buffers if necessary
    resizeGpuBuffers(num_points);

    // Copy point cloud data to GPU
    float* input_data = reinterpret_cast<float*>(input_cloud->points.data());
    cudaMemcpyAsync(d_points_, input_data, sizeof(float) * 4 * num_points,
                    cudaMemcpyHostToDevice, cuda_stream_);

    // Run CUDA ground segmentation
    cudaGroundSegmentation(d_points_, num_points, d_coefficients_, d_indices_,
                          distance_threshold_, max_iterations_);

    // Create output cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    filtered_cloud->width = input_cloud->width;
    filtered_cloud->height = input_cloud->height;
    filtered_cloud->points.resize(input_cloud->points.size());

    // Copy non-ground points to output cloud
    int output_idx = 0;
    for (size_t i = 0; i < num_points; ++i) {
      if (d_indices_[i] == 0) { // Non-ground points
        filtered_cloud->points[output_idx].x = input_cloud->points[i].x;
        filtered_cloud->points[output_idx].y = input_cloud->points[i].y;
        filtered_cloud->points[output_idx].z = input_cloud->points[i].z;
        output_idx++;
      }
    }
    filtered_cloud->points.resize(output_idx);
    filtered_cloud->width = output_idx;
    filtered_cloud->height = 1;

    return filtered_cloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr applyGroundFilterCpu(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
  {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    seg.setOptimizeCoefficients(optimize_coefficients_);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold_);
    seg.setMaxIterations(max_iterations_);
    seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
    seg.setEpsAngle(angle_threshold_ * M_PI / 180.0);

    seg.setInputCloud(input_cloud);
    seg.segment(*inliers, *coefficients);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*filtered_cloud);

    return filtered_cloud;
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
    marker.header.frame_id = "os0_128_10hz_512_rev7";
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
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

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

  // CUDA-specific members
  bool cuda_enabled_;
  cudaStream_t cuda_stream_;
  float* d_points_;
  int* d_indices_;
  float* d_coefficients_;
  size_t allocated_points_;
};


int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CudaLidarFilterNode>());
  rclcpp::shutdown();
  return 0;
}