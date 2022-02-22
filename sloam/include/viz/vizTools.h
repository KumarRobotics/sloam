#pragma once

// ROS
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <definitions.h>
#include <cylinder.h>
#include <plane.h>

namespace sloam
{

  // functions adapted from ros tf2/tf2_eigen since
  // the original implementation needs double...
  geometry_msgs::Quaternion toMsg_(const Quaternionf &in);
  geometry_msgs::Point toMsg_(const Vector3 &in);
  geometry_msgs::Quaternion toRosQuat_(const SO3 &R);
  geometry_msgs::Pose toRosPose_(const SE3 &T);
  nav_msgs::Odometry toRosOdom_(const SE3& pose, const std::string slam_ref_frame, const ros::Time stamp);
  geometry_msgs::PoseStamped makeROSPose(const SE3 &tf, std::string frame_id);
  SE3 toSE3(geometry_msgs::PoseStamped pose);
  visualization_msgs::MarkerArray vizTrajectory(const std::vector<SE3>& poses);
  visualization_msgs::MarkerArray vizGroundModel(const std::vector<Plane> &gplanes, const std::string &frame_id, int idx);
  void vizTreeModels(const std::vector<Cylinder> &scanTm,
                     visualization_msgs::MarkerArray &tMarkerArray,
                     size_t &cylinderId);
  visualization_msgs::Marker vizGroundModel(const Plane &gplane, const std::string &frame_id, const int idx);
  void landmarksToCloud(const std::vector<std::vector<TreeVertex>> &landmarks,
                        CloudT::Ptr &cloud);
  cv::Mat DecodeImage(const sensor_msgs::ImageConstPtr &image_msg);
} // namespace sloam
