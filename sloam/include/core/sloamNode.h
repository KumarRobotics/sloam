#pragma once

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// pcl
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <tf2/buffer_core.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <sloam_msgs/ROSObservation.h>
#include <definitions.h>
#include <sloam.h>
#include <inference.h>
#include <trellis.h>
#include <utils.h>
#include <vizTools.h>
#include <mapManager.h>

namespace sloam
{
  class SLOAMNode : public sloam
  {
  public:
    explicit SLOAMNode(const ros::NodeHandle &nh);
    SLOAMNode(const SLOAMNode &) = delete;
    SLOAMNode operator=(const SLOAMNode &) = delete;
    using Ptr = boost::shared_ptr<SLOAMNode>;
    using ConstPtr = boost::shared_ptr<const SLOAMNode>;
    bool run(const SE3 initialGuess, const SE3 prevKeyPose, CloudT::Ptr cloud, ros::Time stamp, SE3 &outPose);

  private:
    void initParams_();
    Cloud::Ptr trellisCloud(const std::vector<std::vector<TreeVertex>> &landmarks);
    void publishMap_(const ros::Time stamp);
    /*
    * --------------- Visualization ------------------
    */
    ros::NodeHandle nh_;

    tf::TransformBroadcaster worldTfBr_;
    // SLOAM Output Publishers
    ros::Publisher pubMapPose_;
    ros::Publisher pubObs_;

    // DEBUG TOPICS
    // ros::Publisher pubMapTreeFeatures_;
    ros::Publisher pubTrajectory_;
    ros::Publisher pubMapGroundFeatures_;
    ros::Publisher pubObsTreeFeatures_;
    ros::Publisher pubObsGroundFeatures_;
    ros::Publisher pubMapTreeModel_;
    ros::Publisher pubSubmapTreeModel_;
    ros::Publisher pubObsTreeModel_;
    ros::Publisher pubMapGroundModel_;
    ros::Publisher pubObsGroundModel_;

    // Transform
    tf2_ros::Buffer tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    bool static_transforms_initialized_ = false;

    std::string world_frame_id_, robot_frame_id_, map_frame_id_;

    // Submodule objects
    boost::shared_ptr<seg::Segmentation> segmentator_ = nullptr;
    Instance graphDetector_;
    MapManager semanticMap_;
    FeatureModelParams fmParams_;

    std::vector<SE3> trajectory;
    bool firstScan_;
    bool debugMode_;
  };
} // namespace sloam
