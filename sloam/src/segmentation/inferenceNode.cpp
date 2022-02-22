#include <fstream>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <inference.h>
#include <definitions.h>
#include <trellis.h>
#include <serialization.h>

using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;

class SegNode
{
public:
  explicit SegNode(const ros::NodeHandle &nh);

private:
  void SegCb_(const sensor_msgs::PointCloud2ConstPtr &cloudMsg);
  Cloud::Ptr trellisCloud(const std::vector<std::vector<TreeVertex>> &landmarks);
  boost::shared_ptr<seg::Segmentation> segmentator_ = nullptr;

  ros::NodeHandle nh_;
  ros::Subscriber cloudSub_;
  ros::Publisher treePub_;
  ros::Publisher trellisPub_;
  ros::Publisher groundPub_;
  ros::Time prevStamp;
  Instance graphDetector_;
};

SegNode::SegNode(const ros::NodeHandle &nh) : nh_(nh)
{

  std::string cloud_topic;
  nh_.param<std::string>("cloud_topic", cloud_topic, "/os_cloud_node/cloud_undistort");
  cloudSub_ = nh_.subscribe(cloud_topic, 10, &SegNode::SegCb_, this);

  treePub_ = nh_.advertise<CloudT>("segmentation/tree", 1);
  groundPub_ = nh_.advertise<CloudT>("segmentation/ground", 1);
  trellisPub_ = nh_.advertise<CloudT>("segmentation/trellis", 1);

  std::string modelFilepath;
  nh_.getParam("seg_model_path", modelFilepath);
  std::cout << "model: " << modelFilepath << std::endl;
  float fov = nh_.param("seg_lidar_fov", 22.5);
  int lidar_w = nh_.param("seg_lidar_w", 2048);
  int lidar_h = nh_.param("seg_lidar_h", 64);
  bool do_destagger = nh_.param("do_destagger", true);

  auto temp_seg = boost::make_shared<seg::Segmentation>(modelFilepath, fov, -fov, lidar_w, lidar_h, 1, do_destagger);
  segmentator_ = std::move(temp_seg);
  prevStamp = ros::Time::now();

  float beam_cluster_threshold = nh_.param("beam_cluster_threshold", 0.1);
  int min_vertex_size = nh_.param("min_vertex_size", 2);
  float max_dist_to_centroid = nh_.param("max_dist_to_centroid", 0.2);
  int min_landmark_size = nh_.param("min_landmark_size", 4);
  float min_landmark_height = nh_.param("min_landmark_height", 1);

  Instance::Params params;
  params.beam_cluster_threshold = beam_cluster_threshold;
  params.min_vertex_size = min_vertex_size;
  params.max_dist_to_centroid = max_dist_to_centroid;
  params.min_landmark_size = min_landmark_size;
  params.min_landmark_height = min_landmark_height;

  graphDetector_ = Instance();
  graphDetector_.set_params(params);
  graphDetector_.reset_tree_id();
}

Cloud::Ptr SegNode::trellisCloud(const std::vector<std::vector<TreeVertex>> &landmarks)
{
  CloudT::Ptr vtxCloud = CloudT::Ptr(new CloudT);
  std::vector<float> color_values((int)landmarks.size());
  std::iota(std::begin(color_values), std::end(color_values), 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(color_values.begin(), color_values.end(), gen);
  int color_id = 0;

  for (auto landmark : landmarks)
  {
    for (auto vtx : landmark)
    {
      for (auto point : vtx.points)
      {
        point.intensity = color_values[color_id];
        vtxCloud->points.push_back(point);
      }
    }
    color_id++;
  }
  vtxCloud->height = 1;
  vtxCloud->width = vtxCloud->points.size();
  return vtxCloud;
}

void SegNode::SegCb_(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{

  if ((prevStamp - cloudMsg->header.stamp).toSec() < 1.0)
    return;

  CloudT::Ptr cloud(new CloudT);
  pcl::fromROSMsg(*cloudMsg, *cloud);
  pcl_conversions::toPCL(ros::Time::now(), cloud->header.stamp);

  // RUN SEGMENTATION
  cv::Mat rMask = cv::Mat::zeros(cloudMsg->height, cloudMsg->width, CV_8U);
  segmentator_->run(cloud, rMask);
  // cv::imwrite("mask.jpg", rMask);

  CloudT::Ptr groundCloud(new CloudT());
  segmentator_->maskCloud(cloud, rMask, groundCloud, 1);

  CloudT::Ptr treeCloud(new CloudT);
  segmentator_->maskCloud(cloud, rMask, treeCloud, 255, true);

  std::vector<std::vector<TreeVertex>> landmarks;
  graphDetector_.computeGraph(cloud, treeCloud, landmarks);

  // std::cout << "SAVING DATA" << std::endl;
  // // write class instance to archive
  // std::ofstream ofs("landmarks");
  // boost::archive::text_oarchive oa(ofs);
  // oa << landmarks;
  // // write ground cloud to archive
  // pcl::io::savePCDFileASCII("ground.pcd", *groundCloud);
  // pcl::io::savePCDFileASCII("tree.pcd", *treeCloud);

  auto trellis = trellisCloud(landmarks);

  trellis->header = cloud->header;
  treeCloud->header = cloud->header;
  groundCloud->header = cloud->header;

  trellisPub_.publish(trellis);
  treePub_.publish(treeCloud);
  groundPub_.publish(groundCloud);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "sloam");
  ros::NodeHandle n("sloam");
  SegNode segnode(n);
  ros::spin();

  return 0;
}
