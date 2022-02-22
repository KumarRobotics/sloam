#include <trellis.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <chrono>

Instance::Instance() { tree_id_ = 0; }

void Instance::findClusters(const CloudT::Ptr pc,
      pcl::PointCloud<pcl::Label>& euclidean_labels, std::vector<pcl::PointIndices>& label_indices){
    if(pc->size() == 0) return;

    pcl::EuclideanClusterComparator<PointT, pcl::Label>::Ptr
        euclidean_cluster_comparator(new pcl::EuclideanClusterComparator<PointT, pcl::Label>());

    euclidean_cluster_comparator->setInputCloud(pc);
    euclidean_cluster_comparator->setDistanceThreshold(1.0, false);

    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label>
        euclidean_segmentation(euclidean_cluster_comparator);
    euclidean_segmentation.setInputCloud(pc);
    euclidean_segmentation.segment(euclidean_labels, label_indices);

    // pcl::PCDWriter writer;
    // for (size_t i = 0; i < label_indices.size (); i++){
    //   // std::cout << "LABEL INDICES SIZE: " << label_indices.at(i).indices.size() << std::endl;
    //   if (label_indices.at(i).indices.size () > 80){
    //     CloudT cluster;
    //     pcl::copyPointCloud(*pc, label_indices.at(i).indices, cluster);
    //     ROS_DEBUG_STREAM(cluster.width << " x " << cluster.height);
    //     // clusters.push_back(cluster);
    //     std::stringstream ss;
    //     ss << "/opt/bags/inf/pcds/sloam/" << "cloud_cluster_" << i << ".pcd";
    //     writer.write<PointT> (ss.str (), cluster, false);
    //   }
    // }
}

TreeVertex Instance::computeTreeVertex(CloudT::Ptr beam, int label){
  TreeVertex v;
  Slash filteredPoints;
  PointT median;
  Scalar radius;
  bool valid = computeVertexProperties(beam, filteredPoints, median, radius);

  v.treeId = label;
  v.prevVertexSize = 0;
  v.points = filteredPoints;
  v.coords = median;
  // v.isRoot = false;
  v.isValid = valid;
  v.beam = 0;
  v.radius = radius;
  return v;
}

bool Instance::computeVertexProperties(CloudT::Ptr &pc, Slash& filteredPoints, PointT& median_point, Scalar& radius) {
  // Compute median in each x,y,z
  int num_points = pc->points.size();
  int middle_point = (int)(num_points / 2.0);
  Scalar median_x = 0;
  Scalar median_y = 0;
  Scalar median_z = 0;

  std::sort(pc->points.begin(), pc->points.end(),
            [](const PointT &p1, const PointT &p2) { return p1.x < p2.x; });
            
  median_x = pc->points[middle_point].x;

  std::sort(pc->points.begin(), pc->points.end(),
            [](const PointT &p1, const PointT &p2) { return p1.y < p2.y; });
  median_y = pc->points[middle_point].y;

  std::sort(pc->points.begin(), pc->points.end(),
            [](const PointT &p1, const PointT &p2) { return p1.z < p2.z; });
  median_z = pc->points[middle_point].z;

  // PointT median_point;
  median_point.x = median_x;
  median_point.y = median_y;
  median_point.z = median_z;

  for (const auto &point : pc->points) {
    if (euclideanDistance(point, median_point) < params_.max_dist_to_centroid) {
      filteredPoints.push_back(point);
    }
  }

  if(filteredPoints.size() > 1){
    PointT pointA = filteredPoints[0];
    PointT pointB = filteredPoints[filteredPoints.size() - 1];
    radius = euclideanDistance(pointA, pointB);
    return true;
  }
  return false;
}

void Instance::findTrees(const CloudT::Ptr pc,
    pcl::PointCloud<pcl::Label>& euclidean_labels,
    std::vector<pcl::PointIndices>& label_indices, std::vector<std::vector<TreeVertex>>& landmarks){

    for (size_t i = 0; i < label_indices.size(); i++){
      if (label_indices.at(i).indices.size () > 80){
        std::vector<TreeVertex> tree;
        for (int row_idx = pc->height - 1; row_idx >= 0; --row_idx) {
          CloudT::Ptr beam(new CloudT);
          for (size_t col_idx = 0; col_idx < pc->width; ++col_idx) {
            if(euclidean_labels.points[row_idx * pc->width + col_idx].label == i){
              PointT p = pc->at(col_idx, row_idx);
              beam->points.push_back(p);
            }
          }
          if(beam->points.size() > 3){
            TreeVertex v = computeTreeVertex(beam, i);
            if(v.isValid) tree.push_back(v);
          }
        }
        if(tree.size() > 16){
          if(tree.size() > 56) {
            tree.resize(56);
          }
          landmarks.push_back(tree);
        }
      }
    }
}

void Instance::computeGraph(const CloudT::Ptr cloud, const CloudT::Ptr tree_cloud,
                            std::vector<std::vector<TreeVertex>> &landmarks) {
  pcl::PointCloud<pcl::Label> euclidean_labels;
  std::vector<pcl::PointIndices> label_indices;
  findClusters(tree_cloud, euclidean_labels, label_indices);
  findTrees(tree_cloud, euclidean_labels, label_indices, landmarks);
}