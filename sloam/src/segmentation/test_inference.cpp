#include "inference.h"
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>

using namespace seg;
using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;

void findClusters(const CloudT::Ptr pc){
    pcl::PointCloud<pcl::Label> euclidean_labels;
    std::vector<pcl::PointIndices> label_indices;

    pcl::EuclideanClusterComparator<PointT, pcl::Label>::Ptr
        euclidean_cluster_comparator(new pcl::EuclideanClusterComparator<PointT, pcl::Label>());

    std::cout << pc->points.size() << std::endl;
    std::cout << pc->width << std::endl;
    std::cout << pc->height << std::endl;

    euclidean_cluster_comparator->setInputCloud(pc);
    euclidean_cluster_comparator->setDistanceThreshold(0.5, false);

    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label>
        euclidean_segmentation(euclidean_cluster_comparator);
    euclidean_segmentation.setInputCloud(pc);
    euclidean_segmentation.segment(euclidean_labels, label_indices);

    pcl::PCDWriter writer;
    std::cout << "NUM CLUSTERS: " << label_indices.size() << std::endl;
    for (size_t i = 0; i < label_indices.size (); i++){
      if (label_indices.at(i).indices.size () > 100){
        std::cout << "LABEL INDICES SIZE: " << label_indices.at(i).indices.size() << std::endl;
     
        CloudT cluster;
        pcl::copyPointCloud(*pc, label_indices.at(i).indices, cluster);
        // clusters.push_back(cluster);
        std::stringstream ss;
        ss << "/opt/bags/inf/pcds/realcc/" << "cloud_cluster_" << i << ".pcd";
        writer.write<PointT> (ss.str (), cluster, false);
      }
    }

}

int main(int argc, char* argv[])
{
    // SIM
    // std::string modelFilepath{"/opt/bags/inf/models/sim_erfnet.onnx"};
    // std::string cloudPath{"/opt/bags/inf/pcds/sim_point_cloud_120.pcd"};
    // auto segmenter = Segmentation(modelFilepath, 22.5, -22.5, 2048, 64, 1);
    
    // // // // READ PCD FROM FILE, PROJECT
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::io::loadPCDFile<pcl::PointXYZI> (cloudPath, *cloud) == -1;
    // cv::Mat mask = cv::Mat::zeros(64, 2048, CV_8U);
    // // segmenter.run(cloud, mask);
    // segmenter.speedTest(cloud, 10);

    // Cloud::Ptr trees(new Cloud);
    // segmenter.maskCloud(cloud, mask, trees, 255, true);
    // findClusters(trees);
    // pcl::io::savePCDFileASCII ("/opt/bags/inf/pcds/trees.pcd", *trees);
    // // std::cout << "grouind" << std::endl;
    // Cloud::Ptr ground(new Cloud);
    // segmenter.maskCloud(cloud, mask, ground, 1);
    // pcl::io::savePCDFileASCII ("/opt/bags/inf/pcds/ground.pcd", *ground);

    ////////////////////////////////
    std::string realModelFilepath{"/opt/bags/inf/models/real_erfnet.onnx"};
    std::string realCloudPath{"/opt/bags/inf/pcds/800m.pcd"};
    // // std::string realCloudPath{"/opt/bags/inf/pcds/s3_p1_f2_800.pcd"};
    auto realSegmenter = Segmentation(realModelFilepath, 22.5, -22.5, 2048, 64, 1, true);

    // // // // READ PCD FROM FILE
    pcl::PointCloud<pcl::PointXYZI>::Ptr realCloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI> (realCloudPath, *realCloud) == -1;

    // for(auto i = 0; i < realCloud->points.size(); i++){
    //     realCloud->points[i].intensity == 1;
    //     if(realCloud->points[i].z == 0){
    //         PointT p;
    //         p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
    //         realCloud->points[i] = p;
    //     }
    // }

    cv::Mat realMask = cv::Mat::zeros(64, 2048, CV_8U);
    realSegmenter.run(realCloud, realMask);

    Cloud::Ptr realTrees(new Cloud);
    realSegmenter.maskCloud(realCloud, realMask, realTrees, 255, true);
    pcl::io::savePCDFileASCII ("/opt/bags/inf/pcds/erf_trees.pcd", *realTrees);
    findClusters(realTrees);

    Cloud::Ptr realGround(new Cloud);
    realSegmenter.maskCloud(realCloud, realMask, realGround, 1, false);
    pcl::io::savePCDFileASCII ("/opt/bags/inf/pcds/erf_ground.pcd", *realGround);
}

// auto seg = Segmentation(modelFilepath, 22.5, -22.5, 2048, 64, 1);

// REAL WORLD ARKANSAS
// std::string modelFilepath{"/opt/bags/inf/models/realbdark.onnx"};
// std::string cloudPath{"/opt/bags/inf/pcds/s4_p2_f1_800.pcd"};
// auto seg = Segmentation(modelFilepath, 16.6, -16.6, 2048, 64, 1);

// REAL WORLD WHARTON
// std::string modelFilepath{"/opt/bags/inf/models/realbdark.onnx"};
// std::string cloudPath{"/opt/bags/inf/pcds/sloam_sim_cloud.pcd"};