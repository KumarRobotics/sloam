#pragma once

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <definitions.h>
#include <iostream>
#include <pcl/common/distances.h>
#include <random>
#include <utils.h>

#include <glog/logging.h>

using namespace boost;

class Instance {
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief The Params struct
   * beam_cluster_threshold: range difference to cluster consecutive point
   *    pixels
   * max_dist_to_centroid: remove points in vertex that are far from the
   *    centroid (median). Critical for good performance
   * min_vertex_size: minimum points in a vertex for it to be valid
   * min_landmark_size: minimum number of vertices for a landmark to be valid
   * min_landmark_height: minimum height (m) for a landmark to be valid
   */
  struct Params {
    // Parameters for vertex construction
    float beam_cluster_threshold;
    float max_dist_to_centroid;
    int min_vertex_size;

    // Parameters to determine valid landmark
    int min_landmark_size;
    float min_landmark_height;
  };

  explicit Instance();

  /**
   * @brief params: Getter function
   * @return params_
   */
  const Params &params() const { return params_; }
  /**
   * @brief set_params: Setter function
   * @param params
   */
  void set_params(const Params &params) { params_ = params; }
  void reset_tree_id() { tree_id_ = 0; }

  /**
   * @brief computeGraph: Wrapper function that creates and solves trellis graph
   * @param rangeImg: range image
   * @param mask: mask of point to consider
   * @param cloud: cloud of points
   * @param graph: graph
   * @param landmarks: landmarks to return
   */
  void computeGraph(const CloudT::Ptr cloud, const CloudT::Ptr tree_cloud,
                    std::vector<std::vector<TreeVertex>> &landmarks);

  bool computeVertexProperties(CloudT::Ptr &pc, Slash& filteredPoints, PointT& median_point, Scalar& radius);
  TreeVertex computeTreeVertex(CloudT::Ptr beam, int label);
  void findClusters(const CloudT::Ptr pc,
      pcl::PointCloud<pcl::Label>& euclidean_labels, std::vector<pcl::PointIndices>& label_indices);

  void findTrees(const CloudT::Ptr pc,
    pcl::PointCloud<pcl::Label>& euclidean_labels,
    std::vector<pcl::PointIndices>& label_indices, std::vector<std::vector<TreeVertex>>& landmarks);
  // void findClusters(const CloudT::Ptr pc,
  //     pcl::PointCloud<pcl::Label>& euclidean_labels, std::vector<pcl::PointIndices>& label_indices);
  // void regionGrowing(CloudT::Ptr cloud_filtered, pcl::PointCloud<pcl::Normal>::Ptr cloud_normals);
  // void passThroughFilter(const CloudT::Ptr pc, pcl::IndicesPtr& indices);
  // CloudT::Ptr filterPointCloud(const CloudT::Ptr pc);
  // pcl::PointCloud<pcl::Normal>::Ptr findPointCloudNormals(const CloudT::Ptr pc);

private:
  // Identifies arches and creates graph nodes based on centroids
  /**
   * @brief getVerticesFromBeam_: Cluster a beam according to range jumps
   * @param vertices: List of vertices to populate
   * @param rowIdx: the row in the range image
   * @param maxColIdx: the maximum number of columns in the range image
   * @param depthBeam: the depth of each point in beam
   * @param maskBeam: the mask that specifies which points to consider
   * @param cloud: the corresponding point cloud
   * @param graph: the trellis graph
   */
  void getVerticesFromBeam_(std::vector<vertex_t> &vertices,
                            const size_t rowIdx, const size_t maxColIdx,
                            const float *depthBeam,
                            const unsigned char *maskBeam,
                            const CloudT::Ptr cloud, graph_t &graph);

  /**
   * @brief processVertex_: Filter points, compute the mean, check if valid
   * @param vertexIndices: Indices in beam for vertex
   * @param vertexPoints: Vertex 3D points to fill
   * @param cloud: Point cloud of 3D points
   * @param rowIdx: the row in the range image
   * @param meanPoint: the point representing the mean of points (after filter)
   * @return
   */
  bool processVertex_(VectorType &vertexPoints, PointT &meanPoint,
                      std::vector<int> &vertexIndices, const CloudT::Ptr &cloud,
                      const size_t rowIdx);

  /**
   * @brief computeMedianAndFilterPoints_: Remove outlier points in vertex based on the median. Also returns the median.
   * @param vertexPoints: vector of points in vertex
   */
  void computeMedianAndFilterPoints_(VectorType &vertexPoints, PointT& median_point);

  /**
   * @brief computeFocusPointAndRadius_: Compute the focus (mean of furthest 2
   * points) and radius as distance between those 2 points
   * @param g: graph
   * @param vtx: vertex
   */
  void computeFocusPointAndRadius_(graph_t &g, vertex_t vtx);

  /**
   * @brief createVertex_: Create vertex in the graph
   * @param g: graph
   * @param vertexPoints: points in the vertex
   * @param meanPoint: mean of the vertex points
   * @param rowIdx: row in range image (beam)
   * @return
   */
  vertex_t createVertex_(graph_t &g, VectorType &vertexPoints, PointT meanPoint,
                         const size_t rowIdx);

  /**
   * @brief computeCandidateEdges: Compute the edges in the graph
   * @param graph: graph
   */
  void computeCandidateEdges(graph_t &graph);

  /**
   * @brief calculateEdgeCost_: Compute the cost on each edge
   * @param graph: graph
   * @param queryPoint: one vertex
   * @param nPoint: other vertex
   * @return
   */
  float calculateEdgeCost_(graph_t &graph, TreeVertex &queryPoint,
                           TreeVertex &nPoint);

  /**
   * @brief solveGreedy_: Find the shortest paths
   * @param graph: graph
   * @param vertices: vertices (sorted by beam)
   * @param landmarks: vector of landmarks (vector of TreeVertex)
   */
  void solveGreedy_(graph_t &graph,
                    std::vector<std::vector<vertex_t>> &vertices,
                    std::vector<std::vector<TreeVertex>> &landmarks);

  Params params_;
  int tree_id_;
};
