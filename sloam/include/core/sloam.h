#pragma once

#include <cylinder.h>
#include <plane.h>
#include <definitions.h>
#include <trellis.h>
#include <utils.h>

/*
 * --------------- Feature and Model Structures ------------------
 */

template <typename T>
struct ObjectMatch
{
  ObjectMatch(PointT ft, T obj, Scalar dist) : object(obj)
  {
    Vector3d tfeat;
    tfeat << ft.x, ft.y, ft.z;

    feature = tfeat;
    dist = dist;
  }
  Vector3 feature;
  T object;
  double dist;
};

/*
 * --------------- odom/map inputs and outputs ------------------
 */
struct SloamInput
{
  SloamInput()
  {
    groundCloud = CloudT::Ptr(new CloudT());
    // groundKDTree = KDTree::Ptr(new KDTree());
    // allCloud = CloudT::Ptr(new CloudT());
  };

  SE3 poseEstimate;
  Scalar distance;
  CloudT::Ptr groundCloud;
  std::vector<Cylinder> mapModels;
  std::vector<std::vector<TreeVertex>> landmarks;
};

struct SloamOutput
{
  SloamOutput(){};
  std::vector<int> matches;
  std::vector<Cylinder> tm;
  SE3 T_Map_Curr;
  SE3 T_Delta;
};

namespace sloam
{
  class sloam
  {
  public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit sloam();

    const FeatureModelParams &fmParams() const { return fmParams_; }
    void setFmParams(const FeatureModelParams &fmParams) { fmParams_ = fmParams; }
    bool RunSloam(SloamInput &in, SloamOutput &out);

    // Pose Optimization
    void OptimizeXYYaw(const SE3 &poseEstimate, const bool optimize, const std::vector<ObjectMatch<Cylinder>> &allTMatch, double*);
    void OptimizeZRollPitch(const SE3 &poseEstimate, const bool optimize, const std::vector<ObjectMatch<Plane>> &allGMatch, double*);
    bool TwoStepOptimizePose(const SE3& poseEstimate, const bool optimTrees, const bool optimGround,
                              const std::vector<ObjectMatch<Cylinder>> &allTMatch,
                              const std::vector<ObjectMatch<Plane>> &allGMatch,
                              SE3 &tf);
    bool OptimizePose(const SE3 &poseEstimate,
                      const std::vector<ObjectMatch<Cylinder>> &allTMatch,
                      const std::vector<ObjectMatch<Plane>> &allGMatch,
                      SE3 &tf);
    // Model estimation
    void projectModels(const SE3 &tf, std::vector<Cylinder> &landmarks, std::vector<Plane> &planes);
    void computeModels(SloamInput &in, std::vector<Cylinder> &landmarks, std::vector<Plane> &planes);
    void binGroundPoints(const SE3 pose, const VectorType &points, boost::multi_array<VectorType, 2> &scgf);
    // Data Association
    template <typename T>
    void matchModels(const std::vector<T> &currObjects, const std::vector<T> &mapObjects, std::vector<int> &matchIndices);
    template <typename T>
    std::vector<ObjectMatch<T>> matchFeatures(const SE3 tf, const std::vector<T> &currObjects,
                                              const std::vector<T> &mapObjects, const Scalar distThresh);
    // std::vector<ObjectMatch<T>> matchFeatures(const std::vector<T> &currObjects, const std::vector<T> &mapObjects, const Scalar distThresh);
    template <typename T>
    void addFeatureMatches(const VectorType &features, const T &object, const double dist, std::vector<ObjectMatch<T>> &matches);

    std::vector<Plane> getPrevGroundModel();
    CloudT getPrevGroundFeatures();

  private:
    SE3 T_Map_Anchor_; // Pose of ANCHOR frame in the Map frame
    // boost::shared_ptr<Plane> prevGPlane_;
    std::vector<Plane> prevGPlanes_;
    int numMapTrees_;
    double totalDistance_;
    FeatureModelParams fmParams_;
    bool firstScan_;
    double minPlanes_;
  };

} // namespace sloam
