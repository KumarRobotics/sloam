#include <sloam.h>

namespace sloam
{

  sloam::sloam()
  {
    // Initialize the map parameters
    T_Map_Anchor_ = SE3();
    firstScan_ = true;
    minPlanes_ = (fmParams_.groundRadiiBins*fmParams_.groundThetaBins)*0.1;
  }

  std::vector<Plane> sloam::getPrevGroundModel()
  {
    return prevGPlanes_;
  }

  CloudT sloam::getPrevGroundFeatures()
  {
    CloudT pc;
    for (const auto &ground : prevGPlanes_)
    {
      for (const auto &pt : ground.features)
        pc.points.push_back(pt);
    }

    pc.height = 1;
    pc.width = pc.points.size();
    return pc;
  }

  bool sloam::TwoStepOptimizePose(const SE3& poseEstimate, const bool optimTrees, const bool optimGround,
                                  const std::vector<ObjectMatch<Cylinder>> &allTMatch,
                                  const std::vector<ObjectMatch<Plane>> &allGMatch,
                                  SE3 &tf)
  {
    double treeOut[3];
    double groundOut[3];
    OptimizeXYYaw(poseEstimate, optimTrees, allTMatch, treeOut);
    OptimizeZRollPitch(poseEstimate, optimGround, allGMatch, groundOut);

    // roll, pitch, yaw
    double q[4];
    double angleAxis[3] = {groundOut[1], groundOut[2], treeOut[2]};
    ceres::AngleAxisToQuaternion(angleAxis, q);
    Quat quat(q[0], q[1], q[2], q[3]);
    tf.setQuaternion(quat);
    tf.translation()[0] = treeOut[0];
    tf.translation()[1] = treeOut[1];
    tf.translation()[2] = groundOut[0];
    return true;
  }

  void sloam::OptimizeXYYaw(const SE3 &poseEstimate, const bool optimize, const std::vector<ObjectMatch<Cylinder>> &allTMatch, double* out)
  {

    auto t = poseEstimate.translation();
    auto q = poseEstimate.unit_quaternion();
    double quat[4] = {q.w(), q.x(), q.y(), q.z()};
    double rpy[3];
    ceres::QuaternionToAngleAxis(quat, rpy);
    double params[6] = {t[0], t[1], t[2], rpy[0], rpy[1], rpy[2]};

    bool success = true;
    if(optimize)
    {
      ceres::LossFunction *loss = NULL;
      loss = new ceres::HuberLoss(0.1);
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);

      problem.AddParameterBlock(params, 6);
      // setting z, roll and pitch as constant
      ceres::SubsetParameterization *subset_parameterization =
        new ceres::SubsetParameterization(6, {2, 3, 4});
      problem.SetParameterization(params, subset_parameterization);

      for (auto tMatch : allTMatch)
      {
        Vector3 root = tMatch.object.model.root;
        Vector3 ray = tMatch.object.model.ray;
        double radius = (double)tMatch.object.model.radius;
        double weight = 1.0;
        ceres::CostFunction *cost =
            new ceres::AutoDiffCostFunction<XYYawCylinderCost, 1, 6>(
                new XYYawCylinderCost(tMatch.feature, root, ray, radius, weight));
        problem.AddResidualBlock(cost, loss, params);
      }

      ceres::Solver::Options options;
      options.max_num_iterations = 50;

      options.linear_solver_type = ceres::DENSE_QR;
      options.logging_type = ceres::SILENT;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      success = summary.termination_type == 0;
    }
    
    // bool x_diff = abs(abs(params[0]) - abs(t[0])) < 0.5;
    // bool y_diff = abs(abs(params[1]) - abs(t[1])) < 0.5;
    if (optimize && success)
    {
      out[0] = params[0];
      out[1] = params[1];
      out[2] = params[5];
    } else
    {
      out[0] = t[0];
      out[1] = t[1];
      out[2] = rpy[2];
    }
  }

  void sloam::OptimizeZRollPitch(const SE3 &poseEstimate, const bool optimize, const std::vector<ObjectMatch<Plane>> &allGMatch, double* out)
  {

    auto t = poseEstimate.translation();
    auto q = poseEstimate.unit_quaternion();
    double quat[4] = {q.w(), q.x(), q.y(), q.z()};
    double rpy[3];
    ceres::QuaternionToAngleAxis(quat, rpy);
    double params[6] = {t[0], t[1], t[2], rpy[0], rpy[1], rpy[2]};

    bool success = true;
    if(optimize)
    {
      ceres::LossFunction *loss = NULL;
      loss = new ceres::HuberLoss(0.1);
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);

      problem.AddParameterBlock(params, 6);
      // setting x, y and yaw as constant
      ceres::SubsetParameterization *subset_parameterization =
        new ceres::SubsetParameterization(6, {0, 1, 5});
      problem.SetParameterization(params, subset_parameterization);

      for (auto gMatch : allGMatch)
      {
        double weight = 1.0;
        ceres::CostFunction *cost =
            new ceres::AutoDiffCostFunction<ZRollPitchPlaneCost, 1, 6>(
                new ZRollPitchPlaneCost(gMatch.feature, gMatch.object.model.plane, weight));
        auto rbid = problem.AddResidualBlock(cost, loss, params);
      }

      ceres::Solver::Options options;
      options.max_num_iterations = 50;

      options.linear_solver_type = ceres::DENSE_QR;
      options.logging_type = ceres::SILENT;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      // if converged
      success = summary.termination_type == 0;
    }

    double output[3];
    if (optimize && success)
    {
      out[0] = params[2];
      out[1] = params[3];
      out[2] = params[4];
    } else
    {
      out[0] = t[2];
      out[1] = rpy[0];
      out[2] = rpy[1];
    }
  }

  bool sloam::OptimizePose(const SE3 &poseEstimate,
                           const std::vector<ObjectMatch<Cylinder>> &allTMatch,
                           const std::vector<ObjectMatch<Plane>> &allGMatch,
                           SE3 &tf)
  {

    ceres::LossFunction *loss = NULL;
    loss = new ceres::HuberLoss(0.1);
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    auto t = poseEstimate.translation();
    auto q = poseEstimate.unit_quaternion();
    double para_t[3] = {t[0], t[1], t[2]};
    double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
    ceres::LocalParameterization *q_parameterization =
        new ceres::EigenQuaternionParameterization();

    problem.AddParameterBlock(para_q, 4, q_parameterization);
    problem.AddParameterBlock(para_t, 3);

    for (auto tMatch : allTMatch)
    {
      Vector3 root = tMatch.object.model.root;
      Vector3 ray = tMatch.object.model.ray;
      double radius = (double)tMatch.object.model.radius;
      double weight = 1.0;
      ceres::CostFunction *cost =
          new ceres::AutoDiffCostFunction<CylinderCost, 1, 3, 4>(
              new CylinderCost(tMatch.feature, root, ray, radius, weight));
      problem.AddResidualBlock(cost, loss, para_t, para_q);
    }

    std::vector<ceres::ResidualBlockId> residual_block_ids;
    for (auto gMatch : allGMatch)
    {
      double weight = 1.0;
      // auto norm_feature = -(gMatch.feature - gMatch.object.model.centroid);
      auto norm_feature = gMatch.feature;
      ceres::CostFunction *cost =
          new ceres::AutoDiffCostFunction<PlaneCost, 1, 3, 4>(
              new PlaneCost(norm_feature, gMatch.object.model.plane, weight));
      auto rbid = problem.AddResidualBlock(cost, loss, para_t, para_q);
      residual_block_ids.push_back(rbid);
    }

    // SOLVE
    ceres::Solver::Options options;
    options.max_num_iterations = 50;

    options.linear_solver_type = ceres::DENSE_QR;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ceres::Problem::EvaluateOptions evoptions;
    // evoptions.residual_blocks = residual_block_ids;
    // double total_cost = 0.0;
    // std::vector<double> residuals;
    // problem.Evaluate(evoptions, &total_cost, &residuals, nullptr, nullptr);
    // for(auto i = 0; i < residuals.size(); ++i)
    //   std::cout << "R: " << residuals[i] << std::endl;
    // std::cout << "TOTAL COST: " << total_cost << std::endl;

    Quat tf_quat(para_q[3], para_q[0], para_q[1], para_q[2]);
    Vector3 tf_trans(para_t[0], para_t[1], para_t[2]);
    // ROS_DEBUG_STREAM("QUAT OUT: " << tf_quat.matrix().eulerAngles(1,1,2));
    bool success = summary.termination_type == 0;
    if (success)
    {
      tf.setQuaternion(tf_quat);
      tf.translation() = tf_trans;
    }

    ROS_INFO_STREAM("\n---------- OPTMIZATION DELTA OUTPUT -----------------\n"
                    << tf.matrix()
                    << "\n-----------------------------------------------------\n");

    double deltaDist = tf.translation().norm();
    ROS_DEBUG_STREAM("Distance: " << deltaDist << std::endl);
    return success;
  }

  template <typename T>
  std::vector<ObjectMatch<T>> sloam::matchFeatures(const SE3 tf, const std::vector<T> &currObjects, const std::vector<T> &mapObjects, const Scalar distThresh)
  {
    std::vector<ObjectMatch<T>> matches;
    for (const auto &co : currObjects)
    {
      T proj_obj = co;
      proj_obj.project(tf);
      // find closest model in map
      Scalar bestDist = distThresh + 100;
      T bestObject = mapObjects[0];
      for (const auto &mo : mapObjects)
      {
        Scalar d = mo.distance(proj_obj.getModel());
        if (d < bestDist)
        {
          bestDist = d;
          bestObject = mo;
        }
      }
      // create feature matches according to model association
      if (bestDist < distThresh)
      {
        addFeatureMatches(co.features, bestObject, bestDist, matches);
      }
    }

    return matches;
  }

  template <typename T>
  void sloam::addFeatureMatches(const VectorType &features, const T &object, const double dist, std::vector<ObjectMatch<T>> &matches)
  {

    for (const auto &feature : features)
    {
      ObjectMatch<T> match(feature, object, dist);
      matches.push_back(match);
    }
  }

  template <typename T>
  void sloam::matchModels(const std::vector<T> &currObjects, const std::vector<T> &mapObjects, std::vector<int> &matchIndices)
  {

    size_t treeCounter = 0;
    for (const auto &co : currObjects)
    {
      // find closest model in map
      Scalar bestDist = fmParams_.treeMatchThresh + 100;
      T bestObject = mapObjects[0];
      size_t bestKey = 0;
      size_t key = 0;
      for (const auto &mo : mapObjects)
      {
        Scalar d = mo.distance(co.getModel());
        if (d < bestDist)
        {
          bestDist = d;
          bestObject = mo;
          bestKey = key;
        }
        ++key;
      }
      // create feature matches according to model association
      if (bestDist < fmParams_.AddNewTreeThreshDist)
      {
        matchIndices[treeCounter] = bestKey;
      }
      treeCounter++;
    }
  }

  void sloam::binGroundPoints(const SE3 pose, const VectorType &points, boost::multi_array<VectorType, 2> &scgf)
  {

    PointT origin;
    origin.x = pose.translation()[0];
    origin.y = pose.translation()[1];
    origin.z = pose.translation()[2];

    // Add points into the corresponding bin
    for (const PointT &p : points)
    {
      // Sort into the multiarray grid
      Scalar pointRadius = euclideanDist2D(origin, p);
      if (pointRadius < fmParams_.maxGroundLidarDist &&
          pointRadius > fmParams_.minGroundLidarDist)
      {
        Scalar pointTheta = atan2(p.y - origin.y, p.x - origin.x);
        int pointRadiiBin =
            floor(pointRadius / (fmParams_.maxGroundLidarDist /
                                 (Scalar)fmParams_.groundRadiiBins));
        int pointThetaBin =
            floor((PIDEF + pointTheta) /
                  (2 * PIDEF / (Scalar)fmParams_.groundThetaBins));
        // Edge case when pointTheta = 180 etc
        pointRadiiBin =
            std::max(0, std::min(pointRadiiBin, fmParams_.groundRadiiBins - 1));
        pointThetaBin =
            std::max(0, std::min(pointThetaBin, fmParams_.groundThetaBins - 1));
        scgf[pointRadiiBin][pointThetaBin].push_back(p);
      }
    }
    // Retain the bottom k% of points in each cell
    for (int i = 0; i < scgf.shape()[0]; i++)
    {
      for (int j = 0; j < scgf.shape()[1]; j++)
      {
        if (scgf[i][j].size() > 0)
        {
          double retainNum = (1 / fmParams_.groundRetainThresh);
          if (retainNum < scgf[i][j].size())
          {
            // Bottom 10%
            int bottomIdx =
                int(scgf[i][j].size() / retainNum);

            // Sort by z value
            std::sort(
                scgf[i][j].begin(), scgf[i][j].end(),
                [](const PointT &p1, const PointT &p2)
                { return p1.z < p2.z; });
            // Erase all points after the groundRetainThresh%
            scgf[i][j].erase(scgf[i][j].begin() + bottomIdx, scgf[i][j].end());
          }
        }
      }
    }
  }

  void sloam::computeModels(SloamInput &in, std::vector<Cylinder> &landmarks, std::vector<Plane> &planes)
  {
    boost::multi_array<VectorType, 2> scgf(
        boost::extents[fmParams_.groundRadiiBins][fmParams_.groundThetaBins]);
    binGroundPoints(SE3(), in.groundCloud->points, scgf);

    Vector3 src_vec(0, 0, 1);
    Matrix4 tfm = in.poseEstimate.inverse().matrix().transpose();
    for (auto r = 0; r < fmParams_.groundRadiiBins; ++r)
    {
      for (auto t = 0; t < fmParams_.groundThetaBins; ++t)
      {
        Plane ground(scgf[r][t], fmParams_);
        // check if model is roughly consistent with what the ground should look like
        auto normal = tfm * ground.model.plane;
        Quat q_rot = Quat::FromTwoVectors(src_vec, normal.segment(0, 3));
        auto angles = q_rot.toRotationMatrix().eulerAngles(0, 1, 2);
        bool angleCheck = (angles[0] < 0.1 && angles[1] < 0.1 && angles[2] < 0.1) ||
                      (M_PI - abs(angles[0]) < 0.1 && M_PI - abs(angles[1]) < 0.1 && M_PI - abs(angles[2]) < 0.1);
        // ground should be under the robot
        bool heightCheck = (in.poseEstimate * ground.model.centroid)(2) < in.poseEstimate.translation()(2);
        if (ground.isValid && angleCheck && heightCheck)
          planes.push_back(ground);
      }
    }

    if (planes.size() == 0)
      return;

    // landmarks
    for (const std::vector<TreeVertex> t : in.landmarks)
    {
      auto approxTreePos = t[1].coords;
      Scalar bestDist = 100000;
      Plane bestPlane = planes[0];
      for (auto &ground : planes)
      {
        Scalar d = ground.distance(approxTreePos);
        if (d < bestDist)
        {
          bestDist = d;
          bestPlane = ground;
        }
      }
      auto c = Cylinder(t, bestPlane, fmParams_);
      if (c.isValid)
        landmarks.push_back(c);
    }
  }

  void sloam::projectModels(const SE3 &tf, std::vector<Cylinder> &landmarks, std::vector<Plane> &planes)
  {

    // projecting objects from current observations
    for (auto &p : planes)
    {
      p.project(tf);
    }

    for (auto &l : landmarks)
    {
      l.project(tf);
    }
  }

  bool sloam::RunSloam(SloamInput &in, SloamOutput &out)
  {
    std::vector<Plane> planes;
    std::vector<Cylinder> landmarks;
    computeModels(in, landmarks, planes);
    ROS_INFO("Num Tree models: %ld", landmarks.size());
    ROS_INFO("Num Ground models: %ld", planes.size());

    std::vector<int> matchIndices(landmarks.size(), -1);
    bool success = true;
    if (firstScan_)
    {
      projectModels(in.poseEstimate, landmarks, planes);
      // First run, pose will be same as odom
      out.T_Map_Curr = in.poseEstimate;
      // Matches will be all -1
      out.matches = matchIndices;
      out.tm = landmarks;
      prevGPlanes_ = planes;
      firstScan_ = false;
    }
    else
    {
      if (in.mapModels.size() == 0)
      {
        ROS_WARN("Not the first scan but map is empty! Ignoring Message");
        return false;
      }

      if (planes.size() == 0 || landmarks.size() == 0)
      {
        ROS_WARN("No ground or landmark models found");
        return false;
      }

      // matching models
      std::vector<ObjectMatch<Cylinder>> treeMatches = matchFeatures(in.poseEstimate, landmarks, in.mapModels, fmParams_.treeMatchThresh);
      std::vector<ObjectMatch<Plane>> planeMatches = matchFeatures(in.poseEstimate, planes, prevGPlanes_, 1.0);

      ROS_INFO("Num Tree feature matches: %ld", treeMatches.size());
      ROS_INFO("Num Ground feature Matches: %ld", planeMatches.size());

      // Optimize Pose
      // We do some checks to make sure we have enough information to perform pose estimation
      SE3 T_Delta = SE3();
      SE3 currPose = in.poseEstimate;
      bool treeCheck = landmarks.size() > fmParams_.minTreeModels && treeMatches.size() > 5 * fmParams_.featuresPerTree;
      bool groundCheck = planeMatches.size() > fmParams_.minGroundModels && planes.size() > minPlanes_;
      if(fmParams_.twoStepOptim)
      {
        success = TwoStepOptimizePose(in.poseEstimate, treeCheck, groundCheck, treeMatches, planeMatches, currPose);
      }
      else if(treeCheck && groundCheck)
      {
        success = OptimizePose(in.poseEstimate, treeMatches, planeMatches, T_Delta);
        if (success)
          currPose = T_Delta;
      }

      ROS_DEBUG_STREAM("\n---------- OPTMIZATION INITIAL OUTPUT -----------------\n"
                       << in.poseEstimate.matrix()
                       << "\n-----------------------------------------------------\n");

      ROS_DEBUG_STREAM("\n---------- OPTMIZATION POSE OUTPUT -----------------\n"
                       << currPose.matrix()
                       << "\n-----------------------------------------------------\n");

      projectModels(currPose, landmarks, planes);
      // Associate tree models
      matchModels(landmarks, in.mapModels, matchIndices);

      // Update the estimated pose
      prevGPlanes_ = planes;
      out.matches = matchIndices;
      out.T_Map_Curr = currPose;
      out.T_Delta = T_Delta;
      out.tm = landmarks;
    }
    return success;
  }

} // namespace sloam