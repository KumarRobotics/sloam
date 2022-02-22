#include <vizTools.h>

namespace sloam
{

  // functions adapted from ros tf2/tf2_eigen 
  geometry_msgs::Quaternion toMsg_(const Quat &in)
  {
    geometry_msgs::Quaternion msg;
    msg.w = in.w();
    msg.x = in.x();
    msg.y = in.y();
    msg.z = in.z();
    return msg;
  }

  geometry_msgs::Point toMsg_(const Vector3 &in)
  {
    geometry_msgs::Point msg;
    msg.x = in.x();
    msg.y = in.y();
    msg.z = in.z();
    return msg;
  }

  geometry_msgs::Quaternion toRosQuat_(const SO3 &R)
  {
    return toMsg_(R.unit_quaternion());
  }

  geometry_msgs::Pose toRosPose_(const SE3 &T)
  {
    geometry_msgs::Pose pose;
    pose.position = toMsg_(T.translation());
    pose.orientation = toRosQuat_(T.so3());
    return pose;
  }

  nav_msgs::Odometry toRosOdom_(const SE3 &pose, const std::string slam_ref_frame, const ros::Time stamp)
  {
    nav_msgs::Odometry odom;
    odom.header.frame_id = slam_ref_frame;
    odom.header.stamp = stamp;

    geometry_msgs::Pose rosPose;
    rosPose.position.x = pose.translation()[0];
    rosPose.position.y = pose.translation()[1];
    rosPose.position.z = pose.translation()[2];
    auto quat = pose.unit_quaternion();
    rosPose.orientation.w = quat.w();
    rosPose.orientation.x = quat.x();
    rosPose.orientation.y = quat.y();
    rosPose.orientation.z = quat.z();
    odom.pose.pose = rosPose;
    boost::array<double, 36> cov;
    for (int i = 0; i < 6; i++)
    {
      double var = 0.0;
      if (i < 3)
      {
        var = 0.01;
      }
      else
      {
        var = 0.01;
      }
      for (int j = 0; j < 6; j++)
      {
        if (j == i)
        {
          cov[6 * i + j] = var;
        }
        else
        {
          cov[6 * i + j] = 1e-5;
        }
      }
    }
    odom.pose.covariance = cov;
    return odom;
  }

  visualization_msgs::MarkerArray vizTrajectory(const std::vector<SE3> &poses)
  {
    visualization_msgs::MarkerArray tMarkerArray;
    visualization_msgs::Marker points, line_strip;
    points.header.frame_id = line_strip.header.frame_id = "quadrotor/map";
    points.header.stamp = line_strip.header.stamp = ros::Time::now();
    points.ns = line_strip.ns = "points_and_lines";
    points.action = line_strip.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = line_strip.pose.orientation.w = 1.0;

    points.id = 1000;
    line_strip.id = 1001;

    points.type = visualization_msgs::Marker::POINTS;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    // POINTS markers use x and y scale for width/height respectively
    points.scale.x = 0.5;
    points.scale.y = 0.5;

    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    line_strip.scale.x = 0.5;

    // Points are green
    points.color.g = 1.0f;
    points.color.a = 1.0;

    // Line strip is blue
    line_strip.color.b = 1.0;
    line_strip.color.a = 1.0;

    for (auto o : poses)
    {
      // Between odom line strips
      geometry_msgs::Point pt;
      auto obs_posit = o.translation();
      pt.x = obs_posit[0];
      pt.y = obs_posit[1];
      pt.z = obs_posit[2];
      points.points.push_back(pt);
      line_strip.points.push_back(pt);
    }
    tMarkerArray.markers.push_back(points);
    tMarkerArray.markers.push_back(line_strip);
    return tMarkerArray;
  }

  geometry_msgs::PoseStamped makeROSPose(const SE3 &tf, std::string frame_id)
  {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = frame_id;
    pose.pose = toRosPose_(tf);
    return pose;
  }

  SE3 toSE3(geometry_msgs::PoseStamped pose)
  {
    Vector3d pos;
    Quaterniond quat;
    tf2::fromMsg(pose.pose.position, pos);
    tf2::fromMsg(pose.pose.orientation, quat);

    SE3 tf;
    tf.translation() = pos;
    tf.setQuaternion(quat);

    return tf;
  }

  void vizTreeModels(const std::vector<Cylinder> &scanTm,
                     visualization_msgs::MarkerArray &tMarkerArray,
                     size_t &cylinderId)
  {
    for (const auto &tree : scanTm)
    {

      Scalar maxTreeRadius = 0.25;
      Scalar maxAxisTheta = 45;
      visualization_msgs::Marker marker;
      marker.header.frame_id = "quadrotor/map";
      // marker.header.frame_id = "quadrotor/map";
      marker.header.stamp = ros::Time();
      marker.id = cylinderId;
      marker.type = visualization_msgs::Marker::CYLINDER;
      marker.action = visualization_msgs::Marker::ADD;

      // Center of cylinder
      marker.pose.position.x = tree.model.root[0] + 0.5 * tree.model.ray[0];
      marker.pose.position.y = tree.model.root[1] + 0.5 * tree.model.ray[1];
      marker.pose.position.z = tree.model.root[2] + 0.5 * tree.model.ray[1];
      // marker.pose.position.z = 0;

      // Orientation of cylidner
      Vector3 src_vec(0, 0, 1);
      Quat q_rot = Quat::FromTwoVectors(src_vec, tree.model.ray);
      marker.pose.orientation.x = q_rot.x();
      marker.pose.orientation.y = q_rot.y();
      marker.pose.orientation.z = q_rot.z();
      marker.pose.orientation.w = q_rot.w();

      marker.scale.x = 2 * tree.model.radius;
      marker.scale.y = 2 * tree.model.radius;
      marker.scale.z = 10;

      if (cylinderId < 200000)
      {
        marker.color.a = 0.5;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
      }
      else
      {
        marker.color.a = 0.5;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
      }

      tMarkerArray.markers.push_back(marker);
      cylinderId++;
    }
    // HACK TO MAKE SURE WE ALWAYS DELETE THE PREVIOUS CYLINDERS
    for (auto i = cylinderId; i < cylinderId + 50; ++i)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "quadrotor/map";
      marker.header.stamp = ros::Time();
      marker.id = i;
      marker.type = visualization_msgs::Marker::CYLINDER;
      marker.action = visualization_msgs::Marker::DELETE;
      tMarkerArray.markers.push_back(marker);
    }
  }

  visualization_msgs::MarkerArray vizGroundModel(const std::vector<Plane> &gplanes, const std::string &frame_id, int idx)
  {

    visualization_msgs::MarkerArray groundModels;
    for (const auto &g : gplanes)
    {
      auto gmodel = vizGroundModel(g, frame_id, idx);
      groundModels.markers.push_back(gmodel);
      idx++;
    }

    // HACK TO MAKE SURE WE ALWAYS DELETE THE PREVIOUS GROUND MODELS
    for (auto i = idx; i < idx + 300; ++i)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "quadrotor/map";
      marker.header.stamp = ros::Time();
      marker.id = i;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::DELETE;
      marker.ns = "plane";
      groundModels.markers.push_back(marker);
    }
    return groundModels;
  }
  visualization_msgs::Marker vizGroundModel(const Plane &gplane, const std::string &frame_id, const int idx)
  {
    visualization_msgs::Marker cube;
    cube.type = visualization_msgs::Marker::CUBE;
    cube.action = visualization_msgs::Marker::ADD;
    cube.id = idx;
    cube.ns = "plane";
    cube.scale.x = 2.5;
    cube.scale.y = 2.5;
    cube.scale.z = 0.1;
    if (idx < 200)
    {
      cube.color.r = 1.0;
      cube.color.g = 1.0;
      cube.color.b = 0.0;
      cube.color.a = 0.5;
    }
    else
    {
      cube.color.r = 1.0;
      cube.color.g = 1.0;
      cube.color.b = 1.0;
      cube.color.a = 0.5;
    }

    cube.header.frame_id = frame_id;
    cube.header.stamp = ros::Time();
    cube.pose.position.x = gplane.model.centroid(0);
    cube.pose.position.y = gplane.model.centroid(1);
    cube.pose.position.z = gplane.model.centroid(2);

    Vector3 src_vec(0, 0, 1);
    Quat q_rot = Quat::FromTwoVectors(src_vec, gplane.model.plane.segment(0, 3));
    cube.pose.orientation.x = q_rot.x();
    cube.pose.orientation.y = q_rot.y();
    cube.pose.orientation.z = q_rot.z();
    cube.pose.orientation.w = q_rot.w();
    return cube;
  }

  void landmarksToCloud(const std::vector<std::vector<TreeVertex>> &landmarks,
                        CloudT::Ptr &cloud)
  {
    size_t numPoints = 0;
    size_t color_id = 0;
    std::vector<float> color_values((int)landmarks.size());

    std::iota(std::begin(color_values), std::end(color_values), 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(color_values.begin(), color_values.end(), gen);

    ROS_DEBUG_STREAM("Color values size: " << color_values.size());
    for (const auto &tree : landmarks)
    {
      if (tree[0].treeId == -1)
        continue;
      for (auto vtx : tree)
      {
        for (const auto &point : vtx.points)
        {
          PointT vp = point;
          vp.intensity = color_values[color_id];
          cloud->push_back(vp);
          numPoints++;
        }
      }
      color_id++;
    }
    cloud->width = numPoints;
    cloud->height = 1;
    cloud->is_dense = false;
  }

  cv::Mat DecodeImage(const sensor_msgs::ImageConstPtr &image_msg)
  {
    cv::Mat image;
    cv_bridge::CvImagePtr input_bridge;
    try
    {
      input_bridge = cv_bridge::toCvCopy(image_msg, image_msg->encoding);
      image = input_bridge->image;
    }
    catch (cv_bridge::Exception &ex)
    {
      ROS_ERROR("Failed to convert depth image");
    }
    return image;
  }

} // namespace sloam