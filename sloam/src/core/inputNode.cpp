#include <queue>
#include <deque>
#include <ros/ros.h>
#include <ros/console.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Header.h>
#include <pcl_ros/point_cloud.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <definitions.h>
#include <sloamNode.h>

struct StampedSE3
{
    StampedSE3(SE3 p, ros::Time s) : pose(p), stamp(s){};
    StampedSE3() : pose(SE3()), stamp(ros::Time::now()){};
    SE3 pose;
    ros::Time stamp;
};

enum
{
    CLOUD_TOO_OLD,
    CLOUD_TOO_NEW,
    CLOUD_FOUND
};

class InputManager
{
public:
    explicit InputManager(ros::NodeHandle nh);
    bool Run();

private:
    void OdomCb_(const nav_msgs::OdometryConstPtr &odom_msg);
    int FindPC(const ros::Time stamp, CloudT::Ptr cloud);
    void PCCb_(const sensor_msgs::PointCloud2ConstPtr &cloudMsg);
    bool callSLOAM(SE3 relativeMotion, ros::Time stamp);
    void PublishAccumOdom_(const SE3 &relativeMotion);
    void Odom2SlamTf();
    void PublishOdomAsTf(const nav_msgs::Odometry &odom_msg,
                         const std::string &parent_frame_id,
                         const std::string &child_frame_id);

    std::queue<sensor_msgs::PointCloud2ConstPtr> pcQueue_;
    std::deque<StampedSE3> odomQueue_;
    ros::NodeHandle nh_;
    ros::Publisher pubPose_;
    ros::Subscriber OdomSub_;
    ros::Subscriber PCSub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster broadcaster_;

    // params
    std::string map_frame_id_;
    std::string odom_frame_id_;
    std::string robot_frame_id_;
    std::string cloud_topic_;
    std::string odom_topic_;
    float minOdomDistance_;
    float minSLOAMAltitude_;
    size_t maxQueueSize_;

    // vars
    boost::shared_ptr<sloam::SLOAMNode> sloam_ = nullptr;
    std::vector<SE3> keyPoses_;
    StampedSE3 latestOdom;
    bool firstOdom_;
    bool publishTf_;
    size_t odomCounter_;
    size_t odomFreqFilter_;
};

InputManager::InputManager(ros::NodeHandle nh) : nh_(nh), tf_listener_{tf_buffer_}
{
    nh_.param<float>("min_odom_distance", minOdomDistance_, 0.5);
    nh_.param<float>("min_sloam_z", minSLOAMAltitude_, 0.5);
    maxQueueSize_ = nh_.param("maxQueueSize", 30);
    odomFreqFilter_ = nh_.param("odom_freq_filter", 20);
    publishTf_ = nh_.param("publish_tf", true);
    nh_.param<std::string>("robot_frame_id", robot_frame_id_, "robot");
    nh_.param<std::string>("odom_frame_id", odom_frame_id_, "odom");
    nh_.param<std::string>("map_frame_id", map_frame_id_, "map");
    nh_.param<std::string>("cloud_topic", cloud_topic_, "cloud");
    nh_.param<std::string>("odom_topic", odom_topic_, "/quadrotor/vio/odom");

    pubPose_ =
        nh_.advertise<geometry_msgs::PoseStamped>("debug/ref_frame_sloam_pose", 10);
    OdomSub_ = nh_.subscribe(odom_topic_, 10, &InputManager::OdomCb_, this);
    PCSub_ = nh_.subscribe(cloud_topic_, 10, &InputManager::PCCb_, this);

    auto sloam_ptr = boost::make_shared<sloam::SLOAMNode>(nh_);
    sloam_ = std::move(sloam_ptr);
    firstOdom_ = true;
    odomCounter_ = 0;

    ROS_INFO("SLOAM initialized");
}

void InputManager::OdomCb_(const nav_msgs::OdometryConstPtr &odom_msg)
{
    odomCounter_++;
    if (odomCounter_ % odomFreqFilter_ != 0)
        return;
    odomCounter_ = 0;

    auto pose = odom_msg->pose.pose;
    ros::Time odomStamp = odom_msg->header.stamp;
    Quat rot(pose.orientation.w, pose.orientation.x,
             pose.orientation.y, pose.orientation.z);
    Vector3 pos(pose.position.x, pose.position.y, pose.position.z);

    SE3 odom = SE3();
    odom.setQuaternion(rot);
    odom.translation() = pos;

    // odom is in cam frame, we need it in robot frame
    geometry_msgs::TransformStamped transform_cam_body;
    try
    {
        auto transform_cam_body = tf_buffer_.lookupTransform(
            odom_msg->child_frame_id, robot_frame_id_, ros::Time(0));
        SE3 cam_body_tf(tf2::transformToEigen(transform_cam_body).matrix().cast<double>());
        odom = odom * cam_body_tf;
    }
    catch (tf2::TransformException &ex)
    {
        ROS_INFO_THROTTLE(10, "Camera to body transform is not found, now setting it as identity... If you're in simulator, ignore this.");
    }

    if (firstOdom_ && pose.position.z < minSLOAMAltitude_)
    {
        ROS_INFO_THROTTLE(5, "Quad is too low, will not call sloam");
        return;
    }

    odomQueue_.emplace_back(odom, odomStamp);
    if (odomQueue_.size() > 10 * maxQueueSize_)
        odomQueue_.pop_front();
}

bool InputManager::Run()
{

    if (odomQueue_.empty() || pcQueue_.empty())
        return false;

    // ROS_DEBUG_THROTTLE(5, "First odom stamp: %f", odomQueue_.front().stamp.toSec());
    // ROS_DEBUG_THROTTLE(5, "First cloud stamp: %f", pcQueue_.front()->header.stamp.toSec());
    // ROS_DEBUG_THROTTLE(5, "Last odom stamp: %f", odomQueue_.back().stamp.toSec());
    // ROS_DEBUG_THROTTLE(5, "Last cloud stamp: %f", pcQueue_.back()->header.stamp.toSec());

    for (auto i = 0; i < odomQueue_.size(); ++i)
    {
        auto odom = odomQueue_[i];
        // Use odom to estimate motion since last key frame
        SE3 currRelativeMotion = latestOdom.pose.inverse() * odom.pose;

        if (firstOdom_)
        {
            ROS_INFO_THROTTLE(1.0, "first sloam call");
            if (callSLOAM(currRelativeMotion, odom.stamp))
            {
                firstOdom_ = false;
                latestOdom.pose = odom.pose;
                latestOdom.stamp = odom.stamp;
                if(publishTf_)
                    Odom2SlamTf();
                return true;
            }
        }
        else
        {
            double accumMovement = currRelativeMotion.translation().norm();
            if (accumMovement > minOdomDistance_)
            {
                ROS_DEBUG_THROTTLE(1.0, "Distance %f", (accumMovement));
                if (callSLOAM(currRelativeMotion, odom.stamp))
                {
                    latestOdom.pose = odom.pose;
                    latestOdom.stamp = odom.stamp;
                    if(publishTf_)
                        Odom2SlamTf();
                    return true;
                }
            }
        }
    }
    return false;
}

bool InputManager::callSLOAM(SE3 relativeMotion, ros::Time stamp)
{

    CloudT::Ptr cloud(new CloudT);
    auto r = FindPC(stamp, cloud);
    if (r == CLOUD_FOUND)
    {
        odomQueue_.pop_front();
        SE3 keyPose = SE3();
        SE3 prevKeyPose = firstOdom_ ? SE3() : keyPoses_[keyPoses_.size() - 1];
        pcl_conversions::toPCL(stamp, cloud->header.stamp);
        bool success = sloam_->run(relativeMotion, prevKeyPose, cloud, stamp, keyPose);
        if (success)
        {
            keyPoses_.push_back(keyPose);
            return true;
        }
    }
    else
    {
        if (r == CLOUD_TOO_NEW)
            odomQueue_.pop_front();
        ROS_DEBUG_THROTTLE(1, "Corresponding point cloud not found. Skipping.");
    }
    return false;
}

void InputManager::PCCb_(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
    pcQueue_.push(cloudMsg);
    if (pcQueue_.size() > maxQueueSize_)
        pcQueue_.pop();
}

int InputManager::FindPC(const ros::Time stamp, CloudT::Ptr cloud)
{
    if (pcQueue_.empty())
        return false;
    while (!pcQueue_.empty())
    {
        if (pcQueue_.front()->header.stamp.toSec() < stamp.toSec() - 0.05)
        {
            // message too old
            ROS_DEBUG_THROTTLE(1, "PC MSG TOO OLD");
            pcQueue_.pop();
        }
        else if (pcQueue_.front()->header.stamp.toSec() > stamp.toSec() + 0.05)
        {
            // message too new
            ROS_DEBUG_THROTTLE(1, "PC MSG TOO NEW");
            return CLOUD_TOO_NEW;
        }
        else
        {
            pcl::fromROSMsg(*pcQueue_.front(), *cloud);
            pcQueue_.pop();
            ROS_DEBUG("Calling SLOAM");
            return CLOUD_FOUND;
        }
    }
    return CLOUD_TOO_OLD;
}

void InputManager::Odom2SlamTf()
{
    if (keyPoses_.size() == 0)
        return;

    auto slam_pose = keyPoses_[keyPoses_.size()-1];
    //vio_odom is odomTb, slam_pose which is slamTb
    //get odom2slam transform slamTodom = slamTb * inv(odomTb)

    // compute the tf based on odom when the graph slam optimization is called
    SE3 vio_odom = latestOdom.pose;
    SE3 odom2slam = slam_pose * (vio_odom.inverse());

    std::string parent_frame_id = map_frame_id_;
    std::string child_frame_id = odom_frame_id_;
    PublishOdomAsTf(sloam::toRosOdom_(odom2slam, map_frame_id_, latestOdom.stamp), parent_frame_id, child_frame_id);
    // publish pose AFTER TF so that the visualization looks correct
    pubPose_.publish(sloam::toRosOdom_(slam_pose, map_frame_id_, latestOdom.stamp));
}

void InputManager::PublishOdomAsTf(const nav_msgs::Odometry &odom_msg,
                                   const std::string &parent_frame_id,
                                   const std::string &child_frame_id)
{
    geometry_msgs::TransformStamped tf;
    tf.header = odom_msg.header;
    // note that normally parent_frame_id should be the same as
    // odom_msg.header.frame_id
    tf.header.frame_id = parent_frame_id;
    tf.child_frame_id = child_frame_id;
    tf.transform.translation.x = odom_msg.pose.pose.position.x;
    tf.transform.translation.y = odom_msg.pose.pose.position.y;
    tf.transform.translation.z = odom_msg.pose.pose.position.z;
    tf.transform.rotation = odom_msg.pose.pose.orientation;
    broadcaster_.sendTransform(tf);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sloam");
    ros::NodeHandle n("sloam");
    InputManager in(n);
    // ros::spin();

    ros::Rate r(20); // 10 hz
    while (ros::ok())
    {
        for (auto i = 0; i < 10; ++i)
        {
            ros::spinOnce();
            if (i % 5 == 0)
                in.Run();
            r.sleep();
        }
    }

    return 0;
}