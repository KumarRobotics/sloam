#include <pluginlib/class_list_macros.h>
#include <sloamNodelet.h>

namespace sloam {
void SLOAMNodelet::onInit() {
  // sloam.reset(new SLOAM(getMTPrivateNodeHandle()));
  sloamNode.reset(new SLOAMNode(getPrivateNodeHandle()));
  ROS_INFO("Created Sloam Nodelet");
}

} // namespace sloam

PLUGINLIB_EXPORT_CLASS(sloam::SLOAMNodelet, nodelet::Nodelet);
