#pragma once

#include <nodelet/nodelet.h>
#include <sloamNode.h>

namespace sloam {
class SLOAMNodelet : public nodelet::Nodelet {
 public:
  SLOAMNodelet() {}
  ~SLOAMNodelet() {}
  virtual void onInit();

 private:
  SLOAMNode::Ptr sloamNode;
};
}  // namespace sloam
