cmake_minimum_required(VERSION 3.16)
project(sloam)

set(CMAKE_CXX_STANDARD 17)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_TESTING "Build tests" On)

set(CC_TARGET_PREFIX sloam)
include(CMakeHelpers)

find_package(catkin REQUIRED COMPONENTS
  sloam_msgs
  sensor_msgs
  std_msgs
  geometry_msgs
  tf2_geometry_msgs
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  tf2_ros
  tf2_eigen
  tf2
  tf
  roscpp
)

find_package(Boost REQUIRED system serialization)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

catkin_package()

cc_library(
  NAME log
  DEPS fmt::fmt-header-only glog::glog
  INTERFACE)

cc_library(
    NAME base
    DEPS ${Boost_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${ONNX_RUNTIME_LIB}
    INCS include/helpers ${Boost_INCLUDE_DIR} ${Sophus_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}
    INTERFACE)

cc_library(
  NAME objects
  DEPS sloam_base
  INCS include/objects
  SRCS "src/objects/plane.cpp" "src/objects/cylinder.cpp"
)

cc_library(
    NAME viz
    DEPS sloam_log sloam_base sloam_objects 
    INCS include/viz 
    SRCS "src/viz/vizTools.cpp"
    )

cc_library(
    NAME segmentation
    INCS include/segmentation
    SRCS "src/segmentation/inference.cpp" "src/segmentation/trellis.cpp"
    DEPS sloam_base)

cc_library(
  NAME core
  INCS include/core
  SRCS "src/core/mapManager.cpp" "src/core/sloam.cpp"
  DEPS sloam_log sloam_base sloam_objects sloam_viz sloam_segmentation
)

cc_binary(
  NAME inference_node
  SRCS "src/segmentation/inferenceNode.cpp" 
  DEPS sloam_base sloam_segmentation)

cc_binary(
  NAME node
  SRCS "src/core/sloamNode.cpp" "src/core/inputNode.cpp"
  DEPS sloam_log sloam_base sloam_objects sloam_core sloam_viz sloam_segmentation
)

# TESTS
if(BUILD_TESTING)
enable_testing()
# Example usage: ./sloam_ws/devel/lib/sloam/sloam_all_test --gtest_filter=CylinderTest* (optional filter)
cc_test(
  NAME all_test 
  SRCS "src/tests/run_all_test.cpp" "src/tests/plane_test.cpp" "src/tests/cylinder_test.cpp" "src/tests/core_test.cpp"
  DEPS sloam_base sloam_objects sloam_core GTest::GTest)
endif()