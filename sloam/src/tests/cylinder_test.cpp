#include <stdio.h>
#include <stdlib.h>

#include <definitions.h>
#include <serialization.h>
#include <cylinder.h>
#include <plane.h>

#include "gtest/gtest.h"
using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

class CylinderTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    setupParams();
    PointT pta, ptb, ptc, ptd;
    pta.x = 0; pta.y = 0; pta.z = 0;
    ptb.x = 0; ptb.y = 1; ptb.z = 0;
    ptc.x = 1; ptc.y = 0; ptc.z = 0;
    ptd.x = 1; ptd.y = 1; ptd.z = 0;
    VectorType gfeatures{pta, ptb, ptc, ptd};
    plane = boost::make_shared<Plane>(gfeatures, params);

    std::ifstream ifs("/opt/sloam_ws/src/sloam/src/tests/aux/still_landmarks_t0");
    boost::archive::text_iarchive ia(ifs);
    // read class state from archive
    ia >> landmarks;
  }

  void setupParams()
  {

    params.scansPerSweep = 1;
    params.maxLidarDist = 30.0;
    params.maxGroundLidarDist = 30.0;
    params.minGroundLidarDist = 0.0;

    params.groundRadiiBins = 1;
    params.groundThetaBins = 1;
    params.groundMatchThresh = 2.0;
    params.groundRetainThresh = 0.1;

    params.maxTreeRadius = 0.3;
    params.maxAxisTheta = 10;
    params.maxFocusOutlierDistance = 0.5;
    params.roughTreeMatchThresh = 3.0;
    params.treeMatchThresh = 1.0;

    params.AddNewTreeThreshDist = 2.0;

    params.featuresPerTree = 2;
    params.numGroundFeatures = 3;
    params.defaultTreeRadius = 0.1;
  }

  std::vector<std::vector<TreeVertex>> landmarks;
  FeatureModelParams params;
  boost::shared_ptr<Plane> plane;
};

TEST_F(CylinderTest, Initalizes)
{

  for (auto l : landmarks)
  {
    auto c = Cylinder(l, *plane, params);
  }
  EXPECT_TRUE(true);
}

TEST_F(CylinderTest, DistanceToModel)
{

  std::vector<Cylinder> cylinders;
  for (auto l : landmarks)
  {
    auto c = Cylinder(l, *plane, params);
    if (c.isValid)
      cylinders.push_back(c);
  }

  std::cout << cylinders.size() << std::endl;
  float d = cylinders[0].distance(cylinders[0].model);
  EXPECT_EQ(0.0, d);
}

TEST_F(CylinderTest, DistanceToFeature)
{

  std::vector<Cylinder> cylinders;
  for (auto l : landmarks)
  {
    auto c = Cylinder(l, *plane, params);
    if (c.isValid)
      cylinders.push_back(c);
  }

  float d = cylinders[0].distance(cylinders[0].features[0]);
  EXPECT_NEAR(0.0, d, 0.1);
}

TEST_F(CylinderTest, TranslateModel)
{

  std::vector<Cylinder> cylinders;
  for (auto l : landmarks)
  {
    auto c = Cylinder(l, *plane, params);
    if (c.isValid)
      cylinders.push_back(c);
  }

  SE3 tf;
  tf.translation()[0] = 1;
  auto root = cylinders[0].model.root;
  cylinders[0].project(tf);

  EXPECT_EQ(root[0]+1, cylinders[0].model.root[0]);

}