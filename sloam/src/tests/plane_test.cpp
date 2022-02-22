#include <stdio.h>
#include <stdlib.h>

#include <definitions.h>
#include <serialization.h>
#include <cylinder.h>

#include "gtest/gtest.h"
using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

class PlaneTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    setupParams();

    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug))
      ros::console::notifyLoggerLevelsChanged();
  }

  void setupParams()
  {

    params.scansPerSweep = 1;
    params.maxLidarDist = 15.0;
    params.maxGroundLidarDist = 30.0;
    params.minGroundLidarDist = 0.0;

    params.groundRadiiBins = 1;
    params.groundThetaBins = 18;
    params.groundMatchThresh = 2.0;
    params.groundRetainThresh = 0.1;
    params.numGroundFeatures = 1;

    params.maxTreeRadius = 0.3;
    params.maxAxisTheta = 10;
    params.maxFocusOutlierDistance = 0.5;
    params.roughTreeMatchThresh = 3.0;
    params.treeMatchThresh = 1.0;

    params.AddNewTreeThreshDist = 2.0;

    params.featuresPerTree = 2;
    params.defaultTreeRadius = 0.1;
  }

  FeatureModelParams params;
};

TEST_F(PlaneTest, Initalizes)
{
  PointT pta, ptb, ptc;
  pta.x = 0; pta.y = 0; pta.z = 0;
  ptb.x = 0; ptb.y = 1; ptb.z = 0;
  ptc.x = 1; ptc.y = 0; ptc.z = 0;
  VectorType gfeatures{pta, ptb, ptc};
  Plane plane(gfeatures, params);
  EXPECT_TRUE(plane.isValid);
}

TEST_F(PlaneTest, DistanceToFeature)
{

  PointT pta, ptb, ptc;
  pta.x = 0; pta.y = 0; pta.z = 0;
  ptb.x = 0; ptb.y = 1; ptb.z = 0;
  ptc.x = 1; ptc.y = 0; ptc.z = 0;
  VectorType gfeatures{pta, ptb, ptc};
  Plane plane(gfeatures, params);

  float d = plane.distance(pta);
  EXPECT_NEAR(0.0, d, 0.1);
}

TEST_F(PlaneTest, TranslateModel)
{

  PointT pta, ptb, ptc;
  pta.x = 0; pta.y = 0; pta.z = 0;
  ptb.x = 0; ptb.y = 1; ptb.z = 0;
  ptc.x = 1; ptc.y = 0; ptc.z = 0;
  VectorType gfeatures{pta, ptb, ptc};
  Plane plane(gfeatures, params);

  SE3 tf;
  tf.translation()[2] = 1;
  auto centroid = plane.model.centroid;
  plane.project(tf);
  EXPECT_EQ(centroid[2]+1, plane.model.centroid[2]);
}