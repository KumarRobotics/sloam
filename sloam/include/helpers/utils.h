#pragma once

#include <definitions.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <chrono>

inline float pow_2(const Scalar &x) { return x * x; }

inline float euclideanDist2D(const PointT &vecA, const PointT &vecB)
{
  return std::sqrt(pow_2(vecA.x - vecB.x) + pow_2(vecA.y - vecB.y));
}

inline PointT computeCentroid(const VectorType &features)
{
    PointT centroid;
    for (const auto &p : features)
    {
        centroid.x += p.x;
        centroid.y += p.y;
        centroid.z += p.z;
    }

    centroid.x /= (Scalar)features.size();
    centroid.y /= (Scalar)features.size();
    centroid.z /= (Scalar)features.size();
    return centroid;
}

inline void quatMsg2SE3(const geometry_msgs::QuaternionStampedConstPtr &quatMsg,
                 SE3 &pose)
{
  Quat q;
  q.w() = quatMsg->quaternion.w;
  q.x() = quatMsg->quaternion.x;
  q.y() = quatMsg->quaternion.y;
  q.z() = quatMsg->quaternion.z;
  pose.setQuaternion(q);
}

inline Vector3 rayPlaneIntersection(const Vector3 &planeCentroid, const Vector3 &planeNormal,
                                    const Vector3 &rayOrigin, const Vector3 &rayDirection)
{
  float denom = planeNormal.dot(rayDirection);
  if (abs(denom) > 0.001f)
  {
    float t = (planeCentroid - rayOrigin).dot(planeNormal) / denom;
    if (t >= 0.001f)
      return rayOrigin + t * rayDirection;
  }
  return rayOrigin;
}