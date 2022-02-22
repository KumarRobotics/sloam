#pragma once

#include <algorithm>
#include <definitions.h>
#include <pcl/filters/random_sample.h>
#include <pcl/common/geometry.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <semanticObject.h>
#include <utils.h>

struct PlaneParameters
{
    Vector4 plane;
    Vector3 centroid;
    double dist;
};

class Plane : public SemanticObject<PlaneParameters>
{
public:
    explicit Plane(const VectorType &points, const FeatureModelParams &fmParams);
    Scalar distance(const PlaneParameters &tgt) const;
    Scalar distance(const PointT &point) const;
    void project(const SE3 &tf);

private:
    void filter(const VectorType &points, const FeatureModelParams &fmParams);
    void computeModel();
};

class ZRollPitchPlaneCost
{
public:
    ZRollPitchPlaneCost(Eigen::Vector3d curr_point, Eigen::Vector4d plane, double weight)
        : curr_point_(curr_point), plane_(plane), weight_(weight) {}

    template <typename T>
    bool operator()(const T *const params, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point_[0]), T(curr_point_[1]),
                                  T(curr_point_[2])};

        // TRANSLATION
        Eigen::Matrix<T, 3, 1> t_last_curr{params[0], params[1], params[2]};
        Eigen::Matrix<T, 4, 1> ground_p{T(plane_[0]), T(plane_[1]), T(plane_[2]),
                                        T(plane_[3])};

        // ROTATION AS ANGLE AXIS
        T r_last_curr[3] = {params[3], params[4], params[5]};

        // last point
        Eigen::Matrix<T, 3, 1> lp;
        ceres::AngleAxisRotatePoint(r_last_curr, cp.data(), lp.data());
        lp += t_last_curr;

        T denominator = ground_p.segment(0, 3).norm();
        T numerator = ceres::abs(ground_p[0] * lp[0] + ground_p[1] * lp[1] +
                                 ground_p[2] * lp[2] + ground_p[3]);
        T Distance = numerator / denominator;

        residual[0] = Distance;
        return true;
    }

private:
    // The measured x,y,z coordinate that should be on the cylinder.
    Eigen::Vector3d curr_point_;
    Eigen::Vector4d plane_;
    double weight_;
};

class PlaneCost
{
public:
    PlaneCost(Eigen::Vector3d curr_point, Eigen::Vector4d plane, double weight)
        : curr_point_(curr_point), plane_(plane), weight_(weight) {}

    template <typename T>
    bool operator()(const T *const para_t, const T *const para_q, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point_[0]), T(curr_point_[1]),
                                  T(curr_point_[2])};

        // TRANSLATION
        Eigen::Matrix<T, 3, 1> t_last_curr{para_t[0], para_t[1], para_t[2]};
        Eigen::Matrix<T, 4, 1> ground_p{T(plane_[0]), T(plane_[1]), T(plane_[2]),
                                        T(plane_[3])};

        // ROTATION AS ANGLE AXIS
        Eigen::Quaternion<T> q_last_curr{para_q[3], para_q[0], para_q[1], para_q[2]};

        // last point
        Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;
        // ceres::AngleAxisRotatePoint(r_last_curr, cp.data(), lp.data());
        // lp += t_last_curr;

        T denominator = ground_p.segment(0, 3).norm();
        T numerator = ceres::abs(ground_p[0] * lp[0] + ground_p[1] * lp[1] +
                                 ground_p[2] * lp[2] + ground_p[3]);
        T Distance = numerator / denominator;

        residual[0] = Distance;
        // residual[0] = ceres::abs(Distance * (weight_));
        return true;
    }

private:
    // The measured x,y,z coordinate that should be on the cylinder.
    Eigen::Vector3d curr_point_;
    Eigen::Vector4d plane_;
    double weight_;
};