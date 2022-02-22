#include <plane.h>

Plane::Plane(const VectorType &points, const FeatureModelParams &fmParams)
{
    // filter(points, fmParams);
    features = points;
    if (features.size() < fmParams.numGroundFeatures)
    {
        isValid = false;
    }
    else
    {
        computeModel();
        features.resize(fmParams.numGroundFeatures);
        isValid = true;
    }
}

void Plane::filter(const VectorType &points, const FeatureModelParams &fmParams)
{
    int numFeatures = 0;
    PointT origin = computeCentroid(points);
    boost::multi_array<VectorType, 2> scgf(
        boost::extents[fmParams.groundRadiiBins][fmParams.groundThetaBins]);
    // Add points into the corresponding bin
    for (const PointT &p : points)
    {
        // Sort into the multiarray grid
        Scalar pointRadius = euclideanDist2D(origin, p);
        if (pointRadius < fmParams.maxGroundLidarDist &&
            pointRadius > fmParams.minGroundLidarDist)
        {
            Scalar pointTheta = atan2(p.y - origin.y, p.x - origin.x);
            int pointRadiiBin =
                floor(pointRadius / (fmParams.maxGroundLidarDist /
                                     (Scalar)fmParams.groundRadiiBins));
            int pointThetaBin =
                floor((PIDEF + pointTheta) /
                      (2 * PIDEF / (Scalar)fmParams.groundThetaBins));
            // Edge case when pointTheta = 180 etc
            pointRadiiBin =
                std::max(0, std::min(pointRadiiBin, fmParams.groundRadiiBins - 1));
            pointThetaBin =
                std::max(0, std::min(pointThetaBin, fmParams.groundThetaBins - 1));
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
                float retainNum = (1 / fmParams.groundRetainThresh);
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
                    // Erase all points after the 10%
                    scgf[i][j].erase(scgf[i][j].begin() + bottomIdx, scgf[i][j].end());
                    numFeatures =
                        numFeatures + bottomIdx + 1; // Update the number of features
                }

                // adding to list of features
                for (int k = 0; k < scgf[i][j].size(); k++)
                {
                    features.push_back(scgf[i][j][k]);
                }
            }
        }
    }

    // CloudT::Ptr cloud (new CloudT);
    // CloudT::Ptr cloud_filtered (new CloudT);
    // cloud->points = features;
    // cloud->height = 1;
    // cloud->width = features.size();

    // pcl::StatisticalOutlierRemoval<PointT> sor;
    // sor.setInputCloud(cloud);
    // sor.setMeanK(50);
    // sor.setStddevMulThresh(1.0);
    // sor.filter(*cloud_filtered);
    // features = cloud_filtered->points;
}

void Plane::computeModel()
{
    // Compute centroid
    PointT centroid = computeCentroid(features);

    // Subtract out centroid
    MatrixX centeredGf = MatrixX::Zero(3, features.size());
    int counter = 0;
    for (auto p : features)
    {
        centeredGf(0, counter) = p.x - centroid.x;
        centeredGf(1, counter) = p.y - centroid.y;
        centeredGf(2, counter) = p.z - centroid.z;
        counter++;
    }

    // Estimate plane using SVD
    JacobiSVD<MatrixX> svd(centeredGf, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Normal is left singular vector of least singular value
    Vector3 normal = (svd.matrixU().block(0, 2, 3, 1)).eval();
    // normal = normal / normal.norm();
    Scalar d = -(normal(0) * centroid.x + normal(1) * centroid.y +
                 normal(2) * centroid.z);

    model.plane(0) = normal(0);
    model.plane(1) = normal(1);
    model.plane(2) = normal(2);
    model.plane(3) = d;

    model.centroid(0) = centroid.x;
    model.centroid(1) = centroid.y;
    model.centroid(2) = centroid.z;
}

// not implemented
Scalar Plane::distance(const PlaneParameters &tgt) const
{
    return (model.centroid - tgt.centroid).norm();
};

Scalar Plane::distance(const PointT &point) const
{
    // PointT norm_pt;
    // norm_pt.x = -(point.x - model.centroid(0));
    // norm_pt.y = -(point.y - model.centroid(1));
    // norm_pt.z = -(point.z - model.centroid(2));
    // return (abs(model.plane[0] * norm_pt.x +
    //             model.plane[1] * norm_pt.y +
    //             model.plane[2] * norm_pt.z) /
    //         model.plane.segment(0,3).norm());
    return (abs(model.plane[0] * point.x +
                model.plane[1] * point.y +
                model.plane[2] * point.z +
                model.plane[3]) /
            model.plane.segment(0,3).norm());
}

void Plane::project(const SE3 &tf)
{
    // features
    Matrix4 tfm = tf.matrix();
    Affine3 affine(tfm);
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        if (!std::isfinite(features[i].x) ||
            !std::isfinite(features[i].y) ||
            !std::isfinite(features[i].z))
            continue;
        features[i] = pcl::transformPoint(features[i], affine);
    }

    // model
    model.plane = tfm.inverse().transpose() * model.plane;
    model.centroid = tf * model.centroid;
    // ROS_DEBUG_STREAM("model.plane:" << model.plane);
    // ROS_DEBUG_STREAM("model.centroid:" << model.centroid);

    // computeModel();
    // ROS_DEBUG_STREAM("model.plane:" << model.plane);
    // ROS_DEBUG_STREAM("model.centroid:" << model.centroid);
}