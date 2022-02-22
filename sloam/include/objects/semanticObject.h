#pragma once

#include <definitions.h>

template <typename T>
class SemanticObject
{
public:
   // pure virtual function
   virtual Scalar distance(const T &model) const = 0;
   virtual Scalar distance(const PointT &point) const = 0;
   virtual void project(const SE3 &tf) = 0;
   T getModel() const {return model;};
   VectorType getFeatures() const {return features;};
   size_t id;
   bool isValid;
   VectorType features;
   T model; // object model
};