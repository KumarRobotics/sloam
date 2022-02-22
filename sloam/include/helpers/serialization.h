#pragma once

#include <definitions.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace boost
{
namespace serialization
{

    template <class Archive>
    void serialize(Archive &ar, PointT &g, const unsigned int version)
    {
        ar &g.getVector3fMap().data()[0];
        ar &g.getVector3fMap().data()[1];
        ar &g.getVector3fMap().data()[2];
    }

    template <class Archive>
    void serialize(Archive &ar, TreeVertex &vtx, const unsigned int version)
    {
        ar &vtx.treeId;
        ar &vtx.beam;
        ar &vtx.prevVertexSize;
        ar &vtx.radius;
        ar &vtx.isValid;
        ar &vtx.coords;
        ar &vtx.points;
    }

} // namespace boost
} // namespace serialization