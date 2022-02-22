#include <mapManager.h>

MapManager::MapManager(const float searchRadius){
    sqSearchRadius = searchRadius*searchRadius;
    landmarks_.reset(new CloudT);
}

void MapManager::updateMap(std::vector<Cylinder>& obs_tms, const std::vector<int>& matches){
    // for(auto i = 0; i < matches.size(); i++){
    size_t i = 0;
    for(auto const& tree : obs_tms){
        PointT pt;
        pt.x = tree.model.root[0]; pt.y = tree.model.root[1]; pt.z = tree.model.root[2];
        if(matches[i] == -1){
            landmarks_->push_back(pt);
            treeModels_.push_back(tree);
            treeHits_.push_back(1);
        } else {
            // transform from observation to map index
            int matchIdx = matchesMap[matches[i]];
            landmarks_->points[matchIdx] = pt;
            treeModels_[matchIdx] = tree; 
            treeHits_[matchIdx] += 1;
        }
        i++;
    }
    matchesMap.clear();
}

std::vector<Cylinder> MapManager::getMap() 
{
    std::vector<Cylinder> map;
    for(auto i = 0; i < treeModels_.size(); ++i)
    {
        if(treeHits_[i] > 2)
            map.push_back(treeModels_[i]);
    }
    return map;
}

void MapManager::getSubmap(const SE3& pose, std::vector<Cylinder>& submap){
    if(landmarks_->size() == 0) return;

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(landmarks_);
    std::vector<int> pointIdxKNNSearch;
    std::vector<float> pointKNNSquaredDistance;
    PointT searchPoint;

    // Search for nearby trees
    searchPoint.x = pose.translation()[0];
    searchPoint.y = pose.translation()[1];
    searchPoint.z = 1;
    if(kdtree.nearestKSearch(searchPoint, 100,
            pointIdxKNNSearch, pointKNNSquaredDistance) > 0){

        int idx_count = 0;
        auto map_size = treeModels_.size();
        for(auto map_idx : pointIdxKNNSearch){
            if(map_size - map_idx < 200)
            {
                matchesMap.insert(std::pair<int, int>(idx_count, map_idx));
                submap.push_back(treeModels_[map_idx]);
                idx_count++;
            }
        }

    } else {
        ROS_INFO("Not enough landmarks around pose: Total: %ld", pointIdxKNNSearch.size());
    }
}