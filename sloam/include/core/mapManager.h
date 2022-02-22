#include <definitions.h>
#include <cylinder.h>
#include <plane.h>

class MapManager {
    public:
        explicit MapManager(const float searchRadius = 50);
        std::vector<Cylinder> getMap();
        void getSubmap(const SE3& pose, std::vector<Cylinder>& submap);
        void updateMap(std::vector<Cylinder>& obs_tms, const std::vector<int>& matches);
    private:
        float sqSearchRadius;
        CloudT::Ptr landmarks_;
        std::vector<Cylinder> treeModels_;
        std::map<int, int> matchesMap;
        std::vector<size_t> treeHits_;
};