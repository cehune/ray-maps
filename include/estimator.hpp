#pragma once
#include "kdtree.hpp"
#include <algorithm>

class IrradianceEstimator{
public:
    // estimator based on 
    float estimate(KdTree& tree, Vec3& x, Vec3& n, int K, float maxDist) const;
};

