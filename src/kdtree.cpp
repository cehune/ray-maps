#include "kdtree.hpp"

void KdTree::build(const std::vector<Ray>& rays, const AABB& sceneBounds) {
    _rays = &rays;

    // get min diagonal
    Vec3 diagonal = sceneBounds.max - sceneBounds.min;
    _rMinDiag = diagonal.norm() * R_MIN_FRAC;

    // fill ray indice vector with inrementing values
    std::vector<int> allIndices(rays.size());
    for (size_t i = 0; i < rays.size(); ++i) {
        allIndices[i] = i;
    }

    // reserve nodes (efficient avoids reallocation)
    _nodes.reserve(rays.size() * 2);

    // push root place holds
    _nodes.push_back(KdNode{sceneBounds});

    // recursively build
    buildRecursive(0, allIndices, 0);
}

int KdTree::buildRecursive(int nodeIndex, std::vector<int>& rayIndices, int depth) {
        AABB& bounds = _nodes[nodeIndex].bounds;
        // subdivision criteria using per node bounding box
        Vec3 diagonal = bounds.max - bounds.min;
        if (!(diagonal.norm() < _rMinDiag || depth > D_MAX || rayIndices.size() < C_MIN)) { // if SHOULDN'T SPLIT, it's a leaf
            _nodes[nodeIndex].rayIndices = std::move(rayIndices);
            return nodeIndex;
        }

        // not a leaf, have to store and continue recursively!
        // first find the axis split
        int axis = bounds.longestAxis();
        int splitPos = bounds.centroid(axis);

        // set the end and beginning bounds equally on this axis
        AABB leftBounds = bounds;
        AABB rightBounds = bounds;
        leftBounds.max[axis] = splitPos;
        rightBounds.min[axis] = splitPos;

        _nodes[nodeIndex].splitAxis = axis;
        _nodes[nodeIndex].splitPos = splitPos;

        // per ray, calculate if it hits the box, and sort rays left or right to it
        std::vector<int> leftRays, rightRays;
        for (int rayIndex: rayIndices) {
            const Ray& ray = (*_rays)[rayIndex];
            float tEnter = 0.0f;
            float tExit = 0.0f;
            if (leftBounds.intersect(ray, tEnter, tExit)) {
                leftRays.push_back(rayIndex);
            }
            if (rightBounds.intersect(ray, tEnter, tExit)) {
                rightRays.push_back(rayIndex);
            }
        }
        rayIndices.clear();

        // left subtree
        _nodes.push_back(KdNode{leftBounds});
        buildRecursive(_nodes.size(), leftRays, depth + 1);

        //right subtree
        _nodes.push_back(KdNode{rightBounds});
        buildRecursive(_nodes.size(), rightRays, depth + 1);

        return nodeIndex;
}