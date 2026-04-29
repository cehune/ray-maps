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
    //validate(0);
}

int KdTree::buildRecursive(int nodeIndex, std::vector<int>& rayIndices, int depth) {
        // printf("[buildRecursive] nodeIdx=%d depth=%d rays=%zu\n",
        //     nodeIndex, depth, rayIndices.size());
            
        AABB& bounds = _nodes[nodeIndex].bounds;
        // subdivision criteria using per node bounding box
        Vec3 diagonal = bounds.max - bounds.min;
        if (diagonal.norm() < _rMinDiag || depth >= D_MAX || rayIndices.size() < C_MIN) { // shouldn't split, its a leaf
            _nodes[nodeIndex].rayIndices = std::move(rayIndices);
            return nodeIndex;
        }

        // set the end and beginning bounds equally on this axis
        AABB leftBounds = bounds;
        AABB rightBounds = bounds;

        //find the axis split
        int axis = bounds.longestAxis();
        float splitPos = bounds.centroid(axis);

        if (splitPos <= bounds.min[axis] || splitPos >= bounds.max[axis]) {
            _nodes[nodeIndex].rayIndices = std::move(rayIndices);
            return nodeIndex;
        }

        leftBounds.max[axis] = splitPos;
        rightBounds.min[axis] = splitPos;

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

        // helpful debugging
        // printf("[buildRecursive] nodeIdx=%d depth=%d rays=%zu leftRays=%zu rightRays=%zu axis=%d splitPos=%.3f bounds=(%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)\n",
        // nodeIndex, depth, rayIndices.size(),
        // leftRays.size(), rightRays.size(), axis, splitPos,
        // bounds.min.x, bounds.min.y, bounds.min.z,
        // bounds.max.x, bounds.max.y, bounds.max.z;
        
        // avoid making any empty leaves
        if (leftRays.empty() || rightRays.empty()) {
            _nodes[nodeIndex].rayIndices = std::move(rayIndices);
            return nodeIndex;
        }

        // now we commit to being an interior node.
        _nodes[nodeIndex].splitAxis = axis;
        _nodes[nodeIndex].splitPos = splitPos;
        rayIndices.clear(); // just free some mem

        int leftIndex = (int)_nodes.size();
        _nodes.push_back(KdNode{leftBounds});
        int rightIndex = (int)_nodes.size();
        _nodes.push_back(KdNode{rightBounds});

        // set before any push backs so node index doesn't go stale from reallocation
        _nodes[nodeIndex].leftChild  = leftIndex;
        _nodes[nodeIndex].rightChild = rightIndex;

        buildRecursive(leftIndex,  leftRays,  depth + 1);
        buildRecursive(rightIndex, rightRays, depth + 1);

        return nodeIndex;
}

void KdTree::print(int nodeIdx, int depth) const {
    if (nodeIdx < 0 || nodeIdx >= (int)_nodes.size()) {
        printf("%*s[INVALID NODE %d]\n", depth*2, "", nodeIdx);
        return;
    }

    const KdNode& node = _nodes[nodeIdx];
    std::string indent(depth * 2, ' ');

    if (node.isLeaf()) {
        printf("%s[LEAF %d] rays=%zu bounds=(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)\n",
            indent.c_str(), nodeIdx,
            node.rayIndices.size(),
            node.bounds.min.x, node.bounds.min.y, node.bounds.min.z,
            node.bounds.max.x, node.bounds.max.y, node.bounds.max.z);
    } else {
        printf("%s[INTERIOR %d] axis=%d splitPos=%.2f left=%d right=%d\n",
            indent.c_str(), nodeIdx,
            node.splitAxis, node.splitPos,
            node.leftChild, node.rightChild);
        print(node.leftChild,  depth + 1);
        print(node.rightChild, depth + 1);
    }
}

void KdTree::validate(int nodeIdx) const {
    if (nodeIdx < 0) return;
    const KdNode& node = _nodes[nodeIdx];

    if (node.isLeaf()) {
        // A leaf must have rays if it was worth creating
        if (node.rayIndices.empty()) {
            printf("[INVALID] Leaf %d has no rays!\n", nodeIdx);
        }
    } else {
        // An interior node must have valid children
        if (node.leftChild  < 0) printf("[INVALID] Interior %d has no left child\n",  nodeIdx);
        if (node.rightChild < 0) printf("[INVALID] Interior %d has no right child\n", nodeIdx);
        // And must not have rays stored on it
        if (!node.rayIndices.empty()) {
            printf("[INVALID] Interior %d has %zu rays — should be empty\n",
                nodeIdx, node.rayIndices.size());
        }
        validate(node.leftChild);
        validate(node.rightChild);
    }
}