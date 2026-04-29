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

float KdTree::metricA(const Ray& ray, const Vec3& x, const Vec3& n) const {
    // x is the point, n is the plane normal, ray is our onset ray
    // recall that normal gives equation for a plane
    // n.a(x - xo) + n.b(y-yo) + n.c(z-zo) = 0, where xo, yo, zo are from ray origin + t dir
    // so t is the only unknown, we can directly find the unknwon vector
    
    float planeDir = ray.dir.dot(n);

    // Ray parallel to tangent plane — never pierces disc
    if (std::abs(planeDir) < 1e-6f) return FLT_MAX;

    // Only consider rays coming from the correct hemisphere
    // (incoming rays have dir.dot(n) < 0 for a surface normal pointing up)
    if (planeDir >= 0.f) return FLT_MAX;
    
    float planeSurfacePoint = (x - ray.origin).dot(n);
    float t = planeSurfacePoint / planeDir;

    // Intersection must be within the segment's extent
    if (t < ray.t_min || t > ray.t_max) return FLT_MAX;
    
    Vec3 dist = x - (ray.origin + ray.dir * t);
    return dist.norm2();
}

float KdTree::metricB(const Ray& ray, const Vec3& x) const {
    Vec3 originToSurface = x - ray.origin;
    // projection gives closest point - from there it is perpindicular to the target point
    float t = originToSurface.dot(ray.dir);
    t = std::max(ray.t_min, std::min(ray.t_max, t)); // clamp in bounds

    Vec3 dist = x - (ray.origin + ray.dir * t);
    return dist.norm2();
}

std::vector<RayCandidate> KdTree::knn(const Vec3& x, const Vec3& n, int K, float maxDist) const {
    /* x is the target point on the surface, n is the plane defined tangent to the surface on 
    that point, K is the number of ray candidates we want, maxDist is that maximum distance away
    they can cross the tangent plane. 
    
    returns a list of Ray Candidates that meet the required conditions
    */

    float maxDist2 = maxDist * maxDist;

    // max heap by distance to drop furthest candidates
    std::priority_queue<RayCandidate> candidates;

    // min heap by distance, enter into the closest nodes next
    std::priority_queue<NodeEntry,
                    std::vector<NodeEntry>,
                    std::greater<NodeEntry>> nodeQueue;

    nodeQueue.push({0.f, 0});

    while (!nodeQueue.empty()) {
        NodeEntry nodeEntry = nodeQueue.top();
        nodeQueue.pop();

        // unified pruning radius: either maxDist or current worst candidate
        float currentRadius2 = maxDist2;
        if ((int)candidates.size() == K) {
            currentRadius2 = std::min(currentRadius2, candidates.top().dist2);
        }

        // just ensure the current nodes 
        if ((int)candidates.size() == K && (nodeEntry.dist2 > currentRadius2)) break;

        const KdNode& node = _nodes[nodeEntry.nodeIndex];

        if (node.isLeaf()) {
            for (int rayIndex: node.rayIndices) {
                const Ray& ray = (*_rays)[rayIndex];
                float distA = metricA(ray, x, n);
                float distB = metricB(ray, x);
                // never pierces disc or is not in the range
                if (distA == FLT_MAX) continue;

                // conservative but gaurantees we don't lose real points on some edge cases
                float dist2 = std::max(distA, distB);
                if (dist2 > currentRadius2) continue; // dont include if bad

                candidates.push({dist2, rayIndex});
                if ((int)candidates.size() > K) candidates.pop(); // eliminate furthest
            }
        } else {
            // check if left and right children are valid
            if (node.leftChild >= 0) {
                float dist2 = _nodes[node.leftChild].bounds.sqDistToPoint(x);
                if (dist2 <= currentRadius2) nodeQueue.push({dist2, node.leftChild});
            }
            if (node.rightChild >= 0) {
                float dist2 = _nodes[node.rightChild].bounds.sqDistToPoint(x);
                if (dist2 <= currentRadius2) nodeQueue.push({dist2, node.rightChild});
            }
        }
    }

    std::vector<RayCandidate> result;
    result.reserve(candidates.size());
    while(!candidates.empty()) {
        result.push_back(candidates.top());
        candidates.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
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