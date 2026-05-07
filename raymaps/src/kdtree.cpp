#include "kdtree.hpp"
#include <cassert>

void KdTree::build(const std::vector<Ray>& rays, const AABB& sceneBounds) {

    assert(_nodes.empty() && "Tree not reset between builds");
    assert(rays.size() > 0 && "Empty ray set");
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


// returns nullopt if ray misses hemisphere or disc
std::optional<DiscIntersection> KdTree::metricAFull(const Ray& ray, const Vec3& x, 
    const Vec3& n) const {
        // same logic as metric a but returning the point as well.
    float planeDir = ray.dir.dot(n);
    if (std::abs(planeDir) < 1e-6f) return std::nullopt;
    if (planeDir >= 0.f)            return std::nullopt;
    
    float t = (x - ray.origin).dot(n) / planeDir;
    if (t < ray.t_min || t > ray.t_max) return std::nullopt;

    Vec3 point = ray.origin + ray.dir * t;
    float dist2 = (x - point).norm2();
    return DiscIntersection{dist2, point};
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

    // had a bug where rays were evaluated twice, need to track with set
    std::unordered_set<int> visited;


    nodeQueue.push({0.f, 0});

    // printf("[knn init] K=%d maxDist=%.4f maxDist2=%.4f\n", K, maxDist, maxDist2);
    // printf("[knn init] nodeQueue size=%zu\n", nodeQueue.size());
    // printf("[knn init] root node isLeaf=%d rayIndices=%zu\n", 
    //     _nodes[0].isLeaf(), _nodes[0].rayIndices.size());

    while (!nodeQueue.empty()) {
        auto [boxDist2, nodeIdx] = nodeQueue.top();
        nodeQueue.pop();

        //printf("[traverse] nodeIdx=%d boxDist2=%.4f currentRadius2=%.4f candidates=%zu\n",
        //nodeIdx, boxDist2, 
        // ((int)candidates.size() == K) ? std::min(maxDist2, candidates.top().dist2) : maxDist2,
        // candidates.size());

        // recompute tightest radius fresh each iteration —
        // candidates may have been updated since this node was pushed
        float currentRadius2 = ((int)candidates.size() == K)
            ? std::min(maxDist2, candidates.top().dist2)
            : maxDist2;

        // priority queue guarantees nodes come out in order of increasing
        // box distance — once closest unvisited box is farther than our
        // current K-th candidate, no future node can contain a closer ray
        if (boxDist2 > currentRadius2) break;

        const KdNode& node = _nodes[nodeIdx];

        if (node.isLeaf()) {
            for (int rayIndex : node.rayIndices) {
                if (visited.count(rayIndex)) continue;
                visited.insert(rayIndex);

                const Ray& ray = (*_rays)[rayIndex];

                // metric II.(a): squared dist from x to tangent plane intersection
                float distA = metricA(ray, x, n);

                // never pierces disc or wrong hemisphere
                if (distA == FLT_MAX) continue;

                // metric II.(b): squared dist from x to closest point on segment
                float distB = metricB(ray, x);

                // conservative search metric — max ensures we don't miss rays
                // that are close in one metric but far in the other
                float dist2 = std::max(distA, distB);

                // recompute radius here too — previous candidates in this
                // same leaf may have tightened it
                float leafRadius2 = ((int)candidates.size() == K)
                    ? std::min(maxDist2, candidates.top().dist2)
                    : maxDist2;

                if (dist2 > leafRadius2) continue;

                candidates.push({dist2, rayIndex});

                // evict the furthest candidate if we exceeded K
                if ((int)candidates.size() > K) candidates.pop();
            }
        } else {
            // push children pruned against current tightest radius
            if (node.leftChild >= 0) {
                float d2 = _nodes[node.leftChild].bounds.sqDistToPoint(x);
                if (d2 <= currentRadius2) nodeQueue.push({d2, node.leftChild});
            }
            if (node.rightChild >= 0) {
                float d2 = _nodes[node.rightChild].bounds.sqDistToPoint(x);
                if (d2 <= currentRadius2) nodeQueue.push({d2, node.rightChild});
            }
        }
    }

    std::priority_queue<RayCandidate> temp = candidates;
    // printf("[knn result] candidate count=%zu\n", candidates.size());
    // while (!temp.empty()) {
    //     printf("  rayIdx=%d dist2=%.4f\n", temp.top().rayIndex, temp.top().dist2);
    //     temp.pop();
    // }

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