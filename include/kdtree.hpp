#pragma once
#include "ray.hpp"
#include "aabb.hpp"
#include <vector>
#include <memory>
#include <queue>
#include <algorithm>
#include <cstdio>
#include <string>
#include <float.h>

struct KdNode {
    AABB bounds;

    // Interior node fields
    int   splitAxis = -1;   // -1 means leaf
    float splitPos  =  0.f;
    int   leftChild = -1;   // index into nodes array
    int   rightChild= -1;

    // Leaf node fields
    std::vector<int> rayIndices;  // indices into ray pool

    bool isLeaf() const { return splitAxis == -1; }
};

struct RayCandidate {
    float dist2;  // squared distance metric II.(a)
    int   rayIdx;
    bool operator>(const RayCandidate& o) const { return dist2 > o.dist2; }
};

class KdTree {
public:
    // Build constants from the paper(Cmin=32, Dmax=30, Rmin=0.1% scene diagonal)
    static constexpr int   C_MIN  = 32; // min rays per leaf (any less and end)
    static constexpr int   D_MAX  = 30; // max tree depth
    static constexpr float R_MIN_FRAC = 0.001f; // min spacial extent aabb for a node

    void build(const std::vector<Ray>& rays, const AABB& sceneBounds);
    std::vector<KdNode>& nodes() { return _nodes; }
    void print(int nodeIdx, int depth) const;
    void validate(int nodeIdx) const;
    // TODO: Add the actual KNN logic

    // metric II.(a): distance from x to ray's tangent plane intersection
    // returns squared distance, or FLT_MAX if ray doesn't pierce hemisphere
    float metricA(const Ray& ray, const Vec3& x, const Vec3& n) const;

    // metric II.(b): squared distance from x to closest point on segment
    float metricB(const Ray& ray, const Vec3& x) const;

private:
    std::vector<KdNode> _nodes;
    const std::vector<Ray>* _rays = nullptr;
    float _rMinDiag = 0.f;

    int buildRecursive(
        int nodeIndex, std::vector<int>& rayIndices, int depth
    );
};