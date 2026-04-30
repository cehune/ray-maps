#include "estimator.hpp"
#include <cmath>
#include <cfloat>

Vec3 IrradianceEstimator::estimate(KdTree& tree, Vec3& x, Vec3& n, int K, float maxDist) const {
    std::vector<RayCandidate> candidates = tree.knn(x, n, K, maxDist);
    if (candidates.empty()) return {0.0f, 0.0f, 0.0f}; // nothing

    // Now the estimator relies on a distance R for normalization. This should be the furthest
    // on the disc. 
    // Naively we can assume that the ordering from the knn is correct such that the furthest
    // distance away is the last. However, knn for the paper is sorted based on the max 
    // of metrics A AND B!!! Therefore, we have no gaurantee that the ordering is correct based
    // on the objective distance on the disc. Therefore, we have to have two passes through the 
    // candidates to find the true max distance on the disc. 

    std::vector<DiscIntersection> intersections;
    intersections.reserve(candidates.size());

    float discMaxR2 = 0.0f;

    for (auto& candidate: candidates) {
        const Ray& ray = tree.ray(candidate.rayIndex);
        auto intersection = tree.metricAFull(ray, x, n);
        if (!intersection) continue; // should never really happen just a safety check
        intersections.push_back(*intersection);
        discMaxR2 = std::max(discMaxR2, intersection->dist2);
    }   


    if (discMaxR2 < 1e-8f) return {0.0f, 0.0f, 0.0f};  // degenerate

    // Epanechnikov kernel: w(d) = 1 - d^2/R^2
    // normalized over disc area pi*R^2

    Vec3 irradianceSum = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < intersections.size(); ++i) {
        // can reuse the i index for both candidates and intersections
        const Ray& ray = tree.ray(candidates[i].rayIndex);
        float weight = 1.0f - (intersections[i].dist2 / discMaxR2);
        if (weight < 0.0f) weight = 0.0f;
        irradianceSum += ray.flux * weight;
    }

    return irradianceSum / (M_PI * discMaxR2);
}