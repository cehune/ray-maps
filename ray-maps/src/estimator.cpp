#include "estimator.hpp"
#include <cmath>
#include <cfloat>

Vec3 IrradianceEstimator::estimate(KdTree& tree, const Vec3& x, const Vec3& n, int K, float maxDist) const {
    std::vector<RayCandidate> candidates = tree.knn(x, n, K, maxDist);
    if (candidates.empty()) return {0.f, 0.f, 0.f};

    // Populate disc intersections.
    // knn sorts by max(distA, distB) so we cannot use candidate order
    // for the disc radius — must scan all distA values explicitly.
    std::vector<DiscIntersection> intersections;
    intersections.reserve(candidates.size());

    for (auto& candidate : candidates) {
        const Ray& ray = tree.ray(candidate.rayIndex);
        auto isect = tree.metricAFull(ray, x, n);
        if (!isect) continue;
        intersections.push_back({isect->dist2, isect->point, candidate.rayIndex});
    }

    if (intersections.empty()) return {0.f, 0.f, 0.f};

    // Sort by disc distance ascending — K-th entry sets R2_disc
    std::sort(intersections.begin(), intersections.end(),
        [](const DiscIntersection& a, const DiscIntersection& b){
            return a.dist2 < b.dist2;
        });

    // Keep only K tightest disc intersections
    if ((int)intersections.size() > K)
        intersections.erase(intersections.begin() + K, intersections.end());

    // R2 is the K-th smallest distA — geometrically meaningful disc radius
    float R2 = intersections.back().dist2;
    if (R2 < 1e-8f) return {0.f, 0.f, 0.f};

    // Epanechnikov kernel: w(d) = 1 - d^2/R^2
    // Normalized over disc area pi*R^2
    Vec3 irradianceSum{0.f, 0.f, 0.f};
    for (auto& intersection : intersections) {
        const Ray& ray = tree.ray(intersection.rayIndex);
        float weight = 1.f - (intersection.dist2 / R2);
        if (weight < 0.f) weight = 0.f;
        irradianceSum = irradianceSum + ray.flux * weight;
    }

    return irradianceSum * (1.f / (float(M_PI) * R2));
}