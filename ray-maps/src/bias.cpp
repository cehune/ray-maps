#include <fstream>
#include "kdtree.hpp"
#include "estimator.hpp"

struct QueryPoint {
    Vec3 position;
    Vec3 normal;
};

std::vector<Ray> generateAngledRays(int N, float flux) {
    std::vector<Ray> rays;

    // rays come from x=-60, z=10, angled toward +x and -z
    // they hit z=0 at various x positions
    // some hit x <= 0 (on surface), some hit x > 0 (off surface)
    // but ALL pass through the z=0 plane

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            // origin: spread in x from -60 to +10, y from -4 to 4, at z=10
            float ox = -60.f + 70.f * i / (N-1);
            float oy = -4.f  + 8.f  * j / (N-1);

            // direction: angled so ray hits z=0 at x = ox + 10 (shifts right by 10)
            // dir = normalize(10, 0, -10) = (1/sqrt(2), 0, -1/sqrt(2))
            Vec3 dir = Vec3{1.f, 0.f, -1.f}.normalized();

            rays.push_back({
                Vec3{ox, oy, 10.f},
                dir,
                0.f, 20.f,
                Vec3{flux, flux, flux}
            });
        }

    return rays;
}

std::vector<Vec3> generatePhotonImpactsAgnled(const std::vector<Ray>& rays) {
    std::vector<Vec3> impacts;
    for (const Ray& ray : rays) {
        // find t where ray hits z=0
        // z = origin.z + dir.z * t = 0 → t = -origin.z / dir.z
        float t = -ray.origin.z / ray.dir.z;
        if (t < ray.t_min || t > ray.t_max) continue;
        Vec3 hit = ray.origin + ray.dir * t;
        // only impacts on the surface
        if (hit.x <= 0.f)
            impacts.push_back(hit);
    }
    return impacts;
}

std::vector<Vec3> generatePhotonImpacts(const std::vector<Ray>& rays) {
    std::vector<Vec3> impacts;
    for (const Ray& ray : rays) {
        // ray travels -z from z=10, hits z=0 at t=10
        // only keep impacts that land on the surface x <= 0
        Vec3 hit = ray.origin + ray.dir * 10.f;  // t=10 → z=0
        if (hit.x <= 0.f)
            impacts.push_back(hit);
    }
    return impacts;
}

float photonMapEstimate(
    const std::vector<Vec3>& impacts,
    const Vec3& x,
    int K,
    float maxDist,
    float flux
) {
    // brute force KNN on impact points — simple for comparison
    std::vector<float> dists;
    for (const Vec3& p : impacts) {
        float d2 = (p - x).norm2();
        if (d2 <= maxDist * maxDist)
            dists.push_back(d2);
    }
    std::sort(dists.begin(), dists.end());
    if ((int)dists.size() > K) dists.resize(K);
    if (dists.empty()) return 0.f;

    float R2 = dists.back();
    if (R2 < 1e-8f) return 0.f;

    // flat kernel — no Epanechnikov for simplicity
    float sum = dists.size() * flux;
    return sum / (float(M_PI) * R2);
}

std::vector<Ray> generateParallelRays(int N, float flux) {
    std::vector<Ray> rays;

    // Plane A: x in [-50, 0], rays traveling -z from z=10 to z=-10
    // rays cover x in [-50, 0], y in [-25, 25]
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float px = -50.f + 50.f * i / (N-1);  // x in [-50, 0]
            float py = -4.f + 8.f * j / (N-1);  // y in [-25, 25]
            rays.push_back({
                Vec3{px, py, 10.f},
                Vec3{0.f, 0.f, -1.f},
                0.f, 20.f,
                Vec3{flux, flux, flux}
            });
        }

    return rays;
}

std::vector<QueryPoint> generateQueryPath(int steps) {
    std::vector<QueryPoint> path;

    // query path sweeps x from -40 to 0 at z=0, y=0
    // normal points up +z to receive downward traveling rays
    for (int i = 0; i < steps; ++i) {
        float t  = float(i) / (steps - 1);
        float px = -40.f + 40.f * t;  // x in [-40, 0]
        path.push_back({Vec3{px, 0.f, 0.f}, Vec3{0.f, 0.f, 1.f}});
    }

    return path;
}

void reproduceFigure4a(const std::string& outPath) {
    int   N       = 50;
    float flux    = 1.f;
    int   K       = 100;
    float maxDist = 4.f;
    int   steps   = 50;

    auto rays = generateAngledRays(N, flux);

    KdTree tree;
    // bounds must contain all ray origins (z=10) and endpoints (z=-10)
    // x in [-50, 0], y in [-25, 25], z in [-10, 10]
    // AABB bounds{{-51.f, -26.f, -11.f}, {1.f, 26.f, 11.f}}; non angle
    AABB bounds{{-61.f, -5.f, -1.f}, {11.f, 5.f, 11.f}};
    tree.build(rays, bounds);

    IrradianceEstimator est;
    auto path = generateQueryPath(steps);

    std::ofstream csv(outPath);

    auto impacts = generatePhotonImpactsAgnled(rays);

    csv << "step,px,raymap,photonmap\n";
    for (int i = 0; i < (int)path.size(); ++i) {
        Vec3  e_rm = est.estimate(tree, path[i].position, path[i].normal, K, maxDist);
        float e_pm = photonMapEstimate(impacts, path[i].position, K, maxDist, flux);
        csv << i << "," << path[i].position.x << "," << e_rm.x << "," << e_pm << "\n";
    

        // debug every 10 steps
        if (i % 10 == 0) {
            auto dbg = tree.knn(path[i].position, path[i].normal, K, maxDist);
            printf("step=%d pos=(%.2f,%.2f,%.2f) candidates=%zu irradiance=%.4f\n",
                i,
                path[i].position.x,
                path[i].position.y,
                path[i].position.z,
                dbg.size(),
                e_rm.x);
        }
    }

    // sanity check first 5 rays
    for (int i = 0; i < 5; ++i)
        printf("ray %d origin=(%.2f,%.2f,%.2f) dir=(%.2f,%.2f,%.2f)\n",
            i,
            rays[i].origin.x, rays[i].origin.y, rays[i].origin.z,
            rays[i].dir.x,    rays[i].dir.y,    rays[i].dir.z);

    csv.close();
    printf("wrote %s\n", outPath.c_str());
}

int main() {
    reproduceFigure4a("figure4a.csv");
    return 0;
}