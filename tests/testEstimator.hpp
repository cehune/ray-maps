#pragma once;
#include "estimator.hpp"
#include "ray.hpp"
#include "helpers.hpp"

void testEstimatorUniformFlat() {
    // parallel rays along z, uniform grid over xy, equal flux
    std::vector<Ray> rays;
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            rays.push_back({
                {float(i), float(j), 0.f},
                {0.f, 0.f, 1.f},
                0.f, 20.f,
                1.f  // unit flux
            });

    KdTree tree;
    tree.build(rays, AABB{{0,0,0},{10,10,20}});

    IrradianceEstimator est;
    Vec3 n{0.f, 0.f, -1.f};

    // query several interior points — should all give similar irradiance
    Vec3 point1 = Vec3(5.f, 5.f, 10.f);
    Vec3 point2 = Vec3(4.f, 4.f, 10.f);
    Vec3 point3 = Vec3(6.f, 6.f, 10.f);
    float e1 = est.estimate(tree, point1, n, 16, 100.f);
    float e2 = est.estimate(tree, point2, n, 16, 100.f);
    float e3 = est.estimate(tree, point3, n, 16, 100.f);

    //printf("e1=%.4f e2=%.4f e3=%.4f\n", e1, e2, e3);

    // not checking exact value — checking consistency
    assert_true(std::abs(e1 - e2) < e1 * 0.1f, "Estimator should be spatially consistent");
    assert_true(std::abs(e1 - e3) < e1 * 0.1f, "Estimator should be spatially consistent");
}

// Test 2: Dense cluster near query point gives higher irradiance than sparse uniform
// This tests the Epanechnikov kernel weighting — rays near the centre get w ≈ 1,
// rays near R get w ≈ 0. A concentrated cluster should dominate.
void testEstimatorConcentrationEffect() {
    std::vector<Ray> rays;

    // Group A: 16 rays tightly packed near (5, 5) — will land close to query point
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            rays.push_back({
                {4.8f + i * 0.1f, 4.8f + j * 0.1f, 0.f},
                {0.f, 0.f, 1.f},
                0.f, 20.f,
                1.f
            });

    // Group B: 16 rays spread wide around the perimeter
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            rays.push_back({
                {1.f + i * 2.5f, 1.f + j * 2.5f, 0.f},
                {0.f, 0.f, 1.f},
                0.f, 20.f,
                1.f
            });

    KdTree treeDense, treeSparse;
    AABB bounds{{0,0,0},{10,10,20}};

    // Dense tree: only the cluster
    std::vector<Ray> denseOnly(rays.begin(), rays.begin() + 16);
    treeDense.build(denseOnly, bounds);

    // Sparse tree: only the perimeter
    std::vector<Ray> sparseOnly(rays.begin() + 16, rays.end());
    treeSparse.build(sparseOnly, bounds);

    IrradianceEstimator est;
    Vec3 x{5.f, 5.f, 10.f};
    Vec3 n{0.f, 0.f, -1.f};

    float eDense  = est.estimate(treeDense,  x, n, 16, 10.f);
    float eSparse = est.estimate(treeSparse, x, n, 16, 10.f);

    // printf("eDense=%.4f eSparse=%.4f\n", eDense, eSparse);

    // Same ray count, same flux, same K — but dense cluster sits near kernel
    // centre where w ≈ 1, sparse sits near R where w ≈ 0. Dense wins.
    assert_true(eDense > eSparse * 2.0f,
        "Dense cluster near query should give significantly higher irradiance than sparse perimeter");
}


// Test 3: Flux scaling — doubling flux on every ray should double irradiance
// This is a linearity check on the weighted sum before normalisation.
void testEstimatorFluxScaling() {
    auto makeRays = [](float flux) {
        std::vector<Ray> rays;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                rays.push_back({
                    {float(i), float(j), 0.f},
                    {0.f, 0.f, 1.f},
                    0.f, 20.f,
                    flux
                });
        return rays;
    };

    auto r1 = makeRays(1.f);
    auto r2 = makeRays(2.f);

    KdTree t1, t2;
    AABB bounds{{0,0,0},{5,5,20}};
    t1.build(r1, bounds);
    t2.build(r2, bounds);

    IrradianceEstimator est;
    Vec3 x{2.f, 2.f, 10.f};
    Vec3 n{0.f, 0.f, -1.f};

    float e1 = est.estimate(t1, x, n, 16, 10.f);
    float e2 = est.estimate(t2, x, n, 16, 10.f);

    assert_true(std::abs(e2 - 2.f * e1) < e1 * 0.01f,
        "Doubling flux on all rays should exactly double irradiance");
}

// Test 4: The two-pass max-R correctness test.
// knn sorts by max(metricA, metricB), NOT by disc distance alone.
// A ray with large metricB but small metricA can rank last in knn output
// yet have the largest disc distance — so it must set discMaxR2, not the
// last element of candidates. This test constructs exactly that situation.
//
// Geometry:
//   Query point x = (0, 0, 5), normal n = (0, 0, -1)
//   Ray A: origin=(0, 0, 0), dir=(0, 0, 1) — pierces disc at (0,0,5), dist2=0 on disc,
//           but origin is far from x so metricB is large → ranked last by knn
//   Ray B: origin=(3, 0, 0), dir=(0, 0, 1) — pierces disc at (3,0,5), dist2=9 on disc,
//           origin is closer in 3D to x so metricB is smaller → ranked first by knn
//
// If the estimator naively used candidates.back().dist2 as R^2 instead of
// scanning all disc intersections, it would use R^2=9 from ray B — which
// happens to be correct here only by accident. We instead construct a case
// where the last candidate by knn order has a SMALLER disc dist2 than an
// earlier one, to prove the scan is necessary.
void testEstimatorTwoPassMaxR() {
    std::vector<Ray> rays;

    // Ray A: passes directly through x on the disc (disc dist2 = 0),
    // but origin is at z=0 far below — metricB from x=(0,0,5) to
    // closest point on segment is large, so knn ranks it low.
    // Actually we want Ray A to have LARGE disc dist and be ranked low
    // by knn. Let's place it far off-centre on the disc.
    //
    // Ray A: origin=(-4, 0, 0), dir=(0.6, 0, 0.8) (normalised roughly)
    // Pierces z=5 plane at t such that 0.8t = 5 → t = 6.25
    // Disc hit: x_hit = -4 + 0.6*6.25 = -0.25, z=5 → dist2 from (0,0,5) ≈ 0.0625
    // Closest point on segment to (0,0,5): somewhere mid-segment → metricB small
    // → knn ranks this first (low combined metric)

    // Ray B: origin=(0, 0, 0), dir=(0, 0, 1)
    // Pierces z=5 plane at t=5 → disc hit=(0,0,5), dist2=0 — but wait,
    // we want B to have LARGE disc dist2. Let's rethink.
    //
    // Cleaner construction:
    //   Ray A: hits disc far from x (large disc dist2), but origin is close → small metricB
    //          → knn sorts it FIRST (small combined metric max(A,B))
    //   Ray B: hits disc close to x (small disc dist2), origin also close
    //          → knn sorts it SECOND
    // If estimator uses candidates[last].dist2 instead of scanning, it picks
    // Ray B's small disc dist2 as R, incorrectly shrinking the kernel.

    // Ray A: origin=(0,0,0), dir normalised toward (4,0,5) → hits (4,0,5) on z=5 plane
    {
        Vec3 dir = Vec3{4.f, 0.f, 5.f};
        float len = std::sqrt(dir.dot(dir));
        dir = dir * (1.f / len);
        // t_max just past intersection: t at z=5 is 5/dir.z
        float t_hit = 5.f / dir.z;
        rays.push_back({Vec3{0.f, 0.f, 0.f}, dir, 0.f, t_hit + 1.f, 1.f});
    }

    // Ray B: origin=(0,0,0), dir=(0,0,1) → hits (0,0,5) exactly, disc dist2=0
    rays.push_back({Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 0.f, 1.f}, 0.f, 10.f, 1.f});

    KdTree tree;
    tree.build(rays, AABB{{-1,-1,0},{5,1,6}});

    IrradianceEstimator est;
    Vec3 x{0.f, 0.f, 5.f};
    Vec3 n{0.f, 0.f, -1.f};

    // K=2 forces both rays to be candidates
    float e = est.estimate(tree, x, n, 2, 20.f);

    // Ray B hits exactly at x → dist2=0 → weight=(1 - 0/R^2)=1
    // Ray A hits at (4,0,5)  → dist2=16 → weight=(1 - 16/16)=0
    // So irradiance = (1*1 + 1*0) / (pi * 16) = 1/(16*pi)
    float expected = 1.f / (16.f * float(M_PI));

    // printf("e=%.6f expected=%.6f\n", e, expected);

    assert_true(std::abs(e - expected) < expected * 0.01f,
        "Two-pass max-R must scan all disc intersections, not rely on knn ordering");
}

int runEstimatorTests() {
    testEstimatorUniformFlat();
    testEstimatorFluxScaling();
    testEstimatorConcentrationEffect();
    testEstimatorTwoPassMaxR();
    std::cout << "All Estimator Tests Passed ^-^ \n";
    return 0;
}
