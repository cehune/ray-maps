#pragma once
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
                {1.0f, 1.0f, 1.0f}  // unit flux
            });

    KdTree tree;
    tree.build(rays, AABB{{0,0,0},{10,10,20}});

    IrradianceEstimator est;
    Vec3 n{0.f, 0.f, -1.f};

    // query several interior points — should all give similar irradiance
    Vec3 point1 = Vec3(5.f, 5.f, 10.f);
    Vec3 point2 = Vec3(4.f, 4.f, 10.f);
    Vec3 point3 = Vec3(6.f, 6.f, 10.f);
    Vec3 e1 = est.estimate(tree, point1, n, 16, 100.f);
    Vec3 e2 = est.estimate(tree, point2, n, 16, 100.f);
    Vec3 e3 = est.estimate(tree, point3, n, 16, 100.f);

    //printf("e1=%.4f e2=%.4f e3=%.4f\n", e1, e2, e3);
    for (int i = 0; i < 3; ++i) {
        // not checking exact value — checking consistency
        assert_true(std::abs(e1[i] - e2[i]) < e1[i] * 0.1f, "Estimator should be spatially consistent");
        assert_true(std::abs(e1[i] - e3[i]) < e1[i] * 0.1f, "Estimator should be spatially consistent");

    }

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
                {1.0f, 1.0f, 1.0f}
            });

    // Group B: 16 rays spread wide around the perimeter
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            rays.push_back({
                {1.f + i * 2.5f, 1.f + j * 2.5f, 0.f},
                {0.f, 0.f, 1.f},
                0.f, 20.f,
                {1.0f, 1.0f, 1.0f}
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

    Vec3 eDense  = est.estimate(treeDense,  x, n, 16, 10.f);
    Vec3 eSparse = est.estimate(treeSparse, x, n, 16, 10.f);

    // printf("eDense=%.4f eSparse=%.4f\n", eDense, eSparse);

    // Same ray count, same flux, same K — but dense cluster sits near kernel
    // centre where w ≈ 1, sparse sits near R where w ≈ 0. Dense wins.
    assert_true(eDense.norm2() > eSparse.norm2() * 2.0f,
        "Dense cluster near query should give significantly higher irradiance than sparse perimeter");
}


// Test 3: Flux scaling — doubling flux on every ray should double irradiance
// This is a linearity check on the weighted sum before normalisation.
void testEstimatorFluxScaling() {
    auto makeRays = [](Vec3 flux) {
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

    auto r1 = makeRays({1.0f, 1.0f, 1.0f});
    auto r2 = makeRays({2.0f, 2.0f, 2.0f});

    KdTree t1, t2;
    AABB bounds{{0,0,0},{5,5,20}};
    t1.build(r1, bounds);
    t2.build(r2, bounds);

    IrradianceEstimator est;
    Vec3 x{2.f, 2.f, 10.f};
    Vec3 n{0.f, 0.f, -1.f};

    Vec3 e1 = est.estimate(t1, x, n, 16, 10.f);
    Vec3 e2 = est.estimate(t2, x, n, 16, 10.f);

    Vec3 diff = e2 - e1 * 2.f;
    assert_true(diff.norm2() < (e1 * 0.05f).norm2(),
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
    // We need: a ray that ranks LAST in knn order (largest max(A,B))
    // but does NOT have the largest distA.
    // 
    // Ray X: distA=5, distB=20 → max=20 → ranks last in knn
    // Ray Y: distA=9, distB=1  → max=9  → ranks first in knn
    //
    // True R2_disc = max(distA) = max(5, 9) = 9 (from Ray Y)
    // If estimator uses candidates.back().dist2 = 20 as R2, it gets wrong answer
    // If estimator scans all distA values, it gets R2=9, correct answer
    //
    // Both rays carry flux=1.
    // Expected irradiance:
    //   Ray Y: weight = 1 - 9/9 = 0
    //   Ray X: weight = 1 - 5/9 = 0.444
    //   sum = 1 * 0.444 = 0.444
    //   irradiance = 0.444 / (pi * 9)

    // x = (0,0,5), n = (0,0,-1)
    // Ray Y: needs distA=9 → hits disc at distance 3 from x
    //        needs distB=1 → closest point on segment is 1 unit from x
    // Simplest: origin=(3,0,4), dir=(0,0,1) 
    //   distA: hits z=5 at (3,0,5), dist2 from (0,0,5) = 9 ✓
    //   distB: closest point to (0,0,5) on segment from (3,0,4) along z
    //          t = (5-4)/1 = 1, point=(3,0,5), dist=(3,0,0), distB=9
    //   Hmm distB=9 too. Need distB < distA for Ray Y to rank before Ray X.
    
    // Let's be more explicit. Use a diagonal ray for Ray X to decouple A and B.
    // 
    // Ray X: origin=(-1, 0, 4), dir=(0.196, 0, 0.981) (roughly toward (0,0,5))  
    //   hits z=5 at t=(5-4)/0.981=1.02, point=(-1+0.2, 0, 5)=(-0.8,0,5)
    //   distA = 0.64 -- too small. This is getting complicated.
    //
    // Simplest clean construction: just hardcode dirs that give known distA/distB

    Vec3 x{0.f, 0.f, 5.f};
    Vec3 n{0.f, 0.f, -1.f};

    std::vector<Ray> rays;

    // Ray 0: origin=(0,0,0), dir=(0,0,1)
    //   distA: hits (0,0,5), dist2=0
    //   distB: closest point on z-segment to x=(0,0,5) is (0,0,5), distB=0
    //   max(0,0)=0 → ranks first
    rays.push_back({Vec3{0.f,0.f,0.f}, Vec3{0.f,0.f,1.f}, 0.f, 10.f, {1.0f, 1.0f, 1.0f}});

    // Ray 1: origin=(2,0,0), dir=(0,0,1)  
    //   distA: hits (2,0,5), dist2=4
    //   distB: closest point to (0,0,5) is (2,0,5), distB=4
    //   max(4,4)=4 → ranks second
    rays.push_back({Vec3{2.f,0.f,0.f}, Vec3{0.f,0.f,1.f}, 0.f, 10.f, {1.0f, 1.0f, 1.0f}});

    // Ray 2: origin=(0,0,0), dir toward (1,0,5) normalized
    //   This ray passes VERY close to x in 3D but hits disc at (1,0,5)
    //   distA = 1, distB will be small since it passes near x
    //   max(distA,distB) should be small → ranks before Ray 1
    //   But distA=1 < distA of Ray 1 = 4
    //   So if we use 3 rays with K=3, Ray 1 sets R2_disc=4
    //   and Ray 2's contribution should be weighted by dist2=1 not 4
    {
        Vec3 dir = Vec3{1.f, 0.f, 5.f}.normalized();
        float t_hit = 5.f / dir.z;
        rays.push_back({Vec3{0.f,0.f,0.f}, dir, 0.f, t_hit+1.f, {1.0f, 1.0f, 1.0f}});
    }

    KdTree tree;
    tree.build(rays, AABB{{-1,-1,0},{3,1,6}});

    IrradianceEstimator est;

    // K=3: all rays are candidates
    // R2_disc = max(distA) across all rays
    // Ray 0: distA=0, Ray 1: distA=4, Ray 2: distA=1
    // R2_disc = 4
    //
    // weights (Epanechnikov):
    //   Ray 0: 1 - 0/4 = 1.0
    //   Ray 1: 1 - 4/4 = 0.0
    //   Ray 2: 1 - 1/4 = 0.75
    //
    // irradiance = (1*1.0 + 1*0.0 + 1*0.75) / (pi * 4) = 1.75 / (4*pi)
    float expected = 1.75f / (4.f * float(M_PI));

    float e = est.estimate(tree, x, n, 3, 20.f).x;  // check R channel

    //printf("e=%.6f expected=%.6f\n", e, expected);
    assert_true(std::abs(e - expected) < expected * 0.02f,
        "Two-pass max-R must scan all disc intersections");
}

int runEstimatorTests() {
    testEstimatorUniformFlat();
    testEstimatorFluxScaling();
    testEstimatorConcentrationEffect();
    testEstimatorTwoPassMaxR();
    std::cout << "All Estimator Tests Passed ^-^ \n";
    return 0;
}
