#pragma once
#include "kdtree.hpp"
#include "helpers.hpp"
#include <iostream>

void testBuildBasic() {
    AABB bounds{{0,0,0}, {1,1,1}};

    std::vector<Ray> rays = {
        {{-1,0.5f,0.5f}, {1,0,0}, 0.0f, 10.0f, 0.0f},
        {{0.5f,-1,0.5f}, {0,1,0}, 0.0f, 10.0f, 0.0f}
    };

    KdTree tree;
    tree.build(rays, bounds);

    // root should exist
    assert_true(!tree.nodes().empty(), "Tree should not be empty");

    // root should be either leaf or interior
    assert_true(tree.nodes()[0].splitAxis != -2, "Root valid");
}

void testLeafConditionSmallInput() {
    AABB bounds{{0,0,0}, {1,1,1}};

    std::vector<Ray> rays;
    for (int i = 0; i < 10; ++i) { // < C_MIN (32)
        rays.push_back({{0,0,0}, {1,0,0}, 0.0f, 10.0f, 0.0f});
    }

    KdTree tree;
    tree.build(rays, bounds);

    const KdNode& root = tree.nodes()[0];

    assert_true(root.isLeaf(), "Should be leaf when below C_MIN");
    assert_true(root.rayIndices.size() == rays.size(), "All rays in leaf");
}

void testSplittingOccursOnLargeInput() {
    AABB bounds{{0,0,0}, {10,10,10}};

    std::vector<Ray> rays;
    for (int i = 0; i < 100; ++i) {
        rays.push_back({{-1.0f, (float)i, 5.0f}, {1,0,0}, 0.0f, 20.0f, 0.0f});
    }

    KdTree tree;
    tree.build(rays, bounds);

    const KdNode& root = tree.nodes()[0];

    assert_true(!root.isLeaf(), "Root should split");
    assert_true(root.leftChild != -1, "Left child exists");
    assert_true(root.rightChild != -1, "Right child exists");
}

void testRayDuplicationOverBothSplits() {
    AABB bounds{{0,0,0}, {10,10,10}};

    std::vector<Ray> rays;

    // Two groups of rays at different Y heights
    // so splits along Y will separate them
    for (int i = 0; i < 30; ++i)
        rays.push_back({{0.f, 2.f, 5.f}, {1,0,0}, 0.f, 10.f, 1.f}); // y=2
    for (int i = 0; i < 30; ++i)
        rays.push_back({{0.f, 8.f, 5.f}, {1,0,0}, 0.f, 10.f, 1.f}); // y=8

    KdTree tree;
    tree.build(rays, bounds);

    // Find the first interior node that splits along Y
    // Both children should have rays
    bool foundSplit = false;
    for (int i = 0; i < (int)tree.nodes().size(); ++i) {
        const KdNode& node = tree.nodes()[i];
        if (!node.isLeaf() && node.splitAxis == 1) { // Y split
            foundSplit = true;
            const KdNode& left  = tree.nodes()[node.leftChild];
            const KdNode& right = tree.nodes()[node.rightChild];
            // collect all rays in left/right subtrees
            assert_true(left.rayIndices.size(),  
                        "Left of Y-split should have rays");
            assert_true(right.rayIndices.size(), 
                        "Right of Y-split should have rays");
            break;
        }
    }
    assert_true(foundSplit, "Should have found a Y-axis split");
}

void testTreeIndicesValid() {
    AABB bounds{{0,0,0}, {5,5,5}};

    std::vector<Ray> rays(100, {{0,0,0}, {1,0,0}, 0.0f, 10.0f, 0.0f});

    KdTree tree;
    tree.build(rays, bounds);

    for (size_t i = 0; i < tree.nodes().size(); ++i) {
        const KdNode& node = tree.nodes()[i];

        if (!node.isLeaf()) {
            assert_true(node.leftChild >= 0 && node.leftChild < (int)tree.nodes().size(),
                        "Valid left child index");

            assert_true(node.rightChild >= 0 && node.rightChild < (int)tree.nodes().size(),
                        "Valid right child index");
        }
    }
}

void testDepthLimit() {
    AABB bounds{{0,0,0}, {100,100,100}};

    std::vector<Ray> rays(1000, {{0,0,0}, {1,0,0}, 0.0f, 100.0f, 0.0f});

    KdTree tree;
    tree.build(rays, bounds);

    // If we reached here, no infinite recursion happened
    assert_true(true, "Tree built without infinite recursion");
}

void testMetricA_ValidHit() {
    KdTree tree;

    Ray ray{{0,0,0}, {0,0,1}, 0.0f, 10.0f, 0.0f};
    Vec3 x{0,0,5};
    Vec3 n{0,0,-1}; // facing the ray

    float result = tree.metricA(ray, x, n);

    // Should hit exactly at x → distance = 0
    assert_true(std::abs(result) < 1e-6f,
                "MetricA should be zero for perfect intersection");
}

void testMetricA_ParallelRay() {
    KdTree tree;

    Ray ray{{0,0,0}, {1,0,0}, 0.0f, 10.0f, 0.0f};
    Vec3 x{0,0,5};
    Vec3 n{0,0,1}; // perpendicular to ray

    float result = tree.metricA(ray, x, n);

    assert_true(result == FLT_MAX,
                "Parallel ray should return FLT_MAX");
}

void testMetricA_WrongHemisphere() {
    KdTree tree;

    Ray ray{{0,0,0}, {0,0,1}, 0.0f, 10.0f, 0.0f};
    Vec3 x{0,0,5};
    Vec3 n{0,0,1}; // same direction → rejected

    float result = tree.metricA(ray, x, n);

    assert_true(result == FLT_MAX,
                "Ray in wrong hemisphere should be rejected");
}

void testMetricA_OutOfBoundsT() {
    KdTree tree;

    Ray ray{{0,0,0}, {0,0,1}, 0.0f, 2.0f, 0.0f};
    Vec3 x{0,0,5};  // beyond t_max
    Vec3 n{0,0,-1};

    float result = tree.metricA(ray, x, n);

    assert_true(result == FLT_MAX,
                "Intersection outside t range should be rejected");
}

void testMetricB_PointOnRay() {
    KdTree tree;

    Ray ray{{0,0,0}, {0,0,1}, 0.0f, 10.0f, 0.0f};
    Vec3 x{0,0,5};

    float result = tree.metricB(ray, x);

    assert_true(std::abs(result) < 1e-6f,
                "MetricB should be zero for point on ray");
}

void testMetricB_PerpendicularDistance() {
    KdTree tree;

    Ray ray{{0,0,0}, {0,0,1}, 0.0f, 10.0f, 0.0f};
    Vec3 x{1,0,5}; // 1 unit away from ray

    float result = tree.metricB(ray, x);

    assert_true(result > 0.9f && result < 1.1f,
                "MetricB should be ~1 for perpendicular distance");
}

void testKNNMatchesBruteForce() {
    AABB bounds{{0,0,0}, {10,10,10}};

    std::vector<Ray> rays;
    rays.clear();

    for (int i = 0; i <  50; ++i) {
        rays.push_back({
            {float(i % 10), float(i / 10), 0.f},
            {0,0,1},
            0.f, 20.f, 0.f
        });
    }

    KdTree tree;
    tree.build(rays, bounds);

    Vec3 x{5,5,5};
    Vec3 n{0,0,-1};

    int K = 10;
    float maxDist = 100.f;

    // for (int i = 0; i < 20; ++i) {
    // printf("ray %d: origin=(%.1f,%.1f,%.1f) dir=(%.1f,%.1f,%.1f) tmin=%.1f tmax=%.1f\n",
    //     i,
    //     rays[i].origin.x, rays[i].origin.y, rays[i].origin.z,
    //     rays[i].dir.x, rays[i].dir.y, rays[i].dir.z,
    //     rays[i].t_min, rays[i].t_max);
    // }

    // for (int i = 0; i < (int)rays.size(); ++i) {
    //     float a = tree.metricA(rays[i], x, n);
    //     float b = tree.metricB(rays[i], x);
    //     printf("[BRUTE] rayIndex=%d distA=%.4f distB=%.4f\n", i, a, b);
    // }

    auto result = tree.knn(x, n, K, maxDist);

    // brute force
    std::vector<std::pair<float,int>> brute;
    for (int i = 0; i < (int)rays.size(); ++i) {
        float a = tree.metricA(rays[i], x, n);
        float b = tree.metricB(rays[i], x);
        if (a == FLT_MAX) continue;

        float d = std::max(a,b);
        brute.push_back({d, i});
    }

    std::sort(brute.begin(), brute.end());
    brute.resize(K);

    assert_true(result.size() == brute.size(), "Same number of neighbors");

    for (int i = 0; i < (int)result.size(); ++i) {
        //std::cout << "i: "<< i << "  knn dist: " << result[i].dist2 << "  brute dist: " << brute[i].first << std::endl;
        assert_true(std::abs(result[i].dist2 - brute[i].first) < 1e-4f,
                 "Distances should match brute force");
    }
}

void testKNNRespectsMaxDistance() {
    AABB bounds{{0,0,0}, {10,10,10}};

    std::vector<Ray> rays;
    // near rays
    for (int i = 0; i < 10; ++i)
        rays.push_back({{0,0,float(i)}, {0,0,1}, 0.f, 10.f, 0.f});
    // far rays
    for (int i = 0; i < 10; ++i)
        rays.push_back({{0,0,float(i+50)}, {0,0,1}, 0.f, 100.f, 0.f});

    KdTree tree;
    tree.build(rays, bounds);

    Vec3 x{0,0,5};
    Vec3 n{0,0,-1};

    auto result = tree.knn(x, n, 20, 5.0f);

    for (auto& c : result) {
        assert_true(c.dist2 <= 25.0f,
                    "All results must respect maxDist");
    }
}

void testKNNLessThanKResults() {
    AABB bounds{{0,0,0}, {10,10,10}};

    std::vector<Ray> rays(5, {{0,0,0}, {0,0,1}, 0.f, 10.f, 0.f});

    KdTree tree;
    tree.build(rays, bounds);

    Vec3 x{0,0,5};
    Vec3 n{0,0,-1};

    auto result = tree.knn(x, n, 10, 100.f);

    assert_true(result.size() == 5,
                "Should return all available rays if < K");
}

void testKNNDuplicateDistances() {
    AABB bounds{{0,0,0}, {10,10,10}};

    std::vector<Ray> rays;

    // identical rays
    for (int i = 0; i < 20; ++i)
        rays.push_back({{0,0,0}, {0,0,1}, 0.f, 10.f, 0.f});

    KdTree tree;
    tree.build(rays, bounds);

    Vec3 x{0,0,5};
    Vec3 n{0,0,-1};

    auto result = tree.knn(x, n, 10, 100.f);

    assert_true(result.size() == 10,
                "Should handle duplicate distances correctly");
}

int testBuildKDTree() {
    testDepthLimit();
    testLeafConditionSmallInput();
    testRayDuplicationOverBothSplits();
    testSplittingOccursOnLargeInput();
    testTreeIndicesValid();

    testMetricA_OutOfBoundsT();
    testMetricA_ParallelRay();
    testMetricA_ValidHit();
    testMetricA_WrongHemisphere();
    testMetricB_PerpendicularDistance();
    testMetricB_PointOnRay();

    testKNNDuplicateDistances();
    testKNNLessThanKResults();
    testKNNMatchesBruteForce();
    testKNNRespectsMaxDistance();

    std::cout << "All KD Building Tests Passed ^-^ \n";
    return 0;
}