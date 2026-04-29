#pragma once
#include <cassert>
#include <cfloat>
#include "aabb.hpp"
#include "ray.hpp"
#include <iostream>

void testCenteredRayHitsBoundingBox() {
    AABB box{{0,0,0}, {1,1,1}};
    Ray ray{{-1, 0.5f, 0.5f}, {1, 0, 0}, 0.0f, FLT_MAX, 0.0f};

    float t0, t1;
    bool hit = box.intersect(ray, t0, t1);

    assert(hit);
    assert(t0 >= 1.0f - 1e-5f && t0 <= 1.0f + 1e-5f);
}

void testRayMisses() {
    AABB box{{0,0,0}, {1,1,1}};
    Ray ray{{-1, 2.0f, 0.5f}, {1, 0, 0}, 0.0f, FLT_MAX, 0.0f};

    float t0, t1;
    bool hit = box.intersect(ray, t0, t1);

    assert(!hit);
}

void testParallelRayInsideSlabShouldHit() {
    AABB box{{0,0,0}, {1,1,1}};
    Ray ray{{0.5f, 0.5f, -1}, {0, 0, 1}, 0.0f, FLT_MAX, 0.0f};

    float t0, t1;
    bool hit = box.intersect(ray, t0, t1);

    assert(hit);
}

void testParallelRayOutsideSlabShouldMiss() {
    AABB box{{0,0,0}, {1,1,1}};
    Ray ray{{1.5f, 0.5f, -1}, {0, 0, 1}, 0.0f, FLT_MAX, 0.0f};

    float t0, t1;
    bool hit = box.intersect(ray, t0, t1);

    assert(!hit);
}

void testRayInsideBoxDoesntHit() {
    AABB box{{0,0,0}, {1,1,1}};
    Ray ray{{0.5f, 0.5f, 0.5f}, {1, 0, 0}, 0.0f, FLT_MAX, 0.0f};

    float t0, t1;
    bool hit = box.intersect(ray, t0, t1);

    assert(hit);
    assert(t0 == 0.0f); // since t_min = 0
}

void rayPointingAwayFromBoxShouldMiss() {
    AABB box{{0,0,0}, {1,1,1}};
    Ray ray{{-1, 0.5f, 0.5f}, {-1, 0, 0}, 0.0f, FLT_MAX, 0.0f};

    float t0, t1;
    bool hit = box.intersect(ray, t0, t1);

    assert(!hit);
}

void testSquareDistToPointInside() {
    AABB box{{0,0,0}, {1,1,1}};
    Vec3 p{0.5f, 0.5f, 0.5f};

    float d2 = box.sqDistToPoint(p);
    assert(d2 == 0.0f);
}

void testSquareDistToPointOutside() {
    AABB box{{0,0,0}, {1,1,1}};
    Vec3 p{2.0f, 0.5f, 0.5f};

    float d2 = box.sqDistToPoint(p);
    assert(std::abs(d2 - 1.0f) < 1e-5f); // (2 - 1)^2
}

void testSquareDistToPointDiag() {
    AABB box{{0,0,0}, {1,1,1}};
    Vec3 p{2.0f, 3.0f, 0.5f};

    float d2 = box.sqDistToPoint(p);
    // (2-1)^2 + (3-1)^2 = 1 + 4 = 5
    assert(std::abs(d2 - 5.0f) < 1e-5f);
}

void testLongestAxis() {
    AABB box{{0,0,0}, {2,1,1}};
    assert(box.longestAxis() == 0);

    AABB box2{{0,0,0}, {1,3,1}};
    assert(box2.longestAxis() == 1);

    AABB box3{{0,0,0}, {1,1,5}};
    assert(box3.longestAxis() == 2);
}

void testCentroid() {
    AABB box{{0,0,0}, {2,4,6}};
    assert(box.centroid(0) == 1.0f);
    assert(box.centroid(1) == 2.0f);
    assert(box.centroid(2) == 3.0f);
}

void testGrazesTopFace() {
    AABB box{{0,0,0}, {1,1,1}};
    Ray ray{{-1, 1.0f, 0.5f}, {1, 0, 0}, 0.0f, FLT_MAX, 0.0f};

    float t0, t1;
    bool hit = box.intersect(ray, t0, t1);

    assert(hit); // grazing top face
}

void runAABBTests() {
    testCenteredRayHitsBoundingBox();
    testCentroid();
    testGrazesTopFace();
    testLongestAxis();
    testParallelRayInsideSlabShouldHit();
    testParallelRayOutsideSlabShouldMiss();
    testRayInsideBoxDoesntHit();
    testRayMisses();
    testSquareDistToPointDiag();
    testSquareDistToPointInside();
    testSquareDistToPointOutside();

    std::cout << "All AABB tests passed! ^-^\n";
}