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

int testBuildKDTree() {
    testDepthLimit();
    testLeafConditionSmallInput();
    testRayDuplicationOverBothSplits();
    testSplittingOccursOnLargeInput();
    testTreeIndicesValid();

    std::cout << "All KD Building Tests Passed ^-^ \n";
    return 0;
}