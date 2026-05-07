#include "testAABB.hpp"
#include "testBuildKDTree.hpp"
#include "testEstimator.hpp"

int main() {
    runAABBTests();
    runKdTreeTests();
    runEstimatorTests();
}