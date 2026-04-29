// Axis Aligned Bounding Box

#pragma once
#include "ray.hpp"

struct AABB {
    Vec3 min, max;

    bool intersect(const Ray& ray, float& t_enter, float& t_exit) const;
    float sqDistToPoint(const Vec3& p) const;
    float centroid(int axis) const;
    int longestAxis() const;
};