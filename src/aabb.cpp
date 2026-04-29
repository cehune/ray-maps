#include "aabb.hpp"

bool AABB::intersect(const Ray& ray, float& t_enter, float& t_exit) const {
    t_enter = ray.t_min;
    t_exit = ray.t_max;

    // loop through each axis
    /* doing &min/max.x then indexing makes it agnostic to axes because
    x,y,z are laid out contiguously in memory */
    for (int i = 0; i < 3; ++i) {
        float axesOrigin = ray.origin[i]; // origin of ray for a given axes
        float axesDirection = ray.dir[i];
        // min and max per axes
        float mn = min[i]; 
        float mx = max[i];

        float invD = 1.0f / axesDirection;
        if (axesDirection == 0.0f) { // if exactly on boundary we might get
            if (axesOrigin < mn || axesOrigin > mx) return false;
            continue;
        }

        float t0 = (mn - axesOrigin) * invD; // scalar traveled for that ray component
        float t1 = (mx - axesOrigin) * invD; // max scalar traveled for that ray component
        if (invD < 0.0f) std::swap(t0, t1);

        t_enter = std::max(t_enter, t0);
        t_exit = std::min(t_exit, t1);
        if (t_enter > t_exit) return false;
    }
    return true;
}

// dist squared from a point to closest point on the boxes surface
float AABB::sqDistToPoint(const Vec3& p) const {
    float dist = 0.0f;

    for (int i = 0; i < 3; ++i) {
        float pointAxesComponent = p[i]; // same memory indexing trick
        float mn = min[i];
        float mx = max[i];

        if (pointAxesComponent < mn)  {
            float d = pointAxesComponent-mn;
            dist += d*d;
        }
        else if (pointAxesComponent > mx) {
            float d = pointAxesComponent - mx;
            dist += d*d;
        };
    }
    return dist;
}

int AABB::longestAxis() const {
    // axes labeled 0, 1, 2 for x, y, z respectively
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    
    if (dx >= dy && dx >= dz) return 0;
    if (dy >= dz) return 1;
    return 2;
}

float AABB::centroid(int axis) const {
    return min[axis] + 0.5f * (max[axis] - min[axis]);
}
