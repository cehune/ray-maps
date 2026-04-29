#pragma once
#include <cmath>
#include <array>

class Vec3 {
public:
    float x, y, z;

    Vec3(float x, float y, float z): x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float t) const { return {x*t, y*t, z*t}; }

    float operator[](int i) const { return (&x)[i]; }
    float& operator[](int i) { return (&x)[i]; }

    float dot (const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    float norm2() const {return dot(*this); }
    float norm() const { return std::sqrt(norm2()); } // unit length
    Vec3 normalized() const { 
        float n = norm();
        return {x/n, y/n, z/n};
    }
};

struct Ray {
    Vec3  origin;
    Vec3  dir; // TODO add debug assert, it has to be normalized
    float t_min;
    float t_max;
    float flux; // TODO switch to rgb
};