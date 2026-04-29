#pragma once
#include <cassert>
#include <iostream>

void assert_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "Assertion failed: " << msg << "\n";
        std::abort();
    }
}