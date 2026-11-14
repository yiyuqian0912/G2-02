#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace r3 {
using Point=std::array<double,3>;
// Labels are certificates for CLOSED dyadic boxes: FREE=0, SOLID=1,
// INTERNAL=2, UNKNOWN=-1. UNKNOWN is never interpreted as physical geometry.
class CertifiedSpace {
 std::vector<std::int8_t> labels_;
 std::vector<std::int32_t> children_;
public:
 CertifiedSpace(std::vector<std::int8_t>,std::vector<std::int32_t>);
 int contains(const Point&)const;
 int visibility(const Point&,const Point&,std::uint64_t&)const;
 int certificate(const Point&,const Point&,const Point&,std::uint64_t&)const;
};

// Each basis entry is an outward enclosure of cos/sin of an EXACT dyadic
// affine angle, prepared with the existing Arb implementation. No libm trig,
// interpolation, float32, or sampled-geometry classification is used here.
class CertifiedGrid {
 unsigned n_;
 std::vector<double> coefficients_,basis_;
 std::array<double,2> bias_;
 int constant_;
public:
 CertifiedGrid(unsigned,std::vector<double>,std::vector<double>,std::array<double,2>,int);
 int contains(const Point&)const;
};
}
