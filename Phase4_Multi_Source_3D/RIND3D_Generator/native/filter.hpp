#pragma once
#include <array>
#include <vector>
namespace r3 {
struct Primitive {int kind;std::array<double,3> center,half;};
class Filter {
 std::vector<Primitive> pieces_;
public:
 explicit Filter(std::vector<Primitive> pieces):pieces_(std::move(pieces)){}
 int contains(const std::array<double,3>& p)const;
 int visibility(const std::array<double,3>& source,const std::array<double,3>& p)const;
 std::array<int,2> certificate(const std::array<double,3>& source,const std::array<double,3>& lo,const std::array<double,3>& hi)const;
};
}
