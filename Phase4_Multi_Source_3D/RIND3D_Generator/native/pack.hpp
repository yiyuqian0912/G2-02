#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace r3 {
constexpr std::size_t HEADER=256, RECORD=32, RAW=32768;
std::uint32_t crc(const std::string& s);
struct Encoded { std::uint8_t codec; std::string payload; std::uint32_t checksum,ones; };
Encoded encode(const std::string& raw,int level);
class Writer {
 std::string path_,partial_,identity_,policy_; std::FILE* file_=nullptr;
 std::array<std::uint32_t,3> shape_; std::vector<bool> done_;
 std::uint64_t tail_; int level_; bool finished_=false;
 void header(bool complete);
public:
 Writer(std::string path,std::array<std::uint32_t,3> shape,std::string identity,
        std::string policy,bool resume,int level);
 ~Writer();
 std::vector<std::size_t> committed() const;
 void write(std::size_t index,const std::string& raw);
 void checkpoint();
 void finish();
};
}
