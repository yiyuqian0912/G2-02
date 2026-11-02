// Isolated exact stored-bit analytics and certified convex research kernels.
// Existing production predicates and historical experiment binaries are intact.
#include "pack.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gmpxx.h>
#include <list>
#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <zstd.h>
namespace py = pybind11;
using P = std::array<double, 3>;
using Q = mpq_class;
using VQ = std::array<Q, 3>;
using Rows = std::vector<std::vector<std::vector<std::string>>>;
using C =
    std::array<std::uint64_t, 2>; // lit FREE, solid; shadow = volume - both
static const auto populations = [] {
  std::array<unsigned char, 256> t{};
  for (unsigned i = 0; i < 256; ++i)
    t[i] = std::popcount(i);
  return t;
}();
// Same IEEE CRC32 as RINDPack; slicing by eight avoids rereading/decompressing
// cost being dominated by a byte-at-a-time checksum. Tested against zlib.
std::uint32_t packed_crc(const std::string &s) {
  static const auto tables = [] {
    std::array<std::array<std::uint32_t, 256>, 8> t{};
    for (unsigned i = 0; i < 256; ++i) {
      auto c = i;
      for (int k = 0; k < 8; ++k)
        c = (c >> 1) ^ ((c & 1) ? 0xedb88320U : 0);
      t[0][i] = c;
    }
    for (int k = 1; k < 8; ++k)
      for (int i = 0; i < 256; ++i)
        t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 255];
    return t;
  }();
  std::uint32_t c = 0xffffffffU;
  std::size_t i = 0;
  for (; i + 8 <= s.size(); i += 8) {
    auto p = reinterpret_cast<const unsigned char *>(s.data() + i);
    c ^= std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    c = tables[7][c & 255] ^ tables[6][(c >> 8) & 255] ^
        tables[5][(c >> 16) & 255] ^ tables[4][c >> 24] ^ tables[3][p[4]] ^
        tables[2][p[5]] ^ tables[1][p[6]] ^ tables[0][p[7]];
  }
  for (; i < s.size(); ++i)
    c = (c >> 8) ^ tables[0][(c ^ static_cast<unsigned char>(s[i])) & 255];
  return c ^ 0xffffffffU;
}
struct Record {
  std::uint64_t offset;
  std::uint32_t length, codec, decoded, stored, ones;
};
struct Reader {
  std::ifstream file;
  std::vector<Record> records;
  std::size_t limit;
  std::list<std::pair<std::size_t, std::string>> cache;
  std::unordered_map<std::size_t, decltype(cache.begin())> positions;
  std::uint64_t hits = 0, misses = 0, bytes = 0;
  Reader(std::string path,
         const std::vector<std::array<std::uint64_t, 6>> &index,
         std::size_t cap)
      : file(path, std::ios::binary), limit(std::max(std::size_t(1), cap)) {
    if (!file)
      throw std::runtime_error("Cannot open packed field");
    static const std::array<std::uint32_t, 2> crc{
        packed_crc(std::string(32768, 0)),
        packed_crc(std::string(32768, char(255)))};
    for (auto &r : index) {
      if (r[1] > 32768 || r[2] > 3 || r[5] > 262144)
        throw std::invalid_argument("Invalid packed record");
      if (r[2] < 2 &&
          (r[1] || r[4] || r[3] != crc[r[2]] || r[5] != (r[2] ? 262144 : 0)))
        throw std::runtime_error("Uniform record CRC/population mismatch");
      records.push_back({r[0], std::uint32_t(r[1]), std::uint32_t(r[2]),
                         std::uint32_t(r[3]), std::uint32_t(r[4]),
                         std::uint32_t(r[5])});
    }
  }
  const std::string &block(std::size_t i) {
    if (i >= records.size())
      throw std::out_of_range("Packed brick index");
    if (records[i].codec < 2) {
      static const std::array<std::string, 2> raw{
          std::string(32768, 0), std::string(32768, char(255))};
      ++hits;
      return raw[records[i].codec];
    }
    auto found = positions.find(i);
    if (found != positions.end()) {
      ++hits;
      cache.splice(cache.begin(), cache, found->second);
      return found->second->second;
    }
    if (i >= records.size())
      throw std::out_of_range("Packed brick index");
    ++misses;
    const auto &r = records[i];
    std::string stored(r.length, '\0');
    if (r.length) {
      file.seekg(r.offset);
      file.read(stored.data(), r.length);
      if (!file)
        throw std::runtime_error("Truncated packed payload");
      bytes += r.length;
    }
    if (packed_crc(stored) != r.stored)
      throw std::runtime_error("Stored packed CRC mismatch");
    std::string raw(32768, r.codec == 1 ? char(255) : 0);
    if (r.codec == 2) {
      if (r.length != 32768)
        throw std::runtime_error("Invalid raw brick");
      raw = std::move(stored);
    }
    if (r.codec == 3) {
      auto n =
          ZSTD_decompress(raw.data(), raw.size(), stored.data(), stored.size());
      if (ZSTD_isError(n) || n != 32768)
        throw std::runtime_error("Invalid compressed brick");
    }
    if (packed_crc(raw) != r.decoded)
      throw std::runtime_error("Decoded packed CRC mismatch");
    std::uint32_t ones = 0;
    for (unsigned char b : raw)
      ones += populations[b];
    if (ones != r.ones)
      throw std::runtime_error("Packed population count mismatch");
    cache.emplace_front(i, std::move(raw));
    positions[i] = cache.begin();
    if (cache.size() > limit) {
      positions.erase(cache.back().first);
      cache.pop_back();
    }
    return cache.front().second;
  }
};
std::uint64_t pop_bytes(const unsigned char *a, const unsigned char *m,
                        int first, int last, int lo, int hi, bool solid) {
  std::uint64_t result = 0;
  for (int b = first; b < last; ++b) {
    unsigned char mask = 255;
    if (b == first)
      mask &= static_cast<unsigned char>(255u << (lo % 8));
    if (b == last - 1 && hi % 8)
      mask &= static_cast<unsigned char>((1u << (hi % 8)) - 1);
    unsigned char value =
        solid ? m[b] : (a[b] & static_cast<unsigned char>(~m[b]));
    result += std::popcount(static_cast<unsigned char>(value & mask));
  }
  return result;
}
struct PackedFields {
  std::array<int, 3> shape; // XYZ
  std::array<int, 3> grid;
  Reader y, m;
  std::mutex mutex;
  std::uint64_t rectangles = 0, uniform = 0, scanned_bytes = 0;
  PackedFields(std::string yp, std::string mp, std::array<int, 3> xyz,
               const std::vector<std::array<std::uint64_t, 6>> &yi,
               const std::vector<std::array<std::uint64_t, 6>> &mi, int cap)
      : shape(xyz), y(yp, yi, cap), m(mp, mi, cap) {
    for (int a = 0; a < 3; ++a) {
      if (shape[a] < 1 || shape[a] > 1024)
        throw std::invalid_argument("Invalid packed shape");
      grid[a] = (shape[a] + 63) / 64;
    }
    if (yi.size() != std::size_t(grid[0] * grid[1] * grid[2]) ||
        mi.size() != yi.size())
      throw std::invalid_argument("Packed record count mismatch");
  }
  C piece(std::size_t i, std::array<int, 3> lo, std::array<int, 3> hi) {
    const auto &a = y.records[i];
    const auto &b = m.records[i];
    std::uint64_t volume = 1;
    for (int k = 0; k < 3; ++k)
      volume *= hi[k] - lo[k];
    if (a.codec < 2 && b.codec < 2) {
      // CRC-validate constant summaries too, exactly as the existing bit
      // reader.
      static const std::array<std::uint32_t, 2> crc{
          r3::crc(std::string(32768, 0)),
          r3::crc(std::string(32768, char(255)))};
      if (a.decoded != crc[a.codec] || b.decoded != crc[b.codec] || a.stored ||
          b.stored)
        throw std::runtime_error("Uniform CRC mismatch");
      if (a.codec && b.codec)
        throw std::runtime_error("Lit solid stored voxel");
      ++uniform;
      return {a.codec ? volume : 0, b.codec ? volume : 0};
    }
    const auto &ya = y.block(i);
    const auto &ma = m.block(i);
    C out{};
    for (int z = lo[2]; z < hi[2]; ++z)
      for (int v = lo[1]; v < hi[1]; ++v) {
        auto aa = reinterpret_cast<const unsigned char *>(ya.data()) +
                  (z * 64 + v) * 8;
        auto mm = reinterpret_cast<const unsigned char *>(ma.data()) +
                  (z * 64 + v) * 8;
        int first = lo[0] / 8, last = (hi[0] + 7) / 8;
        for (int x = first; x < last; ++x) {
          unsigned mask = 255;
          if (x == first)
            mask &= (255u << (lo[0] % 8)) & 255;
          if (x == last - 1 && hi[0] % 8)
            mask &= (1u << (hi[0] % 8)) - 1;
          if (aa[x] & mm[x] & mask)
            throw std::runtime_error("Lit solid stored voxel");
          out[0] += populations[unsigned(aa[x] & ~mm[x]) & mask];
          out[1] += populations[unsigned(mm[x]) & mask];
        }
        scanned_bytes += 2 * (last - first);
      }
    return out;
  }
  C rect(std::array<int, 3> lo, std::array<int, 3> hi) {
    ++rectangles;
    for (int a = 0; a < 3; ++a)
      if (lo[a] < 0 || hi[a] < lo[a] || hi[a] > shape[a])
        throw std::invalid_argument("Count rectangle outside field");
    if (lo[0] == hi[0] || lo[1] == hi[1] || lo[2] == hi[2])
      return {};
    C total{};
    for (int bz = lo[2] / 64; bz <= (hi[2] - 1) / 64; ++bz)
      for (int by = lo[1] / 64; by <= (hi[1] - 1) / 64; ++by)
        for (int bx = lo[0] / 64; bx <= (hi[0] - 1) / 64; ++bx) {
          std::array<int, 3> b{bx, by, bz}, a, c;
          for (int k = 0; k < 3; ++k) {
            a[k] = std::max(lo[k] - b[k] * 64, 0);
            c[k] = std::min(hi[k] - b[k] * 64, 64);
          }
          C v = piece((bz * grid[1] + by) * grid[0] + bx, a, c);
          for (int k = 0; k < 2; ++k)
            total[k] += v[k];
        }
    return total;
  }
  py::array_t<std::uint64_t>
  boxes(py::array_t<int, py::array::c_style> values) {
    if (values.ndim() != 2 || values.shape(1) != 6)
      throw std::invalid_argument("[N,6] lower/upper boxes required");
    py::array_t<std::uint64_t> out({values.shape(0), py::ssize_t(2)});
    {
      py::gil_scoped_release release;
      std::lock_guard guard(mutex);
      for (py::ssize_t i = 0; i < values.shape(0); ++i) {
        const int *p = values.data() + 6 * i;
        C v = rect({p[0], p[1], p[2]}, {p[3], p[4], p[5]});
        out.mutable_data()[2 * i] = v[0];
        out.mutable_data()[2 * i + 1] = v[1];
      }
    }
    return out;
  }
  py::array_t<std::uint16_t> leaves() {
    std::array<int, 3> dims;
    for (int a = 0; a < 3; ++a)
      dims[a] = (shape[a] + 15) / 16;
    py::array_t<std::uint16_t> out({dims[2], dims[1], dims[0], 2});
    {
      py::gil_scoped_release release;
      std::lock_guard guard(mutex);
      for (int bz = 0; bz < grid[2]; ++bz)
        for (int by = 0; by < grid[1]; ++by)
          for (int bx = 0; bx < grid[0]; ++bx) {
            for (int z = 0; z < 4; ++z)
              for (int v = 0; v < 4; ++v)
                for (int x = 0; x < 4; ++x) {
                  std::array<int, 3> block{bx * 4 + x, by * 4 + v, bz * 4 + z};
                  if (block[0] >= dims[0] || block[1] >= dims[1] ||
                      block[2] >= dims[2])
                    continue;
                  std::array<int, 3> lo{x * 16, v * 16, z * 16}, hi;
                  for (int a = 0; a < 3; ++a)
                    hi[a] = std::min(lo[a] + 16, shape[a] - (a == 0   ? bx
                                                             : a == 1 ? by
                                                                      : bz) *
                                                                64);
                  C c = piece((bz * grid[1] + by) * grid[0] + bx, lo, hi);
                  std::size_t i =
                      ((block[2] * dims[1] + block[1]) * dims[0] + block[0]) *
                      2;
                  out.mutable_data()[i] = c[0];
                  out.mutable_data()[i + 1] = c[1];
                }
          }
    }
    return out;
  }
  py::dict stats() {
    py::dict d;
    d["rectangles"] = rectangles;
    d["uniform_pieces"] = uniform;
    d["scanned_packed_bytes"] = scanned_bytes;
    d["cache_hits"] = y.hits + m.hits;
    d["cache_misses"] = y.misses + m.misses;
    d["stored_bytes_read"] = y.bytes + m.bytes;
    d["cache_capacity_bytes"] = 2 * y.limit * 32768;
    return d;
  }
};

#include "research_nonlocal.hpp"
#include "research_projection.hpp"

PYBIND11_MODULE(_research, m) {
  m.def("crc", [](py::bytes b) { return packed_crc(std::string(b)); });
  projection_research::bind(m);
  nonlocal_research::bind(m);
  py::class_<PackedFields>(m, "PackedFields")
      .def(py::init<std::string, std::string, std::array<int, 3>,
                    const std::vector<std::array<std::uint64_t, 6>> &,
                    const std::vector<std::array<std::uint64_t, 6>> &, int>(),
           py::arg("response"), py::arg("solid"), py::arg("shape_xyz"),
           py::arg("response_index"), py::arg("solid_index"),
           py::arg("cache_bricks") = 128)
      .def("boxes", &PackedFields::boxes)
      .def("leaves", &PackedFields::leaves)
      .def("stats", &PackedFields::stats);
}
