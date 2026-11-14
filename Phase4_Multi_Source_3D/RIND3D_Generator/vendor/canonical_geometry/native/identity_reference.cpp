// Independent C++ reference encoder for frozen RIND identity vectors.
// Build: c++ -std=c++20 -O2 identity_reference.cpp -lcrypto -o identity_reference
#include <openssl/sha.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<std::uint8_t>;

struct Field {
    std::string name;
    std::uint8_t tag;
    std::vector<std::uint64_t> shape;
    Bytes payload;
};

void integer(Bytes& out, std::uint64_t value, int width) {
    for (int i = 0; i < width; ++i) {
        out.push_back(static_cast<std::uint8_t>(value & 255));
        value >>= 8;
    }
}

void string(Bytes& out, const std::string& value) {
    integer(out, value.size(), 4);
    out.insert(out.end(), value.begin(), value.end());
}

Field text(const std::string& name, const std::string& value) {
    return {name, 7, {value.size()}, Bytes(value.begin(), value.end())};
}

Field u8(const std::string& name, const std::vector<std::uint64_t>& shape,
         const Bytes& values) {
    return {name, 2, shape, values};
}

Field u32(const std::string& name, const std::vector<std::uint64_t>& shape,
          const std::vector<std::uint32_t>& values) {
    Bytes out;
    for (auto value : values) integer(out, value, 4);
    return {name, 4, shape, out};
}

Field u64(const std::string& name, const std::vector<std::uint64_t>& shape,
          const std::vector<std::uint64_t>& values) {
    Bytes out;
    for (auto value : values) integer(out, value, 8);
    return {name, 5, shape, out};
}

Field f64(const std::string& name, const std::vector<std::uint64_t>& shape,
          const std::vector<double>& values) {
    Bytes out;
    for (auto value : values) integer(out, std::bit_cast<std::uint64_t>(value), 8);
    return {name, 1, shape, out};
}

Bytes frame(const std::string& type, std::vector<Field> fields) {
    std::sort(fields.begin(), fields.end(),
              [](const Field& a, const Field& b) { return a.name < b.name; });
    Bytes out{'R','I','N','D','-','C','O','N','T','E','N','T',0};
    string(out, "rind-content-sha256-v2");
    string(out, "rind-frame-le-1");
    string(out, type);
    integer(out, fields.size(), 4);
    for (const auto& field : fields) {
        string(out, field.name);
        out.push_back(field.tag);
        out.push_back(field.shape.size());
        for (auto size : field.shape) integer(out, size, 8);
        integer(out, field.payload.size(), 8);
        out.insert(out.end(), field.payload.begin(), field.payload.end());
    }
    return out;
}

Bytes hash(const Bytes& data) {
    Bytes digest(SHA256_DIGEST_LENGTH);
    SHA256(data.data(), data.size(), digest.data());
    return digest;
}

std::string hex(const Bytes& data) {
    static const char alphabet[] = "0123456789abcdef";
    std::string out;
    out.reserve(data.size() * 2);
    for (auto byte : data) {
        out.push_back(alphabet[byte >> 4]);
        out.push_back(alphabet[byte & 15]);
    }
    return out;
}

void emit(const std::string& name, const Bytes& content) {
    std::cout << name << ' ' << hex(hash(content)) << ' ' << hex(content) << '\n';
}

int main() {
    auto empty_root = frame("empty_root", {u8("dimension", {}, {2})});
    auto empty_geometry = frame("geometry", {
        u8("dimension", {}, {2}), u8("root_kind", {}, {0}),
        u8("root_hash", {32}, hash(empty_root)),
        text("solid_semantics", "regular-closed-omega-1"),
    });
    auto box = frame("leaf", {
        f64("center", {2}, {.5, .5}),
        text("codec_name", "affine_box_v1"), u8("dimension", {}, {2}),
        f64("matrix", {2,2}, {.25, 0, 0, .125}),
    });
    auto polygon = frame("leaf", {
        f64("center", {2}, {0, 0}),
        text("codec_name", "polygon_rings_2d_v1"), u8("dimension", {}, {2}),
        f64("matrix", {2,2}, {1, 0, 0, 1}),
        u64("ring_vertex_offsets", {3}, {0, 4, 8}),
        u32("ring_owner", {2}, {UINT32_MAX, 0}),
        u8("ring_role", {2}, {0, 1}),
        f64("vertices", {8,2}, {
            0,0, 1,0, 1,1, 0,1,
            .25,.25, .25,.75, .75,.75, .75,.25,
        }),
    });
    Bytes duplicate_children = hash(box);
    auto second = hash(box);
    duplicate_children.insert(duplicate_children.end(), second.begin(), second.end());
    auto shared_csg = frame("csg_node", {
        u8("child_hashes", {2,32}, duplicate_children),
        text("csg_semantics", "regular-csg-1"),
        u8("dimension", {}, {2}), u8("opcode", {}, {0}),
    });
    auto binary_physics = frame("physics", {text("physics_codec", "binary_v1")});
    auto binary_scene = frame("scene", {
        f64("driver_position", {3,2}, {.25,.5, .75,.5, .75,.5}),
        u8("geometry_id", {32}, hash(empty_geometry)),
        text("physics_codec", "binary_v1"),
        u8("physics_hash", {32}, hash(binary_physics)),
    });
    auto exp_physics = frame("physics", {
        text("physics_codec", "exponential_v1"), f64("tau", {}, {.25}),
    });
    auto additive_scene = frame("scene", {
        f64("driver_position", {3,2}, {.25,.5, .75,.5, .75,.5}),
        f64("driver_strength", {3}, {1, .5, .5}),
        u8("geometry_id", {32}, hash(empty_geometry)),
        text("physics_codec", "exponential_v1"),
        u8("physics_hash", {32}, hash(exp_physics)),
    });
    emit("empty_root", empty_root);
    emit("empty_geometry", empty_geometry);
    emit("box_leaf", box);
    emit("polygon_hole_leaf", polygon);
    emit("shared_csg", shared_csg);
    emit("binary_physics", binary_physics);
    emit("binary_duplicate_scene", binary_scene);
    emit("exponential_physics", exp_physics);
    emit("additive_duplicate_scene", additive_scene);
}
