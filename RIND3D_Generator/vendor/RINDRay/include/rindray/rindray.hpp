#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <span>
#include <string>

namespace rindray {

bool embree_available() noexcept;
std::string embree_version();
std::string build_id();
class VulkanDevice;
struct VulkanCapabilities {
    bool compiled{}, available{};
    std::string name, error;
    std::uint32_t api_version{}, driver_version{};
    std::uint64_t device_memory_bytes{};
    bool ray_query{}, acceleration_structure{}, buffer_device_address{}, compute_queue{};
};
VulkanCapabilities vulkan_capabilities();
bool vulkan_compiled() noexcept;
std::shared_ptr<VulkanDevice> create_vulkan_device();

// XYZ buffers are flattened contiguous float32 arrays. Bounds have length 1 or N.
// Directions are normalized once per ray; distances are in world units.
struct RayView {
    std::span<const float> origins, directions, tmin, tmax;
};

// Empty spans suppress a field. Nonempty spans must contain N (normal: 3N) items.
struct MutableHitView {
    std::span<bool> hit;
    std::span<float> t;
    std::span<std::int32_t> geometry_id;
    std::span<float> normal;
};

// Mutations/commit take exclusive ownership; committed queries share a read lock.
// Input/output memory remains caller-owned for the complete synchronous call.
class Scene {
public:
    explicit Scene(std::string backend = "reference", unsigned num_threads = 1,
                   std::size_t memory_budget = 1024ull*1024*1024,
                   std::shared_ptr<VulkanDevice> device = {});
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    void add_box(std::array<float, 3> center, std::array<float, 3> size, std::int32_t id);
    void add_sphere(std::array<float, 3> center, float radius, std::int32_t id);
    void add_mesh(std::span<const float> vertices, std::span<const std::uint32_t> indices,
                  std::int32_t id);
    // Exact affine copies are baked at insertion, not a second scene graph.
    // Sphere copies support similarities; general ellipsoids are rejected.
    void add_instance(std::int32_t source_id, std::array<float, 12> transform, std::int32_t id);
    void commit();
    // Calibration may change traversal policy within the constructor's budget.
    void configure(std::string backend, unsigned num_threads);
    void trace_into(RayView rays, MutableHitView out,
                    std::size_t scratch_budget = std::numeric_limits<std::size_t>::max()) const;
    void occluded_into(RayView rays, std::span<bool> out,
                       std::size_t scratch_budget = std::numeric_limits<std::size_t>::max()) const;
    // Solid occupancy is defined for boxes/spheres only; mesh scenes reject it.
    void contains_into(std::span<const float> points, std::span<bool> out) const;
    // kinds: 0=point position, 1=travel direction. Output is source-major [S,N].
    void visibility_into(std::span<const float> sources, std::span<const std::int32_t> kinds,
                         std::span<const float> points, std::span<bool> out,
                         std::array<float, 6> bounds, float epsilon_multiplier = 4,
                         std::size_t scratch_budget = std::numeric_limits<std::size_t>::max()) const;
    void validate_visibility(std::span<const float> sources, std::span<const std::int32_t> kinds,
                             std::span<const float> points, std::array<float, 6> bounds,
                             float epsilon_multiplier = 4) const;
    std::array<float, 6> bounds() const;
    std::array<std::size_t, 3> geometry_counts() const;
    std::string backend() const;
    unsigned num_threads() const;
    std::array<double,6> gpu_metrics() const;
    void trim_gpu_workspace(std::size_t max_bytes) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
