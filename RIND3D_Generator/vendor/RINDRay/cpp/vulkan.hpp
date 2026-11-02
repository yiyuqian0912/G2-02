#pragma once

#include <rindray/rindray.hpp>

namespace rindray {

// Opaque device ownership is shared by scenes from one Python Engine. No
// Vulkan headers or GPU resources are required in the reference-only build.
class VulkanDevice {
public:
    VulkanDevice();
    ~VulkanDevice();
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

class VulkanScene {
public:
    VulkanScene(std::shared_ptr<VulkanDevice> device, std::span<const float> triangle_vertices,
                std::span<const std::int32_t> geometry_ids, std::size_t memory_budget);
    ~VulkanScene();
    void trace_into(RayView rays, MutableHitView out, unsigned threads, std::size_t scratch_budget);
    void occluded_into(RayView rays, std::span<bool> out, unsigned threads, std::size_t scratch_budget);
    void visibility_into(std::span<const float> sources, std::span<const std::int32_t> kinds,
                         std::span<const float> points, std::span<bool> out,
                         std::array<float,6> bounds, float epsilon_multiplier,
                         unsigned threads, std::size_t scratch_budget);
    // pack, transfer, kernel, unpack, total ms, device-shared workspace bytes.
    std::array<double,6> metrics() const;
    void trim_workspace(std::size_t max_bytes);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
