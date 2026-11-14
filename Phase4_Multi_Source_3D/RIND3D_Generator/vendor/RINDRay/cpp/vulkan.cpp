#include "vulkan.hpp"
#include <stdexcept>

#ifndef RINDRAY_HAS_VULKAN
namespace rindray {
struct VulkanDevice::Impl {};
bool vulkan_compiled() noexcept { return false; }
VulkanCapabilities vulkan_capabilities() {
    VulkanCapabilities result;result.error="Vulkan was not compiled; rebuild with RINDRAY_VULKAN=ON for optional GPU support";return result;
}
VulkanDevice::VulkanDevice() { throw std::runtime_error("Vulkan was not compiled"); }
VulkanDevice::~VulkanDevice()=default;
std::shared_ptr<VulkanDevice> create_vulkan_device() { return std::make_shared<VulkanDevice>(); }
}
#else
#include <vulkan/vulkan.h>
#include "query_spv.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace rindray {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
void check(VkResult result,const char* operation) {
    if(result!=VK_SUCCESS) throw std::runtime_error(std::string("Vulkan ")+operation+" failed (VkResult "+std::to_string(result)+"); check device access or use a CPU backend");
}
struct Instance {
    VkInstance value{};
    ~Instance() { if(value) vkDestroyInstance(value,nullptr); }
    void create() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="RINDRay";app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};info.pApplicationInfo=&app;
        check(vkCreateInstance(&info,nullptr,&value),"instance creation");
    }
};
struct Selection {
    VkPhysicalDevice physical{};
    VkPhysicalDeviceProperties properties{};
    std::uint32_t family{},timestamp_bits{};
    VulkanCapabilities capabilities;
};
Selection select(VkInstance instance) {
    std::uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"device enumeration");
    std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"device enumeration");
    for(auto candidate:devices) {
        VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        rq.pNext=&as;as.pNext=&bda;VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};features.pNext=&rq;
        vkGetPhysicalDeviceFeatures2(candidate,&features);VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(candidate,&props);
        if(!rq.rayQuery || !as.accelerationStructure || !bda.bufferDeviceAddress || props.apiVersion<VK_API_VERSION_1_2) continue;
        std::uint32_t ecount=0;check(vkEnumerateDeviceExtensionProperties(candidate,nullptr,&ecount,nullptr),"extension enumeration");
        std::vector<VkExtensionProperties> extensions(ecount);check(vkEnumerateDeviceExtensionProperties(candidate,nullptr,&ecount,extensions.data()),"extension enumeration");
        const auto has=[&](const char* name) {return std::any_of(extensions.begin(),extensions.end(),[&](const auto& e) {return std::strcmp(name,e.extensionName)==0;});};
        if(!has(VK_KHR_RAY_QUERY_EXTENSION_NAME) || !has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) || !has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) continue;
        std::uint32_t qcount=0;vkGetPhysicalDeviceQueueFamilyProperties(candidate,&qcount,nullptr);
        std::vector<VkQueueFamilyProperties> queues(qcount);vkGetPhysicalDeviceQueueFamilyProperties(candidate,&qcount,queues.data());
        for(std::uint32_t i=0;i<qcount;++i) if(queues[i].queueCount && (queues[i].queueFlags&VK_QUEUE_COMPUTE_BIT)) {
            VulkanCapabilities caps;caps.compiled=true;caps.available=true;caps.name=props.deviceName;caps.api_version=props.apiVersion;caps.driver_version=props.driverVersion;
            VkPhysicalDeviceMemoryProperties memory{};vkGetPhysicalDeviceMemoryProperties(candidate,&memory);
            for(std::uint32_t j=0;j<memory.memoryHeapCount;++j) if(memory.memoryHeaps[j].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) caps.device_memory_bytes+=memory.memoryHeaps[j].size;
            caps.ray_query=caps.acceleration_structure=caps.buffer_device_address=caps.compute_queue=true;
            return {candidate,props,i,queues[i].timestampValidBits,std::move(caps)};
        }
    }
    throw std::runtime_error("no visible Vulkan 1.2 device with rayQuery, accelerationStructure, bufferDeviceAddress and compute queues; check GPU permissions or use CPU");
}
struct Buffer {
    VkDevice device{};VkBuffer handle{};VkDeviceMemory memory{};void* mapped{};VkDeviceSize size{},allocated{};
    explicit Buffer(VkDevice d):device(d) {}
    ~Buffer() { if(mapped) vkUnmapMemory(device,memory);if(handle) vkDestroyBuffer(device,handle,nullptr);if(memory) vkFreeMemory(device,memory,nullptr); }
};
struct Acceleration {
    VkDevice device{};PFN_vkDestroyAccelerationStructureKHR destroy{};VkAccelerationStructureKHR handle{};std::unique_ptr<Buffer> storage;
    ~Acceleration() { if(handle) destroy(device,handle,nullptr); }
};
struct Workspace {
    std::unique_ptr<Buffer> input,output,upload,download;
    std::size_t capacity{},bytes{};
    std::uint64_t generation{1};
    void clear() {
        input.reset();output.reset();upload.reset();download.reset();
        capacity=bytes=0;++generation;
    }
};
struct Context {
    Instance instance;
    VkPhysicalDevice physical{};VkDevice device{};VkQueue queue{};
    VkPhysicalDeviceProperties properties{};VkPhysicalDeviceMemoryProperties memory_properties{};
    VkCommandPool command_pool{};VkCommandBuffer command{};VkFence fence{};VkQueryPool queries{};
    VkDescriptorSetLayout descriptor_layout{};VkPipelineLayout pipeline_layout{};VkPipeline pipeline{};
    VkDeviceSize scratch_alignment{256};std::uint32_t timestamp_bits{};std::uint64_t max_primitives{};
    PFN_vkCreateAccelerationStructureKHR create_as{};PFN_vkDestroyAccelerationStructureKHR destroy_as{};
    PFN_vkGetAccelerationStructureBuildSizesKHR build_sizes{};PFN_vkCmdBuildAccelerationStructuresKHR cmd_build{};
    PFN_vkGetAccelerationStructureDeviceAddressKHR as_address{};
    Workspace workspace;
    std::mutex mutex;
    ~Context() {
        if(device) vkDeviceWaitIdle(device);
        workspace.clear();
        if(pipeline) vkDestroyPipeline(device,pipeline,nullptr);
        if(pipeline_layout) vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
        if(descriptor_layout) vkDestroyDescriptorSetLayout(device,descriptor_layout,nullptr);
        if(queries) vkDestroyQueryPool(device,queries,nullptr);
        if(fence) vkDestroyFence(device,fence,nullptr);
        if(command_pool) vkDestroyCommandPool(device,command_pool,nullptr);
        if(device) vkDestroyDevice(device,nullptr);
    }
    void init() {
        instance.create();auto selected=select(instance.value);physical=selected.physical;properties=selected.properties;
        timestamp_bits=properties.limits.timestampComputeAndGraphics ? selected.timestamp_bits : 0;
        vkGetPhysicalDeviceMemoryProperties(physical,&memory_properties);
        VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};props2.pNext=&as_props;vkGetPhysicalDeviceProperties2(physical,&props2);
        scratch_alignment=as_props.minAccelerationStructureScratchOffsetAlignment;
        max_primitives=as_props.maxPrimitiveCount;
        VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};bda.bufferDeviceAddress=VK_TRUE;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};as.accelerationStructure=VK_TRUE;as.pNext=&bda;
        VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};rq.rayQuery=VK_TRUE;rq.pNext=&as;
        const char* extensions[]={VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,VK_KHR_RAY_QUERY_EXTENSION_NAME};
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=selected.family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.pNext=&rq;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.enabledExtensionCount=3;di.ppEnabledExtensionNames=extensions;
        check(vkCreateDevice(physical,&di,nullptr,&device),"device creation");vkGetDeviceQueue(device,selected.family,0,&queue);
        create_as=reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device,"vkCreateAccelerationStructureKHR"));
        destroy_as=reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device,"vkDestroyAccelerationStructureKHR"));
        build_sizes=reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device,"vkGetAccelerationStructureBuildSizesKHR"));
        cmd_build=reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device,"vkCmdBuildAccelerationStructuresKHR"));
        as_address=reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device,"vkGetAccelerationStructureDeviceAddressKHR"));
        if(!create_as || !destroy_as || !build_sizes || !cmd_build || !as_address) throw std::runtime_error("Vulkan acceleration structure entry points are unavailable");
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;pi.queueFamilyIndex=selected.family;
        check(vkCreateCommandPool(device,&pi,nullptr,&command_pool),"command pool");
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=command_pool;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&ai,&command),"command buffer");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device,&fi,nullptr,&fence),"fence");
        if(timestamp_bits) { VkQueryPoolCreateInfo ti{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};ti.queryType=VK_QUERY_TYPE_TIMESTAMP;ti.queryCount=4;check(vkCreateQueryPool(device,&ti,nullptr,&queries),"timestamp pool"); }
        std::array<VkDescriptorSetLayoutBinding,5> bindings{};
        for(std::uint32_t i=0;i<bindings.size();++i) {bindings[i].binding=i;bindings[i].descriptorCount=1;bindings[i].descriptorType=i==0?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};layout.bindingCount=5;layout.pBindings=bindings.data();
        check(vkCreateDescriptorSetLayout(device,&layout,nullptr,&descriptor_layout),"descriptor layout");
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,80};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptor_layout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&push;
        check(vkCreatePipelineLayout(device,&pl,nullptr,&pipeline_layout),"pipeline layout");
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=sizeof(rindray_shader);sm.pCode=rindray_shader;VkShaderModule module{};
        check(vkCreateShaderModule(device,&sm,nullptr,&module),"shader module");
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=pipeline_layout;ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";
        auto result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline);vkDestroyShaderModule(device,module,nullptr);check(result,"compute pipeline");
    }
    std::unique_ptr<Buffer> buffer(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags flags,VkMemoryPropertyFlags preferred=0,std::size_t* remaining=nullptr) {
        auto result=std::make_unique<Buffer>(device);result->size=std::max<VkDeviceSize>(size,16);
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=result->size;bi.usage=usage;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device,&bi,nullptr,&result->handle),"buffer creation");VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device,result->handle,&req);
        if(remaining && req.size>*remaining) throw std::bad_alloc();
        result->allocated=req.size;
        std::uint32_t type=~0u;
        for(std::uint32_t i=0;i<memory_properties.memoryTypeCount;++i) if((req.memoryTypeBits&(1u<<i)) && (memory_properties.memoryTypes[i].propertyFlags&flags)==flags) {
            if(type==~0u || (memory_properties.memoryTypes[i].propertyFlags&preferred)==preferred) type=i;
            if((memory_properties.memoryTypes[i].propertyFlags&preferred)==preferred) break;
        }
        if(type==~0u) throw std::runtime_error("Vulkan device lacks a required buffer memory type");
        VkMemoryAllocateFlagsInfo af{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};af.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};mi.allocationSize=req.size;mi.memoryTypeIndex=type;if(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) mi.pNext=&af;
        check(vkAllocateMemory(device,&mi,nullptr,&result->memory),"buffer memory allocation");check(vkBindBufferMemory(device,result->handle,result->memory,0),"buffer memory binding");
        if(flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) check(vkMapMemory(device,result->memory,0,VK_WHOLE_SIZE,0,&result->mapped),"host mapping");
        if(remaining) *remaining-=static_cast<std::size_t>(req.size);
        return result;
    }
    VkDeviceAddress address(Buffer* b) const { VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=b->handle;return vkGetBufferDeviceAddress(device,&info); }
    void begin() {
        check(vkResetFences(device,1,&fence),"fence reset");check(vkResetCommandBuffer(command,0),"command reset");
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};info.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;check(vkBeginCommandBuffer(command,&info),"command begin");
    }
    void barrier(VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags from,VkAccessFlags to) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=from;b.dstAccessMask=to;vkCmdPipelineBarrier(command,src,dst,0,1,&b,0,nullptr,0,nullptr);
    }
    void submit() {
        check(vkEndCommandBuffer(command),"command end");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
        check(vkQueueSubmit(queue,1,&submit,fence),"queue submit");check(vkWaitForFences(device,1,&fence,VK_TRUE,30'000'000'000ull),"GPU wait");
    }
    void upload(Buffer* destination,const void* data,std::size_t size) {
        auto staging=buffer(size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memcpy(staging->mapped,data,size);begin();barrier(VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy copy{0,0,size};vkCmdCopyBuffer(command,staging->handle,destination->handle,1,&copy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);submit();
    }
    std::unique_ptr<Acceleration> build(VkAccelerationStructureTypeKHR type,VkAccelerationStructureGeometryKHR geometry,std::uint32_t count) {
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};info.type=type;info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;info.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;info.geometryCount=1;info.pGeometries=&geometry;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};build_sizes(device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&info,&count,&sizes);
        auto result=std::make_unique<Acceleration>();result->device=device;result->destroy=destroy_as;
        result->storage=buffer(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};create.buffer=result->storage->handle;create.size=sizes.accelerationStructureSize;create.type=type;
        check(create_as(device,&create,nullptr,&result->handle),"acceleration structure creation");
        auto scratch=buffer(sizes.buildScratchSize+scratch_alignment,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto base=address(scratch.get());info.scratchData.deviceAddress=(base+scratch_alignment-1)&~(scratch_alignment-1);info.dstAccelerationStructure=result->handle;
        VkAccelerationStructureBuildRangeInfoKHR range{};range.primitiveCount=count;const auto* ranges=&range;
        begin();cmd_build(command,1,&info,&ranges);barrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);submit();return result;
    }
};
template<class F> void parallel(std::size_t n,unsigned budget,F fn) {
    const auto threads=std::min<std::size_t>(budget,std::max<std::size_t>(1,(n+4095)/4096));std::vector<std::jthread> workers;
    for(std::size_t k=1;k<threads;++k) workers.emplace_back([&,k] { for(auto i=n*k/threads;i<n*(k+1)/threads;++i) fn(i); });
    for(std::size_t i=0;i<n/threads;++i) fn(i);
}
struct alignas(16) GpuRay { float origin_lo[4],direction_hi[4]; };
struct alignas(16) GpuHit { float t_normal[4];std::int32_t hit_id[4]; };
struct BasicHit { float t;std::uint32_t hit;std::int32_t id; };
struct alignas(16) Parameters { std::uint32_t meta[4]{};float driver[4]{},lower[4]{},upper[4]{},limits[4]{0,0,4,0}; };
static_assert(sizeof(GpuRay)==32 && sizeof(GpuHit)==32 && sizeof(BasicHit)==12 && sizeof(Parameters)==80);
}

struct VulkanDevice::Impl { Context context;Impl() { context.init(); } };
VulkanDevice::VulkanDevice():impl(std::make_unique<Impl>()) {}
VulkanDevice::~VulkanDevice()=default;
bool vulkan_compiled() noexcept { return true; }
VulkanCapabilities vulkan_capabilities() {
    VulkanCapabilities result;result.compiled=true;
    try { Instance instance;instance.create();result=select(instance.value).capabilities; }
    catch(const std::exception& error) {result.error=error.what();}
    return result;
}
std::shared_ptr<VulkanDevice> create_vulkan_device() { return std::make_shared<VulkanDevice>(); }

struct VulkanScene::Impl {
    std::shared_ptr<VulkanDevice> owner;
    std::unique_ptr<Buffer> vertices,ids;
    std::unique_ptr<Acceleration> blas,tlas;
    VkDescriptorPool pool{};VkDescriptorSet descriptor{};
    std::size_t memory_budget{};
    std::uint64_t descriptor_generation{};
    std::array<double,6> last{};
    bool empty{};
    Context& context() const { return owner->impl->context; }
    ~Impl() { if(pool) vkDestroyDescriptorPool(context().device,pool,nullptr); }
    void descriptors() {
        auto& ctx=context();
        VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};as.accelerationStructureCount=1;as.pAccelerationStructures=&tlas->handle;
        Buffer* buffers[]={ctx.workspace.input.get(),ctx.workspace.output.get(),vertices.get(),ids.get()};std::array<VkDescriptorBufferInfo,4> infos{};std::array<VkWriteDescriptorSet,5> writes{};
        for(std::uint32_t i=0;i<writes.size();++i) {
            auto& w=writes[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=descriptor;w.dstBinding=i;w.descriptorCount=1;
            w.descriptorType=i==0?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            if(i==0) w.pNext=&as;else {infos[i-1]={buffers[i-1]->handle,0,buffers[i-1]->size};w.pBufferInfo=&infos[i-1];}
        }
        vkUpdateDescriptorSets(ctx.device,5,writes.data(),0,nullptr);
        descriptor_generation=ctx.workspace.generation;
    }
    std::size_t workspace(std::size_t wanted,std::size_t scratch_budget) {
        auto& ctx=context();auto& cache=ctx.workspace;scratch_budget=std::min(scratch_budget,memory_budget);
        auto maximum=std::min<std::size_t>({wanted,1'000'000,scratch_budget/128,
            ctx.properties.limits.maxStorageBufferRange/32,
            static_cast<std::size_t>(ctx.properties.limits.maxComputeWorkGroupCount[0])*128});
        if(!maximum) throw std::bad_alloc();
        if(cache.capacity>=maximum && cache.bytes<=scratch_budget) {
            if(descriptor_generation!=cache.generation) descriptors();
            return maximum;
        }
        cache.clear();
        while(maximum) {
            try {
                auto remaining=scratch_budget;
                cache.input=ctx.buffer(maximum*32,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,0,&remaining);
                cache.output=ctx.buffer(maximum*32,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,0,&remaining);
                cache.upload=ctx.buffer(maximum*32,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,0,&remaining);
                cache.download=ctx.buffer(maximum*32,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT,&remaining);
                cache.capacity=maximum;cache.bytes=scratch_budget-remaining;descriptors();return maximum;
            } catch(const std::bad_alloc&) {
                cache.clear();maximum/=2;
            }
        }
        throw std::bad_alloc();
    }
    template<class Pack,class Unpack> void execute(std::size_t count,Parameters params,Pack pack,Unpack unpack) {
        auto& ctx=context();auto& cache=ctx.workspace;auto start=Clock::now();auto phase=start;pack(cache.upload->mapped);last[0]+=elapsed(phase);
        ctx.begin();
        if(ctx.queries) {vkCmdResetQueryPool(ctx.command,ctx.queries,0,4);vkCmdWriteTimestamp(ctx.command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,ctx.queries,0);}
        ctx.barrier(VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy in{0,0,count*(params.meta[1]>=2?12:32)};vkCmdCopyBuffer(ctx.command,cache.upload->handle,cache.input->handle,1,&in);
        ctx.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        if(ctx.queries) vkCmdWriteTimestamp(ctx.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,ctx.queries,1);
        vkCmdBindPipeline(ctx.command,VK_PIPELINE_BIND_POINT_COMPUTE,ctx.pipeline);vkCmdBindDescriptorSets(ctx.command,VK_PIPELINE_BIND_POINT_COMPUTE,ctx.pipeline_layout,0,1,&descriptor,0,nullptr);
        params.meta[0]=static_cast<std::uint32_t>(count);vkCmdPushConstants(ctx.command,ctx.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(params),&params);
        vkCmdDispatch(ctx.command,static_cast<std::uint32_t>((count+127)/128),1,1);
        if(ctx.queries) vkCmdWriteTimestamp(ctx.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,ctx.queries,2);
        ctx.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy out{0,0,count*(params.meta[1]==0?((params.meta[2]&1)?32:12):4)};vkCmdCopyBuffer(ctx.command,cache.output->handle,cache.download->handle,1,&out);
        if(ctx.queries) vkCmdWriteTimestamp(ctx.command,VK_PIPELINE_STAGE_TRANSFER_BIT,ctx.queries,3);
        ctx.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);ctx.submit();
        if(ctx.queries) {
            std::array<std::uint64_t,4> ticks{};check(vkGetQueryPoolResults(ctx.device,ctx.queries,0,4,sizeof(ticks),ticks.data(),sizeof(std::uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"timestamp read");
            const auto mask=ctx.timestamp_bits==64?~std::uint64_t{0}:(std::uint64_t{1}<<ctx.timestamp_bits)-1;
            const double period=ctx.properties.limits.timestampPeriod/1e6;
            last[1]+=(((ticks[1]-ticks[0])&mask)+((ticks[3]-ticks[2])&mask))*period;
            last[2]+=((ticks[2]-ticks[1])&mask)*period;
        } else last[1]=last[2]=std::numeric_limits<double>::quiet_NaN();
        phase=Clock::now();unpack(cache.download->mapped);last[3]+=elapsed(phase);last[4]+=elapsed(start);last[5]=static_cast<double>(cache.bytes);
    }
};

VulkanScene::VulkanScene(std::shared_ptr<VulkanDevice> device,std::span<const float> triangles,
                         std::span<const std::int32_t> logical_ids,std::size_t memory_budget):impl_(std::make_unique<Impl>()) {
    auto& state=*impl_;state.owner=device?std::move(device):create_vulkan_device();state.memory_budget=memory_budget;
    if(logical_ids.empty()) {state.empty=true;return;}
    if(triangles.size()!=logical_ids.size()*9 || logical_ids.size()>std::numeric_limits<std::uint32_t>::max()/3) throw std::invalid_argument("Vulkan triangle array exceeds the supported uint32 range");
    auto& ctx=state.context();std::scoped_lock lock(ctx.mutex);
    if(triangles.size_bytes()>ctx.properties.limits.maxStorageBufferRange || logical_ids.size()>ctx.max_primitives)
        throw std::invalid_argument("Vulkan geometry exceeds this device's storage-buffer/primitive limits; split the scene or use a CPU backend");
    const auto usage=VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    state.vertices=ctx.buffer(triangles.size_bytes(),usage,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    state.ids=ctx.buffer(logical_ids.size_bytes(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ctx.upload(state.vertices.get(),triangles.data(),triangles.size_bytes());ctx.upload(state.ids.get(),logical_ids.data(),logical_ids.size_bytes());
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};geometry.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR;geometry.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& data=geometry.geometry.triangles;data.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;data.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT;
    data.vertexData.deviceAddress=ctx.address(state.vertices.get());data.vertexStride=12;data.maxVertex=static_cast<std::uint32_t>(triangles.size()/3-1);data.indexType=VK_INDEX_TYPE_NONE_KHR;
    state.blas=ctx.build(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,geometry,static_cast<std::uint32_t>(logical_ids.size()));
    VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};address.accelerationStructure=state.blas->handle;
    VkAccelerationStructureInstanceKHR instance{};instance.transform.matrix[0][0]=instance.transform.matrix[1][1]=instance.transform.matrix[2][2]=1;
    instance.mask=0xff;instance.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;instance.accelerationStructureReference=ctx.as_address(ctx.device,&address);
    auto instances=ctx.buffer(sizeof(instance),VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ctx.upload(instances.get(),&instance,sizeof(instance));
    VkAccelerationStructureGeometryKHR top{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};top.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;top.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;top.geometry.instances.data.deviceAddress=ctx.address(instances.get());
    state.tlas=ctx.build(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,top,1);
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool.poolSizeCount=2;pool.pPoolSizes=sizes;pool.maxSets=1;
    check(vkCreateDescriptorPool(ctx.device,&pool,nullptr,&state.pool),"descriptor pool");
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};allocate.descriptorPool=state.pool;allocate.descriptorSetCount=1;allocate.pSetLayouts=&ctx.descriptor_layout;
    check(vkAllocateDescriptorSets(ctx.device,&allocate,&state.descriptor),"descriptor allocation");
}
VulkanScene::~VulkanScene()=default;

void VulkanScene::trace_into(RayView rays,MutableHitView out,unsigned threads,std::size_t scratch_budget) {
    auto& state=*impl_;auto& ctx=state.context();std::scoped_lock lock(ctx.mutex);state.last.fill(0);
    const auto n=rays.origins.size()/3;if(!n) return;
    if(state.empty) {
        parallel(n,threads,[&](std::size_t i) {if(!out.hit.empty()) out.hit[i]=false;if(!out.t.empty()) out.t[i]=std::numeric_limits<float>::infinity();if(!out.geometry_id.empty()) out.geometry_id[i]=-1;if(!out.normal.empty()) std::fill_n(out.normal.data()+3*i,3,0.f);});return;
    }
    const auto step=state.workspace(n,scratch_budget);
    for(std::size_t start=0;start<n;start+=step) {
        const auto count=std::min(step,n-start);Parameters params;params.meta[2]=out.normal.empty()?0u:1u;
        state.execute(count,params,[&](void* pointer) {
            auto* packed=static_cast<GpuRay*>(pointer);
            parallel(count,threads,[&](std::size_t local) {
                auto i=start+local;auto* o=rays.origins.data()+3*i;auto* d=rays.directions.data()+3*i;
                double length=std::hypot(static_cast<double>(d[0]),static_cast<double>(d[1]),static_cast<double>(d[2]));
                for(int k=0;k<3;++k) {packed[local].origin_lo[k]=o[k];packed[local].direction_hi[k]=static_cast<float>(d[k]/length);}
                packed[local].origin_lo[3]=rays.tmin[rays.tmin.size()==1?0:i];packed[local].direction_hi[3]=rays.tmax[rays.tmax.size()==1?0:i];
            });
        },[&](void* pointer) {
            if(out.normal.empty()) {
                const auto* result=static_cast<const BasicHit*>(pointer);
                parallel(count,threads,[&](std::size_t local) {
                    auto i=start+local;if(!out.hit.empty()) out.hit[i]=result[local].hit!=0;
                    if(!out.t.empty()) out.t[i]=result[local].t;if(!out.geometry_id.empty()) out.geometry_id[i]=result[local].id;
                });return;
            }
            const auto* result=static_cast<const GpuHit*>(pointer);
            parallel(count,threads,[&](std::size_t local) {
                auto i=start+local;if(!out.hit.empty()) out.hit[i]=result[local].hit_id[0]!=0;if(!out.t.empty()) out.t[i]=result[local].t_normal[0];if(!out.geometry_id.empty()) out.geometry_id[i]=result[local].hit_id[1];
                if(!out.normal.empty()) std::copy_n(result[local].t_normal+1,3,out.normal.data()+3*i);
            });
        });
    }
}
void VulkanScene::occluded_into(RayView rays,std::span<bool> out,unsigned threads,std::size_t scratch_budget) {
    auto& state=*impl_;auto& ctx=state.context();std::scoped_lock lock(ctx.mutex);state.last.fill(0);const auto n=out.size();if(!n) return;
    if(state.empty) {std::fill(out.begin(),out.end(),false);return;}
    const auto step=state.workspace(n,scratch_budget);
    for(std::size_t start=0;start<n;start+=step) {
        auto count=std::min(step,n-start);Parameters params;params.meta[1]=1;
        state.execute(count,params,[&](void* pointer) {
            auto* packed=static_cast<GpuRay*>(pointer);
            parallel(count,threads,[&](std::size_t local) {auto i=start+local;auto* o=rays.origins.data()+3*i;auto* d=rays.directions.data()+3*i;
                double length=std::hypot(static_cast<double>(d[0]),static_cast<double>(d[1]),static_cast<double>(d[2]));
                for(int k=0;k<3;++k) {packed[local].origin_lo[k]=o[k];packed[local].direction_hi[k]=static_cast<float>(d[k]/length);}
                packed[local].origin_lo[3]=rays.tmin[rays.tmin.size()==1?0:i];packed[local].direction_hi[3]=rays.tmax[rays.tmax.size()==1?0:i];});
        },[&](void* pointer) {const auto* result=static_cast<const std::uint32_t*>(pointer);parallel(count,threads,[&](std::size_t i) {out[start+i]=result[i]!=0;});});
    }
}
void VulkanScene::visibility_into(std::span<const float> sources,std::span<const std::int32_t> kinds,
                                  std::span<const float> points,std::span<bool> out,std::array<float,6> bounds,
                                  float epsilon,unsigned threads,std::size_t scratch_budget) {
    auto& state=*impl_;auto& ctx=state.context();std::scoped_lock lock(ctx.mutex);state.last.fill(0);const auto n=points.size()/3;
    if(out.empty()) return;if(state.empty) {std::fill(out.begin(),out.end(),true);return;}
    const auto step=state.workspace(n,scratch_budget);
    for(std::size_t source=0;source<kinds.size();++source) {
        Parameters params;params.meta[1]=kinds[source]==0?2u:3u;params.limits[2]=epsilon;
        double length=kinds[source]==1?std::hypot(static_cast<double>(sources[3*source]),static_cast<double>(sources[3*source+1]),static_cast<double>(sources[3*source+2])):1;
        for(int k=0;k<3;++k) {params.driver[k]=static_cast<float>(sources[3*source+k]/length);params.lower[k]=bounds[k];params.upper[k]=bounds[k+3];}
        for(std::size_t start=0;start<n;start+=step) {
            auto count=std::min(step,n-start);
            state.execute(count,params,[&](void* pointer) {std::memcpy(pointer,points.data()+3*start,count*12);},
                [&](void* pointer) {const auto* result=static_cast<const std::uint32_t*>(pointer);parallel(count,threads,[&](std::size_t i) {out[source*n+start+i]=result[i]!=0;});});
        }
    }
}
std::array<double,6> VulkanScene::metrics() const {
    auto& state=*impl_;std::scoped_lock lock(state.context().mutex);return state.last;
}
void VulkanScene::trim_workspace(std::size_t max_bytes) {
    auto& state=*impl_;std::scoped_lock lock(state.context().mutex);
    if(state.context().workspace.bytes>max_bytes) {
        state.context().workspace.clear();state.last[5]=0;
    }
}
}
#endif
