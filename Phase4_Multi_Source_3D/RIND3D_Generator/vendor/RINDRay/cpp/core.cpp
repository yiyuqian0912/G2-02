#include <rindray/rindray.hpp>
#include "vulkan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef RINDRAY_HAS_EMBREE
#include <embree4/rtcore.h>
#endif

namespace rindray {
namespace {
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double coordinate_limit = std::numeric_limits<float>::max() / 16.0;
struct Vec {
    double x{}, y{}, z{};
    double operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
    Vec operator+(Vec b) const { return {x+b.x,y+b.y,z+b.z}; }
    Vec operator-(Vec b) const { return {x-b.x,y-b.y,z-b.z}; }
    Vec operator*(double s) const { return {x*s,y*s,z*s}; }
};
Vec load(const float* p) { return {p[0],p[1],p[2]}; }
double dot(Vec a, Vec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec cross(Vec a, Vec b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
double norm(Vec v) { return std::hypot(v.x,v.y,v.z); }
Vec unit(Vec v) { return v*(1.0/norm(v)); }
bool valid(Vec v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)
        && std::abs(v.x) <= coordinate_limit && std::abs(v.y) <= coordinate_limit
        && std::abs(v.z) <= coordinate_limit;
}
bool finite(Vec v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
struct Ray { Vec o,d; double lo,hi; };
struct Hit { double t{inf}; Vec normal{}; std::int32_t id{-1}; };
struct Box { Vec lo,hi; std::int32_t id; };
struct Sphere { Vec center; double radius; std::int32_t id; };
struct Triangle { Vec a,b,c; std::int32_t id; };
constexpr unsigned cube_faces[12][3]={{0,2,1},{1,2,3},{4,5,6},{5,7,6},{0,1,4},{1,5,4},{2,6,3},{3,6,7},{0,4,2},{2,4,6},{1,3,5},{3,7,5}};
std::array<Vec,8> cube_vertices(const Box& box) {
    std::array<Vec,8> vertices;
    for(unsigned i=0;i<8;++i) vertices[i]={(i&1)?box.hi.x:box.lo.x,(i&2)?box.hi.y:box.lo.y,(i&4)?box.hi.z:box.lo.z};
    return vertices;
}
template<class T> void ensure_capacity(std::vector<T>& values,std::size_t extra) {
    if(extra>values.max_size()-values.size()) throw std::length_error("geometry capacity overflow");
    const auto target=values.size()+extra;
    if(target>values.capacity()) values.reserve(std::max(target,std::min(values.max_size(),values.capacity()*2)));
}

struct Bytes { std::uintptr_t begin,end; };
template<typename T> Bytes bytes(std::span<T> values) {
    auto start=reinterpret_cast<std::uintptr_t>(values.data());
    return {start,start+values.size_bytes()};
}
void validate_aliases(std::span<const Bytes> outputs,std::span<const Bytes> inputs) {
    for(std::size_t i=0;i<outputs.size();++i) {
        auto a=outputs[i];
        if(a.begin==a.end) continue;
        const auto overlaps=[&](Bytes b) { return a.begin<b.end && b.begin<a.end; };
        for(auto b:inputs) require(!overlaps(b),"output buffers must not overlap input buffers");
        for(std::size_t j=0;j<i;++j) require(!overlaps(outputs[j]),"output buffers must not overlap each other");
    }
}

Box validate_visibility_input(std::span<const float> sources,std::span<const std::int32_t> kinds,
                              std::span<const float> points,std::array<float,6> bounds,float multiplier) {
    require(sources.size()%3==0 && points.size()%3==0,"sources and points must contain XYZ triples");
    auto s=sources.size()/3, n=points.size()/3;
    require(n==0 || s<=std::numeric_limits<std::size_t>::max()/n,"visibility count overflow");
    require(kinds.size()==s,"driver kinds must have length S");
    Box boundary{load(bounds.data()),load(bounds.data()+3),-1};
    require(valid(boundary.lo) && valid(boundary.hi) && boundary.lo.x<boundary.hi.x && boundary.lo.y<boundary.hi.y && boundary.lo.z<boundary.hi.z,"bounds must be finite and strictly increasing");
    require(std::isfinite(multiplier) && multiplier>0,"epsilon_multiplier must be finite and positive");
    bool directional=false;
    for(std::size_t i=0;i<s;++i) {
        Vec v=load(sources.data()+3*i);
        require(kinds[i]==0 || kinds[i]==1,"driver kind must be point or directional");
        require(kinds[i]==0 ? valid(v) : finite(v),"driver vectors must be finite; positions must fit the world range");
        require(kinds[i]!=1 || norm(v)>0,"directional driver requires a nonzero travel direction");
        directional=directional || kinds[i]==1;
    }
    const double boundary_scale=std::max({std::abs(boundary.lo.x),std::abs(boundary.lo.y),std::abs(boundary.lo.z),
                                          std::abs(boundary.hi.x),std::abs(boundary.hi.y),std::abs(boundary.hi.z)});
    const double tolerance=4*std::numeric_limits<float>::epsilon()*boundary_scale;
    for(std::size_t i=0;i<n;++i) {
        Vec p=load(points.data()+3*i); require(valid(p),"receiver points must be finite and within float32 range");
        if(directional) require(p.x>=boundary.lo.x-tolerance && p.x<=boundary.hi.x+tolerance && p.y>=boundary.lo.y-tolerance && p.y<=boundary.hi.y+tolerance && p.z>=boundary.lo.z-tolerance && p.z<=boundary.hi.z+tolerance,"directional receivers must lie inside configured scene bounds (within float32 tolerance)");
    }
    return boundary;
}

bool slab(const Box& box, const Ray& ray, double& enter, double& leave,
          Vec& nenter, Vec& nleave) {
    enter = -inf; leave = inf;
    for (int axis=0; axis<3; ++axis) {
        const double d=ray.d[axis], o=ray.o[axis];
        if (d == 0) {
            if (o < box.lo[axis] || o > box.hi[axis]) return false;
            continue;
        }
        double a=(box.lo[axis]-o)/d, b=(box.hi[axis]-o)/d;
        Vec na{}, nb{};
        if (axis==0) { na.x=-1; nb.x=1; }
        if (axis==1) { na.y=-1; nb.y=1; }
        if (axis==2) { na.z=-1; nb.z=1; }
        if (a>b) { std::swap(a,b); std::swap(na,nb); }
        if (a>enter) { enter=a; nenter=na; }
        if (b<leave) { leave=b; nleave=nb; }
        if (enter>leave) return false;
    }
    return true;
}
Hit intersect(const Box& box, const Ray& ray, bool normals) {
    double a,b; Vec na,nb;
    if (!slab(box,ray,a,b,na,nb)) return {};
    double t=a; Vec n=na;
    if (t<ray.lo) { t=b; n=nb; }
    if (t<ray.lo || t>ray.hi || !std::isfinite(t)) return {};
    return {t,normals ? n : Vec{},box.id};
}
Hit intersect(const Sphere& sphere, const Ray& ray, bool normals) {
    Vec oc=ray.o-sphere.center;
    const double a=dot(ray.d,ray.d), b=dot(oc,ray.d);
    const Vec perpendicular=cross(oc,ray.d);
    const double disc=sphere.radius*sphere.radius-dot(perpendicular,perpendicular)/a;
    if (disc<0) return {};
    const double center=-b/a, offset=std::sqrt(disc/a);
    const bool near=center-offset>=ray.lo;
    const double t=near ? center-offset : center+offset;
    if (t<ray.lo || t>ray.hi) return {};
    // Project perpendicular to the ray instead of subtracting two large
    // squared distances or reconstructing a tiny normal from a distant hit.
    const Vec normal=normals ? unit(cross(ray.d,perpendicular)*(1/a)+ray.d*(near ? -offset : offset)) : Vec{};
    return {t,normal,sphere.id};
}
Hit intersect(const Triangle& triangle, const Ray& ray, bool normals) {
    Vec e1=triangle.b-triangle.a, e2=triangle.c-triangle.a;
    Vec p=cross(ray.d,e2);
    double det=dot(e1,p);
    if (det == 0) return {};
    Vec s=ray.o-triangle.a;
    double u=dot(s,p)/det;
    if (u<0 || u>1) return {};
    Vec q=cross(s,e1);
    double v=dot(ray.d,q)/det;
    if (v<0 || u+v>1) return {};
    double t=dot(e2,q)/det;
    if (t<ray.lo || t>ray.hi) return {};
    Vec n=cross(e1,e2);
    if (norm(n)==0) return {};
    return {t,normals ? unit(n) : Vec{},triangle.id};
}

std::size_t validate(RayView rays) {
    require(rays.origins.size()%3 == 0, "origins must contain XYZ triples");
    require(rays.directions.size()==rays.origins.size(), "directions must match origins");
    std::size_t n=rays.origins.size()/3;
    require(rays.tmin.size()==1 || rays.tmin.size()==n, "tmin must have length 1 or N");
    require(rays.tmax.size()==1 || rays.tmax.size()==n, "tmax must have length 1 or N");
    if(rays.tmin.size()==1) require(std::isfinite(rays.tmin[0]) && rays.tmin[0]>=0,"tmin scalar must be finite and nonnegative");
    if(rays.tmax.size()==1) require(!std::isnan(rays.tmax[0]) && rays.tmax[0]>=0,"tmax scalar must be nonnegative and not NaN");
    if(rays.tmin.size()==1 && rays.tmax.size()==1) require(rays.tmax[0]>=rays.tmin[0],"ray bounds require tmin <= tmax");
    for (std::size_t i=0;i<n;++i) {
        Vec o=load(rays.origins.data()+3*i), d=load(rays.directions.data()+3*i);
        double lo=rays.tmin[rays.tmin.size()==1 ? 0 : i];
        double hi=rays.tmax[rays.tmax.size()==1 ? 0 : i];
        require(valid(o), "origins must be finite and within float32 world range; rebase large coordinates");
        require(finite(d) && (d.x!=0 || d.y!=0 || d.z!=0), "directions must be finite and nonzero");
        require(std::isfinite(lo) && lo>=0 && !std::isnan(hi) && hi>=lo,
                "ray bounds require finite 0 <= tmin <= tmax (positive infinity allowed for tmax)");
    }
    return n;
}
Ray ray_at(RayView rays, std::size_t i) {
    return {load(rays.origins.data()+3*i),unit(load(rays.directions.data()+3*i)),
            rays.tmin[rays.tmin.size()==1 ? 0 : i], rays.tmax[rays.tmax.size()==1 ? 0 : i]};
}
template<class F> void parallel_for(std::size_t n, unsigned requested, F fn) {
    unsigned count=static_cast<unsigned>(std::min<std::size_t>(requested,(n+4095)/4096));
    if (count<=1) { for (std::size_t i=0;i<n;++i) fn(i); return; }
    std::vector<std::jthread> workers;
    workers.reserve(count-1);
    for (unsigned k=1;k<count;++k) {
        workers.emplace_back([&,k] { for (std::size_t i=n*k/count;i<n*(k+1)/count;++i) fn(i); });
    }
    for (std::size_t i=0;i<n/count;++i) fn(i);
}
}

bool embree_available() noexcept {
#ifdef RINDRAY_HAS_EMBREE
    return true;
#else
    return false;
#endif
}
std::string embree_version() {
#ifdef RINDRAY_HAS_EMBREE
    return RTC_VERSION_STRING;
#else
    return "unavailable";
#endif
}
std::string build_id() { return RINDRAY_BUILD_ID; }

struct Scene::Impl {
    std::string backend;
    unsigned threads;
    unsigned thread_budget;
    std::size_t memory_budget;
    std::shared_ptr<VulkanDevice> gpu_device;
    bool committed{false};
    bool accel_dirty{true};
    bool gpu_dirty{true};
    std::vector<Box> boxes;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::unordered_set<std::int32_t> ids;
    mutable std::shared_mutex mutex;
#ifdef RINDRAY_HAS_VULKAN
    std::unique_ptr<VulkanScene> gpu;
#endif
#ifdef RINDRAY_HAS_EMBREE
    RTCDevice device{};
    RTCScene accel{};
    unsigned tri_geometry{RTC_INVALID_GEOMETRY_ID};
    unsigned sphere_geometry{RTC_INVALID_GEOMETRY_ID};
    std::vector<std::int32_t> triangle_ids;
    ~Impl() { if(accel) rtcReleaseScene(accel); if(device) rtcReleaseDevice(device); }
#endif
    void check_id(std::int32_t id) {
        require(id>=0 && !ids.contains(id), "geometry_id must be a unique nonnegative int32");
    }
    void ready() const {
        if (!committed) throw std::runtime_error("scene must be committed after creation or mutation");
    }
    void invalidate() {
        committed=false; accel_dirty=true; gpu_dirty=true;
#ifdef RINDRAY_HAS_VULKAN
        gpu.reset();
#endif
    }
    std::size_t scratch(std::size_t requested,std::initializer_list<std::size_t> output_bytes) const {
        auto remaining=memory_budget;
        for(auto bytes:output_bytes) { if(bytes>remaining) throw std::bad_alloc();remaining-=bytes; }
        return std::min(requested,remaining);
    }
    std::vector<Triangle> expanded() const {
        std::vector<Triangle> all;
        if(boxes.size()>(all.max_size()-triangles.size())/12) throw std::length_error("expanded geometry capacity overflow");
        all.reserve(triangles.size()+12*boxes.size());all.insert(all.end(),triangles.begin(),triangles.end());
        for(const auto& b:boxes) {
            const auto v=cube_vertices(b);
            for(const auto& f:cube_faces) all.push_back({v[f[0]],v[f[1]],v[f[2]],b.id});
        }
        return all;
    }
#ifdef RINDRAY_HAS_VULKAN
    void build_gpu() {
        if(!spheres.empty()) throw std::runtime_error("Vulkan supports boxes/triangle meshes, not spheres; use reference or Embree for the entire scene");
        if(!gpu_device) gpu_device=create_vulkan_device();
        auto all=expanded();std::vector<float> vertices;std::vector<std::int32_t> logical_ids;
        vertices.reserve(9*all.size());logical_ids.reserve(all.size());
        for(const auto& t:all) {
            for(auto v:{t.a,t.b,t.c}) {vertices.push_back(static_cast<float>(v.x));vertices.push_back(static_cast<float>(v.y));vertices.push_back(static_cast<float>(v.z));}
            logical_ids.push_back(t.id);
        }
        gpu=std::make_unique<VulkanScene>(gpu_device,vertices,logical_ids,memory_budget);gpu_dirty=false;
    }
#endif
    Hit reference(const Ray& ray, bool normals, bool any) const {
        Hit best;
        const auto consider=[&](Hit h) {
            if(h.id>=0 && (h.t<best.t || (h.t==best.t && h.id<best.id))) best=h;
            return any && h.id>=0;
        };
        for(const auto& b: boxes) if(consider(intersect(b,ray,normals))) return best;
        for(const auto& s: spheres) if(consider(intersect(s,ray,normals))) return best;
        for(const auto& t: triangles) if(consider(intersect(t,ray,normals))) return best;
        return best;
    }
    Hit trace(const Ray& ray, bool normals) const {
#ifdef RINDRAY_HAS_EMBREE
        if(backend=="embree") {
            RTCRayHit rh{};
            rh.ray.org_x=static_cast<float>(ray.o.x); rh.ray.org_y=static_cast<float>(ray.o.y); rh.ray.org_z=static_cast<float>(ray.o.z);
            rh.ray.dir_x=static_cast<float>(ray.d.x); rh.ray.dir_y=static_cast<float>(ray.d.y); rh.ray.dir_z=static_cast<float>(ray.d.z);
            rh.ray.tnear=static_cast<float>(ray.lo); rh.ray.tfar=static_cast<float>(ray.hi);
            rh.ray.mask=~0u; rh.hit.geomID=RTC_INVALID_GEOMETRY_ID;
            for(auto& id: rh.hit.instID) id=RTC_INVALID_GEOMETRY_ID;
            rtcIntersect1(accel,&rh);
            if(rh.hit.geomID==RTC_INVALID_GEOMETRY_ID) return {};
            std::int32_t id=rh.hit.geomID==tri_geometry ? triangle_ids[rh.hit.primID] : spheres[rh.hit.primID].id;
            Vec normal{};
            if(normals) normal=unit({rh.hit.Ng_x,rh.hit.Ng_y,rh.hit.Ng_z});
            return {rh.ray.tfar,normal,id};
        }
#endif
        return reference(ray,normals,false);
    }
    bool blocked(const Ray& ray) const {
#ifdef RINDRAY_HAS_EMBREE
        if(backend=="embree") {
            RTCRay r{};
            r.org_x=static_cast<float>(ray.o.x); r.org_y=static_cast<float>(ray.o.y); r.org_z=static_cast<float>(ray.o.z);
            r.dir_x=static_cast<float>(ray.d.x); r.dir_y=static_cast<float>(ray.d.y); r.dir_z=static_cast<float>(ray.d.z);
            r.tnear=static_cast<float>(ray.lo); r.tfar=static_cast<float>(ray.hi); r.mask=~0u;
            rtcOccluded1(accel,&r);
            return r.tfar<0;
        }
#endif
        return reference(ray,false,true).id>=0;
    }
#ifdef RINDRAY_HAS_EMBREE
    void check_embree() const {
        if(rtcGetDeviceError(device)!=RTC_ERROR_NONE) throw std::runtime_error("Embree device operation failed");
    }
    void build_embree() {
        if(!device) {
            const auto config="threads="+std::to_string(threads);
            device=rtcNewDevice(config.c_str());
            if(!device) throw std::runtime_error("cannot create Embree CPU device");
        }
        if(accel) rtcReleaseScene(accel);
        accel=rtcNewScene(device);
        if(!accel) throw std::runtime_error("cannot create Embree scene");
        rtcSetSceneFlags(accel,RTC_SCENE_FLAG_ROBUST);
        auto all=expanded();
        tri_geometry=RTC_INVALID_GEOMETRY_ID;
        sphere_geometry=RTC_INVALID_GEOMETRY_ID;
        triangle_ids.clear();
        if(!all.empty()) {
            require(all.size()<=std::numeric_limits<std::uint32_t>::max()/3, "mesh too large for Embree uint32 indices");
            RTCGeometry g=rtcNewGeometry(device,RTC_GEOMETRY_TYPE_TRIANGLE);
            auto* vertices=static_cast<float*>(rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,3*sizeof(float),3*all.size()));
            auto* indices=static_cast<unsigned*>(rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,3*sizeof(unsigned),all.size()));
            if(!vertices || !indices) { rtcReleaseGeometry(g); throw std::bad_alloc(); }
            triangle_ids.reserve(all.size());
            for(std::size_t i=0;i<all.size();++i) {
                const Vec v[3]={all[i].a,all[i].b,all[i].c};
                for(unsigned j=0;j<3;++j) {
                    vertices[9*i+3*j]=static_cast<float>(v[j].x); vertices[9*i+3*j+1]=static_cast<float>(v[j].y); vertices[9*i+3*j+2]=static_cast<float>(v[j].z);
                    indices[3*i+j]=static_cast<unsigned>(3*i+j);
                }
                triangle_ids.push_back(all[i].id);
            }
            rtcCommitGeometry(g); tri_geometry=rtcAttachGeometry(accel,g); rtcReleaseGeometry(g);
        }
        if(!spheres.empty()) {
            RTCGeometry g=rtcNewGeometry(device,RTC_GEOMETRY_TYPE_SPHERE_POINT);
            auto* vertices=static_cast<float*>(rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT4,4*sizeof(float),spheres.size()));
            if(!vertices) { rtcReleaseGeometry(g); throw std::bad_alloc(); }
            for(std::size_t i=0;i<spheres.size();++i) {
                vertices[4*i]=static_cast<float>(spheres[i].center.x); vertices[4*i+1]=static_cast<float>(spheres[i].center.y);
                vertices[4*i+2]=static_cast<float>(spheres[i].center.z); vertices[4*i+3]=static_cast<float>(spheres[i].radius);
            }
            rtcCommitGeometry(g); sphere_geometry=rtcAttachGeometry(accel,g); rtcReleaseGeometry(g);
        }
        rtcCommitScene(accel); check_embree(); accel_dirty=false;
    }
#endif
};

Scene::Scene(std::string backend,unsigned num_threads,std::size_t memory_budget,std::shared_ptr<VulkanDevice> device): impl_(std::make_unique<Impl>()) {
    require(backend=="reference" || backend=="embree" || backend=="vulkan", "native backend must be reference, embree, or vulkan");
    require(num_threads>=1 && num_threads<=1024,"num_threads must be between 1 and 1024");
    if(backend=="embree" && !embree_available()) throw std::runtime_error("Embree was not compiled; rebuild with Embree 4 or use backend='reference'");
    if(backend=="vulkan" && !vulkan_compiled()) throw std::runtime_error("Vulkan was not compiled; enable RINDRAY_VULKAN or use a CPU backend");
    require(memory_budget>0,"memory budget must be positive");
    impl_->backend=std::move(backend); impl_->threads=num_threads; impl_->thread_budget=num_threads;
    impl_->memory_budget=memory_budget;impl_->gpu_device=std::move(device);
}
Scene::~Scene()=default;

void Scene::add_box(std::array<float,3> center,std::array<float,3> size,std::int32_t id) {
    std::unique_lock lock(impl_->mutex);
    Vec c=load(center.data()), s=load(size.data());
    require(valid(c) && valid(s) && s.x>0 && s.y>0 && s.z>0,"box center must be finite and size strictly positive");
    Box box{c-s*.5,c+s*.5,id};
    require(valid(box.lo) && valid(box.hi),"box exceeds float32 world range");
    impl_->check_id(id); ensure_capacity(impl_->boxes,1); impl_->ids.insert(id);
    impl_->boxes.push_back(box); impl_->invalidate();
}
void Scene::add_sphere(std::array<float,3> center,float radius,std::int32_t id) {
    std::unique_lock lock(impl_->mutex);
    Vec c=load(center.data()); Vec r{radius,radius,radius};
    require(valid(c) && std::isfinite(radius) && radius>0 && valid(c-r) && valid(c+r),"sphere requires finite center and positive radius within float32 range");
    impl_->check_id(id); ensure_capacity(impl_->spheres,1); impl_->ids.insert(id);
    impl_->spheres.push_back({c,radius,id}); impl_->invalidate();
}
void Scene::add_mesh(std::span<const float> vertices,std::span<const std::uint32_t> indices,std::int32_t id) {
    std::unique_lock lock(impl_->mutex);
    require(vertices.size()%3==0 && indices.size()%3==0,"mesh buffers must contain triples");
    for(std::size_t i=0;i<vertices.size();i+=3) require(valid(load(vertices.data()+i)),"mesh vertices must be finite and within float32 range");
    for(auto i:indices) require(i<vertices.size()/3,"mesh index out of range");
    impl_->check_id(id);
    ensure_capacity(impl_->triangles,indices.size()/3);
    impl_->ids.insert(id);
    for(std::size_t i=0;i<indices.size();i+=3) impl_->triangles.push_back({load(vertices.data()+3*static_cast<std::size_t>(indices[i])),load(vertices.data()+3*static_cast<std::size_t>(indices[i+1])),load(vertices.data()+3*static_cast<std::size_t>(indices[i+2])),id});
    impl_->invalidate();
}
void Scene::add_instance(std::int32_t source_id,std::array<float,12> m,std::int32_t id) {
    std::unique_lock lock(impl_->mutex);
    require(impl_->ids.contains(source_id),"instance source geometry_id does not exist");
    impl_->check_id(id);
    for(float value:m) require(std::isfinite(value),"instance transform must be finite");
    Vec rows[3]={{m[0],m[1],m[2]},{m[4],m[5],m[6]},{m[8],m[9],m[10]}};
    double det=dot(rows[0],cross(rows[1],rows[2]));
    require(std::isfinite(det) && det!=0,"instance affine transform must be nonsingular");
    const auto transform=[&](Vec p) {
        Vec result{dot(rows[0],p)+m[3],dot(rows[1],p)+m[7],dot(rows[2],p)+m[11]};
        require(valid(result),"transformed geometry exceeds float32 world range");
        // Ingest one canonical float32 geometry for both backends.
        return Vec{static_cast<float>(result.x),static_cast<float>(result.y),static_cast<float>(result.z)};
    };
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;
    const auto append=[&](Vec a,Vec b,Vec c) {
        Triangle t{transform(a),transform(b),transform(c),id};
        if(det<0) std::swap(t.b,t.c);
        triangles.push_back(t);
    };
    for(const auto& t:impl_->triangles) if(t.id==source_id) append(t.a,t.b,t.c);
    for(const auto& b:impl_->boxes) if(b.id==source_id) {
        const auto v=cube_vertices(b);
        for(const auto& f:cube_faces) append(v[f[0]],v[f[1]],v[f[2]]);
    }
    for(const auto& s:impl_->spheres) if(s.id==source_id) {
        Vec c[3]={{m[0],m[4],m[8]},{m[1],m[5],m[9]},{m[2],m[6],m[10]}};
        const double squared=dot(c[0],c[0]);
        const double tolerance=squared*8*std::numeric_limits<float>::epsilon();
        require(std::abs(dot(c[1],c[1])-squared)<=tolerance && std::abs(dot(c[2],c[2])-squared)<=tolerance
                && std::abs(dot(c[0],c[1]))<=tolerance && std::abs(dot(c[0],c[2]))<=tolerance && std::abs(dot(c[1],c[2]))<=tolerance,
                "sphere instances require uniform scale and orthogonal axes; general ellipsoids are not approximated");
        Vec center=transform(s.center);
        double radius=s.radius*std::sqrt(squared);
        require(std::isfinite(radius) && radius>0 && radius<=coordinate_limit,"transformed sphere radius exceeds float32 range");
        radius=static_cast<float>(radius);
        Vec r{radius,radius,radius};
        require(radius>0 && valid(center-r) && valid(center+r),"transformed sphere exceeds float32 range");
        spheres.push_back({center,radius,id});
    }
    ensure_capacity(impl_->triangles,triangles.size());
    ensure_capacity(impl_->spheres,spheres.size());
    impl_->ids.insert(id);
    impl_->triangles.insert(impl_->triangles.end(),triangles.begin(),triangles.end());
    impl_->spheres.insert(impl_->spheres.end(),spheres.begin(),spheres.end());
    impl_->invalidate();
}
void Scene::commit() {
    std::unique_lock lock(impl_->mutex);
    if(impl_->committed) return;
#ifdef RINDRAY_HAS_EMBREE
    if(impl_->backend=="embree") impl_->build_embree();
#endif
#ifdef RINDRAY_HAS_VULKAN
    if(impl_->backend=="vulkan") impl_->build_gpu();
#endif
    impl_->committed=true;
}
void Scene::configure(std::string backend,unsigned threads) {
    std::unique_lock lock(impl_->mutex); impl_->ready();
    require(backend=="reference" || backend=="embree" || backend=="vulkan","backend must be reference, embree, or vulkan");
    require(threads>=1 && threads<=impl_->thread_budget,"thread count exceeds the original scene budget");
    if(backend=="embree") {
        if(!embree_available()) throw std::runtime_error("Embree was not compiled");
#ifdef RINDRAY_HAS_EMBREE
        if(impl_->accel_dirty) impl_->build_embree();
#endif
    }
    if(backend=="vulkan") {
        if(!vulkan_compiled()) throw std::runtime_error("Vulkan was not compiled");
#ifdef RINDRAY_HAS_VULKAN
        if(impl_->gpu_dirty) impl_->build_gpu();
#endif
    }
    impl_->backend=std::move(backend); impl_->threads=threads;
}
void Scene::trace_into(RayView rays,MutableHitView out,std::size_t scratch_budget) const {
    std::shared_lock lock(impl_->mutex); impl_->ready();
    const auto n=validate(rays);
    require(out.hit.empty() || out.hit.size()==n,"hit buffer has wrong length");
    require(out.t.empty() || out.t.size()==n,"t buffer has wrong length");
    require(out.geometry_id.empty() || out.geometry_id.size()==n,"geometry_id buffer has wrong length");
    require(out.normal.empty() || out.normal.size()==3*n,"normal buffer has wrong length");
    const Bytes outputs[]={bytes(out.hit),bytes(out.t),bytes(out.geometry_id),bytes(out.normal)};
    const Bytes inputs[]={bytes(rays.origins),bytes(rays.directions),bytes(rays.tmin),bytes(rays.tmax)};
    validate_aliases(outputs,inputs);
#ifdef RINDRAY_HAS_VULKAN
    if(impl_->backend=="vulkan") { impl_->gpu->trace_into(rays,out,impl_->threads,impl_->scratch(scratch_budget,{out.hit.size_bytes(),out.t.size_bytes(),out.geometry_id.size_bytes(),out.normal.size_bytes()}));return; }
#else
    static_cast<void>(scratch_budget);
#endif
    parallel_for(n,impl_->threads,[&](std::size_t i) {
        auto h=impl_->trace(ray_at(rays,i),!out.normal.empty());
        if(!out.hit.empty()) out.hit[i]=h.id>=0;
        if(!out.t.empty()) out.t[i]=static_cast<float>(h.t);
        if(!out.geometry_id.empty()) out.geometry_id[i]=h.id;
        if(!out.normal.empty()) {
            out.normal[3*i]=static_cast<float>(h.normal.x); out.normal[3*i+1]=static_cast<float>(h.normal.y); out.normal[3*i+2]=static_cast<float>(h.normal.z);
        }
    });
}
void Scene::occluded_into(RayView rays,std::span<bool> out,std::size_t scratch_budget) const {
    std::shared_lock lock(impl_->mutex); impl_->ready();
    auto n=validate(rays); require(out.size()==n,"occluded buffer has wrong length");
    const Bytes outputs[]={bytes(out)};
    const Bytes inputs[]={bytes(rays.origins),bytes(rays.directions),bytes(rays.tmin),bytes(rays.tmax)};
    validate_aliases(outputs,inputs);
#ifdef RINDRAY_HAS_VULKAN
    if(impl_->backend=="vulkan") { impl_->gpu->occluded_into(rays,out,impl_->threads,impl_->scratch(scratch_budget,{out.size_bytes()}));return; }
#else
    static_cast<void>(scratch_budget);
#endif
    parallel_for(n,impl_->threads,[&](std::size_t i) { out[i]=impl_->blocked(ray_at(rays,i)); });
}
void Scene::contains_into(std::span<const float> points,std::span<bool> out) const {
    std::shared_lock lock(impl_->mutex); impl_->ready();
    if(!impl_->triangles.empty()) throw std::runtime_error("solid occupancy is defined only for boxes and spheres; mesh scenes require a separate watertight-volume contract");
    require(points.size()%3==0 && out.size()==points.size()/3,"occupancy points/output shape mismatch");
    for(std::size_t i=0;i<out.size();++i) require(valid(load(points.data()+3*i)),"points must be finite and within float32 world range");
    const Bytes outputs[]={bytes(out)}, inputs[]={bytes(points)};
    validate_aliases(outputs,inputs);
    parallel_for(out.size(),impl_->threads,[&](std::size_t i) {
        Vec p=load(points.data()+3*i); bool occupied=false;
        for(const auto& b:impl_->boxes) {
            if(p.x>=b.lo.x && p.x<=b.hi.x && p.y>=b.lo.y && p.y<=b.hi.y && p.z>=b.lo.z && p.z<=b.hi.z) { occupied=true; break; }
        }
        if(!occupied) for(const auto& s:impl_->spheres) {
            Vec delta=p-s.center;
            if(dot(delta,delta)<=s.radius*s.radius) { occupied=true; break; }
        }
        out[i]=occupied;
    });
}
void Scene::visibility_into(std::span<const float> sources,std::span<const std::int32_t> kinds,
                            std::span<const float> points,std::span<bool> out,
                            std::array<float,6> bounds,float multiplier,std::size_t scratch_budget) const {
    std::shared_lock lock(impl_->mutex); impl_->ready();
    Box boundary=validate_visibility_input(sources,kinds,points,bounds,multiplier);
    auto s=sources.size()/3, n=points.size()/3;
    require(out.size()==s*n,"visibility shape must be [S,N]");
    const Bytes outputs[]={bytes(out)};
    const Bytes inputs[]={bytes(sources),bytes(kinds),bytes(points)};
    validate_aliases(outputs,inputs);
#ifdef RINDRAY_HAS_VULKAN
    if(impl_->backend=="vulkan") {impl_->gpu->visibility_into(sources,kinds,points,out,bounds,multiplier,impl_->threads,impl_->scratch(scratch_budget,{out.size_bytes()}));return;}
#else
    static_cast<void>(scratch_budget);
#endif
    parallel_for(s*n,impl_->threads,[&](std::size_t index) {
        auto driver=index/n, receiver=index%n;
        Vec v=load(sources.data()+3*driver), p=load(points.data()+3*receiver);
        Vec o=v, delta=p-v;
        double length=norm(delta);
        if(kinds[driver]==1) {
            // First-hit reconstruction may land a few ULPs beyond the inferred
            // bounds. Validation allows only this documented rounding envelope.
            p={std::clamp(p.x,boundary.lo.x,boundary.hi.x),std::clamp(p.y,boundary.lo.y,boundary.hi.y),std::clamp(p.z,boundary.lo.z,boundary.hi.z)};
            o=p; delta=unit(v)*-1;
            double enter,leave; Vec ne,nl;
            slab(boundary,{p,delta,0,inf},enter,leave,ne,nl); length=std::max(0.0,leave);
        }
        double scale=std::max({std::abs(o.x),std::abs(o.y),std::abs(o.z),std::abs(p.x),std::abs(p.y),std::abs(p.z),length});
        double eps=multiplier*std::numeric_limits<float>::epsilon()*scale;
        if(length<=2*eps) { out[index]=true; return; }
        Vec d=kinds[driver]==1 ? delta : delta*(1/length);
        out[index]=!impl_->blocked({o,d,eps,length-eps});
    });
}
void Scene::validate_visibility(std::span<const float> sources,std::span<const std::int32_t> kinds,
                                std::span<const float> points,std::array<float,6> bounds,float multiplier) const {
    std::shared_lock lock(impl_->mutex); impl_->ready();
    validate_visibility_input(sources,kinds,points,bounds,multiplier);
}
std::array<float,6> Scene::bounds() const {
    std::shared_lock lock(impl_->mutex);
    Vec lo{inf,inf,inf},hi{-inf,-inf,-inf};
    const auto include=[&](Vec v) {
        lo={std::min(lo.x,v.x),std::min(lo.y,v.y),std::min(lo.z,v.z)};
        hi={std::max(hi.x,v.x),std::max(hi.y,v.y),std::max(hi.z,v.z)};
    };
    for(auto& b:impl_->boxes) { include(b.lo); include(b.hi); }
    for(auto& s:impl_->spheres) { Vec r{s.radius,s.radius,s.radius}; include(s.center-r); include(s.center+r); }
    for(auto& t:impl_->triangles) { include(t.a); include(t.b); include(t.c); }
    if(!std::isfinite(lo.x)) return {-1,-1,-1,1,1,1};
    const double scale=std::max({std::abs(lo.x),std::abs(lo.y),std::abs(lo.z),std::abs(hi.x),std::abs(hi.y),std::abs(hi.z)});
    if(scale==0) return {-1,-1,-1,1,1,1};
    std::array<float,6> result;
    for(int i=0;i<3;++i) {
        result[i]=static_cast<float>(lo[i]);result[i+3]=static_cast<float>(hi[i]);
        if(!(result[i]<result[i+3])) {
            const double pad=std::max(scale*1e-4,static_cast<double>(std::numeric_limits<float>::denorm_min()));
            result[i]=static_cast<float>(std::max(-coordinate_limit,lo[i]-pad));
            result[i+3]=static_cast<float>(std::min(coordinate_limit,hi[i]+pad));
        }
    }
    return result;
}
std::array<std::size_t,3> Scene::geometry_counts() const {
    std::shared_lock lock(impl_->mutex);
    return {impl_->boxes.size(),impl_->spheres.size(),impl_->triangles.size()};
}
std::string Scene::backend() const { std::shared_lock lock(impl_->mutex); return impl_->backend; }
unsigned Scene::num_threads() const { std::shared_lock lock(impl_->mutex); return impl_->threads; }
std::array<double,6> Scene::gpu_metrics() const {
    std::shared_lock lock(impl_->mutex);
#ifdef RINDRAY_HAS_VULKAN
    if(impl_->backend=="vulkan" && impl_->gpu) return impl_->gpu->metrics();
#endif
    std::array<double,6> empty;empty.fill(std::numeric_limits<double>::quiet_NaN());return empty;
}
void Scene::trim_gpu_workspace(std::size_t max_bytes) const {
    std::shared_lock lock(impl_->mutex);
#ifdef RINDRAY_HAS_VULKAN
    if(impl_->gpu) impl_->gpu->trim_workspace(max_bytes);
#else
    static_cast<void>(max_bytes);
#endif
}
}
