"""RINDRay: batched exact visibility and reproducible response queries."""

from . import _core
from .scene import Engine, Scene, CustomRays
from .drivers import (PointDriver, DirectionalDriver, DriverRealization, QueryResult,
                      sample_drivers, compose)
from .grids import Camera, Volume
from .dataset import Dataset, Sample

__version__ = "0.1.0"


def capabilities(*, probe_devices=False):
    """Report compiled features and implemented operations, never planned APIs."""
    compiled = ["reference"] + (["embree"] if _core.embree_available() else [])
    cpu_backends = list(compiled)
    vulkan = _core.vulkan_compiled()
    device = _core.vulkan_capabilities() if probe_devices else {"compiled": vulkan, "available": None}
    if vulkan:
        compiled.append("vulkan")
    from .sampling import SAMPLERS
    return {
        "version": __version__,
        "compiled_backends": compiled,
        "backends": cpu_backends + (["vulkan"] if device.get("available") else []),
        "geometry": ["box", "sphere", "triangle_mesh", "affine_copy"],
        "instance_strategy": "baked meshes/boxes; similarity-only spheres",
        "queries": ["trace", "occluded", "visibility", "query", "render", "slice", "volume", "custom_rays", "contains_points"],
        "occupancy_geometry": ["box", "sphere"],
        "response_models": ["binary_any", "count_visible", "sum_weighted_visibility"],
        "array_types": ["numpy"],
        "query_samplers": list(SAMPLERS),
        "vulkan": vulkan,
        "device_features": {"vulkan": device},
        "vulkan_geometry": ["box", "triangle_mesh", "affine_copy"] if vulkan else [],
        "backend_versions": {"reference": __version__, "embree": _core.embree_version()},
    }
