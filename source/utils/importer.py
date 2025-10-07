"""
Library open3d is a bit finicky with imports.
One can either import from cuda (for GPU-enables machines) or cpu (for others).
By checking the config, based on the "device" attribute, we can decide which one to import.
For use simply import from utils.importer, i.e. "from utils.importer import Pointcloud".
"""

from utils.recursive_config import Config

import open3d as o3d

_conf = Config()
_device = _conf.get("device", "cpu")

def _select_backend():
    """Return the Open3D backend modules without double-registering types."""
    use_cuda = _device == "cuda"

    def _try_get(module, *attrs):
        """Get nested attributes from a module; return None if any part is missing."""
        for attr in attrs:
            module = getattr(module, attr, None)
            if module is None:
                return None
        return module

    if use_cuda:
        cuda_geometry = _try_get(o3d, "cuda", "pybind", "geometry")
        cuda_utility = _try_get(o3d, "cuda", "pybind", "utility")
        if cuda_geometry and cuda_utility:
            return cuda_geometry, cuda_utility

    cpu_geometry = _try_get(o3d, "cpu", "pybind", "geometry")
    cpu_utility = _try_get(o3d, "cpu", "pybind", "utility")
    if cpu_geometry and cpu_utility:
        return cpu_geometry, cpu_utility

    return o3d.geometry, o3d.utility

_geometry, _utility = _select_backend()

AxisAlignedBoundingBox = _geometry.AxisAlignedBoundingBox
PointCloud = _geometry.PointCloud
TriangleMesh = _geometry.TriangleMesh
Vector3dVector = _utility.Vector3dVector
