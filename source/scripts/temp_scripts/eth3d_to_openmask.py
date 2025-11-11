from __future__ import annotations

"""
Convert an ETH3D scene (e.g., "office") into the OpenMask3D-compatible
scene folder this repo expects under data/aligned_point_clouds/<scan-name>/.

Inputs (typical ETH3D layout after unpacking archives):
  <eth3d_root>/
    cameras.txt            # COLMAP camera intrinsics (undistorted DSLR)
    images.txt             # COLMAP image poses (world->camera)
    images/ or undistorted # JPEG images
    depth/                 # depth maps aligned to images (float meters or uint16)
    scan/ or mesh.*        # optional ground-truth mesh (clean or raw)

Outputs (OpenMask3D scene):
  data/aligned_point_clouds/<scan-name>/
    color/      00000.jpg, ...
    depth/      00000.png, ...  (uint16, millimeters)
    intrinsic/  intrinsic_color.txt  (4x4)
    pose/       00000.txt, ...   (4x4, world->camera)
    scene.ply   fused point cloud (or converted from ground-truth mesh)
    mesh.obj    ground-truth mesh if available, else omitted

Usage:
  python -m scripts.temp_scripts.eth3d_to_openmask \
    --eth3d-root data/eth3d/office \
    --scan-name office

Notes:
  - We assume a fixed DSLR intrinsics for all views (true for ETH3D DSLRs).
  - If a mesh is present (e.g., from "office_scan_clean.7z"), we copy/convert it.
    Otherwise, we fuse a point cloud from the depth maps and poses.
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from utils import recursive_config


def _read_cameras_txt(path: Path) -> Tuple[int, int, np.ndarray]:
    """Parse COLMAP cameras.txt; return (width, height, K 3x3)."""
    if not path.exists():
        raise FileNotFoundError(f"cameras.txt not found at {path}")

    model = None
    width = height = None
    params = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
            parts = line.split()
            if len(parts) < 5:
                continue
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            break

    if model is None:
        raise ValueError("Could not parse intrinsics from cameras.txt")

    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
        # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        # fx == fy, cx, cy
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise NotImplementedError(f"Unsupported camera model: {model}")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return width, height, K


def _read_images_txt(path: Path) -> Dict[str, np.ndarray]:
    """Parse COLMAP images.txt; return mapping image_name -> 4x4 world->camera."""
    if not path.exists():
        raise FileNotFoundError(f"images.txt not found at {path}")

    poses: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        while True:
            header = f.readline()
            if not header:
                break
            if header.startswith("#") or not header.strip():
                continue
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            parts = header.strip().split()
            if len(parts) < 10:
                continue
            qw, qx, qy, qz = [float(x) for x in parts[1:5]]
            tx, ty, tz = [float(x) for x in parts[5:8]]
            name = parts[9]
            # Skip the following point line
            _ = f.readline()

            # COLMAP convention: world->cam: X_c = R * X_w + t
            # The quaternion in images.txt is cam rotation (world->cam)
            # Build rotation matrix
            q = np.array([qw, qx, qy, qz], dtype=np.float64)
            R = _quat_to_rotmat(q)
            t = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3:4] = t
            poses[name] = T
    if not poses:
        raise ValueError("No poses parsed from images.txt")
    return poses


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    # normalized quaternion assumed
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def _find_images_dir(root: Path) -> Path:
    for name in ("images", "undistorted", "dslr_images_undistorted"):
        p = root / name
        if p.exists() and p.is_dir():
            # Check if this directory directly contains images
            has_images = any(x.suffix.lower() in (".jpg", ".jpeg", ".png") for x in p.iterdir() if x.is_file())
            if has_images:
                return p
            # Otherwise check subdirectories
            for subdir in p.iterdir():
                if subdir.is_dir():
                    has_images_in_subdir = any(x.suffix.lower() in (".jpg", ".jpeg", ".png") for x in subdir.iterdir() if x.is_file())
                    if has_images_in_subdir:
                        return subdir
    # fallback: any dir with jpgs
    for child in root.iterdir():
        if child.is_dir() and any(str(x).lower().endswith(".jpg") for x in child.iterdir()):
            return child
    raise FileNotFoundError("Could not locate images directory under ETH3D root")


def _find_depth_dir(root: Path) -> Path | None:
    for name in ("depth", "dslr_depth", "ground_truth_depth"):
        p = root / name
        if p.exists() and p.is_dir():
            # Check if this directory directly contains depth files
            has_depth = any(x.is_file() for x in p.iterdir())
            if has_depth:
                return p
            # Otherwise check subdirectories
            for subdir in p.iterdir():
                if subdir.is_dir():
                    has_depth_in_subdir = any(x.is_file() for x in subdir.iterdir())
                    if has_depth_in_subdir:
                        return subdir
    return None


def _find_mesh(root: Path) -> Path | None:
    """Prefer color-preserving geometry (e.g., scan_clean/*.ply)."""

    candidates: list[tuple[int, Path]] = []
    for ext in (".ply", ".obj"):
        for p in root.rglob(f"*{ext}"):
            if re.search(r"scan|mesh|surface|clean", p.name, re.IGNORECASE):
                score = 0
                lower = p.as_posix().lower()
                if "clean" in lower:
                    score += 3
                if "color" in lower:
                    score += 2
                if "eval" in lower:
                    score -= 1
                if ext == ".obj":
                    score += 1  # prefer explicit meshes when available
                candidates.append((score, p))
    if not candidates:
        return None
    # Highest score wins; fall back to file size (larger first).
    candidates.sort(key=lambda item: (item[0], item[1].stat().st_size), reverse=True)
    return candidates[0][1]


def _sanitize_name_to_index_map(files: list[Path]) -> Dict[str, int]:
    # Map filename (without extension) to contiguous index
    names = sorted([f.name for f in files])
    mapping: Dict[str, int] = {}
    for i, name in enumerate(names):
        stem = Path(name).stem
        mapping[stem] = i
    return mapping


def _ensure_mm_uint16(depth: np.ndarray) -> np.ndarray:
    # Convert to uint16 millimeters
    if depth.dtype == np.uint16:
        return depth
    # many ETH3D depths are float32 meters
    depth_m = depth.astype(np.float32)
    depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    return depth_mm


def _fuse_point_cloud(
    image_files: list[Path],
    depth_dir: Path,
    poses: Dict[str, np.ndarray],
    K: np.ndarray,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    for img_path in image_files:
        name = img_path.stem
        # depth can be png or exr; try both
        dpath_png = depth_dir / f"{name}.png"
        dpath_exr = depth_dir / f"{name}.exr"
        if dpath_png.exists():
            depth = cv2.imread(str(dpath_png), cv2.IMREAD_UNCHANGED)
        elif dpath_exr.exists():
            depth = cv2.imread(str(dpath_exr), cv2.IMREAD_UNCHANGED)
        else:
            continue

        depth_mm = _ensure_mm_uint16(depth)
        h, w = depth_mm.shape[:2]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, intrinsic_matrix=K)
        o3d_depth = o3d.geometry.Image(depth_mm)
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb is None:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
        o3d_color = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, convert_rgb_to_intensity=False
        )
        T_wc = np.linalg.inv(poses[name])  # camera->world
        pview = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic, np.asarray(T_wc, dtype=np.float64)
        )
        pview = pview.voxel_down_sample(voxel_size=0.01)
        pcd += pview
    return pcd


def _load_colored_point_cloud(path: Path) -> Optional[o3d.geometry.PointCloud]:
    """Read a point cloud and ensure it carries RGB colors."""

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        return None
    if not pcd.has_colors():
        return None
    return pcd


def _downsample_if_needed(
    pcd: o3d.geometry.PointCloud,
    *,
    target_points: int = 8_000_000,
    initial_voxel: float = 0.004,
) -> o3d.geometry.PointCloud:
    """Voxel-downsample very dense clouds to keep OpenMask memory use in check."""

    if len(pcd.points) <= target_points:
        return pcd

    voxel = initial_voxel
    best = pcd
    while voxel <= 0.02:
        down = pcd.voxel_down_sample(voxel)
        if len(down.points) == 0:
            break
        best = down
        if len(down.points) <= target_points:
            print(f"[eth3d_to_openmask] Downsampled point cloud to {len(down.points)} points (voxel={voxel:.3f}m).")
            return down
        voxel += 0.002

    print(
        f"[eth3d_to_openmask] Could not reach target density; keeping {len(best.points)} points "
        f"with voxel={voxel - 0.002:.3f}m."
    )
    return best


def convert_eth3d_to_openmask(eth3d_root: Path, scan_name: str, *, overwrite: bool = False) -> Path:
    cfg = recursive_config.Config()
    out_root = Path(cfg.get_subpath("aligned_point_clouds")) / scan_name

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene already exists: {out_root}")
        shutil.rmtree(out_root)

    # Parse calibration and poses
    width, height, K = _read_cameras_txt(eth3d_root / "cameras.txt")
    poses = _read_images_txt(eth3d_root / "images.txt")

    # Locate inputs
    images_dir = _find_images_dir(eth3d_root)
    depth_dir = _find_depth_dir(eth3d_root)
    mesh_path = _find_mesh(eth3d_root)

    out_color = out_root / "color"
    out_depth = out_root / "depth"
    out_pose = out_root / "pose"
    out_intrinsic = out_root / "intrinsic"
    out_root.mkdir(parents=True, exist_ok=False)
    out_color.mkdir()
    out_depth.mkdir()
    out_pose.mkdir()
    out_intrinsic.mkdir()

    # Copy/rename images, write poses
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    name_to_idx = _sanitize_name_to_index_map(image_files)

    for img_path in image_files:
        name = img_path.stem
        if name not in poses:
            # Some ETH3D packs use relative paths in images.txt; match by suffix
            cand = next((k for k in poses.keys() if k.endswith(img_path.name)), None)
            if cand is None:
                continue
            name = cand
        idx = name_to_idx[Path(name).stem]
        out_name = str(idx)
        # preserve original resolution
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out_color / f"{out_name}.jpg"), img)

        # pose
        T_wc = poses[name]  # world->cam
        np.savetxt(out_pose / f"{out_name}.txt", T_wc, fmt="%.8f")

        # depth (optional but recommended)
        if depth_dir is not None:
            for ext in (".png", ".exr", ".JPG", ".jpg"):
                dpath = depth_dir / f"{Path(name).stem}{ext}"
                if dpath.exists():
                    if ext in (".JPG", ".jpg"):
                        # ETH3D stores depth as raw float32 binary with .JPG extension
                        # Read as binary float32 array and infer dimensions
                        depth_data = np.fromfile(str(dpath), dtype=np.float32)
                        # Try common aspect ratios to find dimensions
                        total_pixels = len(depth_data)
                        # Assume 3:2 aspect ratio (common for DSLR)
                        h_depth = int(np.sqrt(total_pixels * 2 / 3))
                        w_depth = total_pixels // h_depth
                        if h_depth * w_depth == total_pixels:
                            depth = depth_data.reshape((h_depth, w_depth))
                        else:
                            # Fallback: try to reshape with camera dims
                            depth = depth_data.reshape((height, width))
                    else:
                        depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
                    depth_mm = _ensure_mm_uint16(depth)
                    cv2.imwrite(str(out_depth / f"{out_name}.png"), depth_mm)
                    break

    # Intrinsics (single file, 4x4)
    K4 = np.eye(4, dtype=np.float64)
    K4[:3, :3] = K
    np.savetxt(out_intrinsic / "intrinsic_color.txt", K4, fmt="%.8f")

    # Scene geometry: use mesh if present, else fuse from depths
    scene_ply = out_root / "scene.ply"
    mesh_obj = out_root / "mesh.obj"

    pcd: Optional[o3d.geometry.PointCloud] = None
    if mesh_path is not None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(mesh_obj), mesh)
            pcd = mesh.sample_points_poisson_disk(number_of_points=500_000)
        else:
            pcd = _load_colored_point_cloud(mesh_path)
            if pcd is None:
                print(
                    f"[eth3d_to_openmask] Candidate geometry at {mesh_path} has no RGB colors; "
                    "falling back to RGB-D fusion."
                )
    if pcd is None or not pcd.has_colors():
        if depth_dir is None:
            raise RuntimeError(
                "No colorized geometry found and no depth directory available to fuse a point cloud."
            )
        print("[eth3d_to_openmask] Fusing RGB-D views to recover colored point cloud...")
        pcd = _fuse_point_cloud(image_files, depth_dir, poses, K)
        if not pcd.has_colors():
            raise RuntimeError(
                "RGB-D fusion did not produce colored points; verify ETH3D inputs."
            )

    pcd = _downsample_if_needed(pcd)
    o3d.io.write_point_cloud(str(scene_ply), pcd)

    return out_root


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert ETH3D scene to OpenMask3D format")
    ap.add_argument("--eth3d-root", required=True, help="Path to unpacked ETH3D scene root")
    ap.add_argument("--scan-name", required=True, help="Name for output scene folder")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing aligned scene folder if it already exists.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    eth3d_root = Path(args.eth3d_root).expanduser().resolve()
    if not eth3d_root.exists():
        raise FileNotFoundError(f"ETH3D root not found: {eth3d_root}")
    out = convert_eth3d_to_openmask(eth3d_root, args.scan_name, overwrite=args.overwrite)
    print(f"Wrote OpenMask3D scene: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
