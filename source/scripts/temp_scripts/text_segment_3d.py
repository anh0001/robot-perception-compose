"""CLI entry point for text-driven 3D segmentation using OpenMask3D outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import open3d as o3d
from utils.openmask_interface import ensure_mask_clip_features, get_mask_points
from utils.recursive_config import Config


def _normalize_config_name(config_arg: str | None) -> str | None:
    if config_arg is None:
        return None
    config_arg = config_arg.strip()
    if not config_arg:
        return None
    config_arg = config_arg.replace("\\", "/")
    if config_arg.endswith(".yaml"):
        config_arg = config_arg[:-5]
    if config_arg.startswith("configs/"):
        config_arg = config_arg[len("configs/") :]
    return config_arg


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a 3D object by natural-language query using iPhone LiDAR scans.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Name of the config file (without extension) living in configs/.",
    )
    parser.add_argument(
        "--item",
        type=str,
        required=True,
        help="Natural language description of the target object (e.g. 'red bottle').",
    )
    parser.add_argument(
        "--scan-name",
        type=str,
        default=None,
        help="Override the configured high_res scan name (folder under data/aligned_point_clouds).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Which ranked match to return if multiple masks match the text (0 = best).",
    )
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help="Force a fresh request to the OpenMask3D server before querying.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write item/environment point clouds (PLY).",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable the Open3D visualization window.",
    )
    return parser.parse_args(argv)


def _load_config(name: str | None) -> Config:
    if name is None:
        return Config()
    normalized = _normalize_config_name(name)
    return Config(file=normalized)


def _save_cloud(cloud: o3d.geometry.PointCloud, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=True)


def _print_stats(cloud: o3d.geometry.PointCloud, label: str) -> None:
    points = np.asarray(cloud.points)
    if points.size == 0:
        print(f"{label}: empty")
        return
    centroid = points.mean(axis=0)
    print(
        f"{label}: {points.shape[0]} points | centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})",
    )


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    config = _load_config(args.config)
    if args.scan_name:
        config["pre_scanned_graphs"]["high_res"] = args.scan_name

    ensure_mask_clip_features(config=config, recompute=args.recompute_features)

    item_cloud, env_cloud = get_mask_points(
        args.item,
        config,
        idx=max(args.idx, 0),
        vis_block=False,
    )

    _print_stats(item_cloud, "Item")
    _print_stats(env_cloud, "Environment")

    if args.output_dir:
        output_root = Path(args.output_dir)
        scan_name = config["pre_scanned_graphs"]["high_res"]
        safe_item = args.item.strip().replace(" ", "_")
        item_path = output_root / scan_name / f"{safe_item}_idx{args.idx}_item.ply"
        env_path = output_root / scan_name / f"{safe_item}_idx{args.idx}_environment.ply"
        _save_cloud(item_cloud, item_path)
        _save_cloud(env_cloud, env_path)
        print(f"Saved item cloud to {item_path}")
        print(f"Saved environment cloud to {env_path}")

    if not args.no_vis:
        item_colored = o3d.geometry.PointCloud(item_cloud)
        env_colored = o3d.geometry.PointCloud(env_cloud)
        item_colored.paint_uniform_color([1, 0, 1])
        env_colored.paint_uniform_color([0.7, 0.7, 0.7])

        # Add a coordinate frame at the world origin to show axes.
        # Size scales with the scene extent for visibility.
        item_pts = np.asarray(item_colored.points)
        env_pts = np.asarray(env_colored.points)
        if item_pts.size and env_pts.size:
            pts = np.vstack([item_pts, env_pts])
        elif item_pts.size:
            pts = item_pts
        elif env_pts.size:
            pts = env_pts
        else:
            pts = np.zeros((1, 3))

        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        scene_diameter = float(np.linalg.norm(max_xyz - min_xyz))
        axis_size = 0.1 * scene_diameter if scene_diameter > 0 else 0.25

        coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=[0.0, 0.0, 0.0]
        )

        # Camera placement:
        # - In Open3D, `front` is the vector from the lookat point
        #   toward the camera position (i.e., where the camera sits).
        # - Put the camera on the +X side and slightly above (+Z), so
        #   the view looks along -X with a small downward tilt.
        # - Use Z-up convention for the viewer.
        front = [1.0, 0.0, 0.5]
        lookat = center.tolist()
        up = [0.0, 0.0, 1.0]
        zoom = 0.7

        o3d.visualization.draw_geometries(
            [env_colored, item_colored, coord_axes],
            front=front,
            lookat=lookat,
            up=up,
            zoom=zoom,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
