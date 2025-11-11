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
    parser.add_argument(
        "--hide-env",
        action="store_true",
        help="Do not render the background environment cloud in Open3D.",
    )
    parser.add_argument(
        "--focus-item",
        action="store_true",
        help="Frame the Open3D camera tightly around the segmented item instead of the full scan.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Point size (in pixels) used for Open3D rendering.",
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


def _collect_points(clouds: list[o3d.geometry.PointCloud]) -> np.ndarray:
    pts = [np.asarray(cloud.points) for cloud in clouds if len(cloud.points)]
    if not pts:
        return np.zeros((1, 3))
    return np.vstack(pts)


def _create_box_wireframe(
    cloud: o3d.geometry.PointCloud,
    color: tuple[float, float, float],
    oriented: bool,
) -> o3d.geometry.Geometry:
    if oriented:
        box = cloud.get_oriented_bounding_box()
    else:
        box = cloud.get_axis_aligned_bounding_box()
    box.color = color
    return box


def _visualize_clouds(
    item_cloud: o3d.geometry.PointCloud,
    env_cloud: o3d.geometry.PointCloud,
    *,
    focus_item: bool,
    hide_env: bool,
    point_size: float,
) -> None:
    item_colored = o3d.geometry.PointCloud(item_cloud)
    env_colored = o3d.geometry.PointCloud(env_cloud)
    item_colored.paint_uniform_color([1.0, 0.0, 1.0])
    env_colored.paint_uniform_color([0.72, 0.72, 0.72])

    geometries: list[o3d.geometry.Geometry] = []
    focus_candidates: list[o3d.geometry.PointCloud] = []

    if not hide_env and len(env_colored.points):
        geometries.append(env_colored)
        geometries.append(_create_box_wireframe(env_colored, (0.55, 0.55, 0.55), oriented=False))
        focus_candidates.append(env_colored)

    if len(item_colored.points):
        geometries.append(item_colored)
        geometries.append(_create_box_wireframe(item_colored, (0.95, 0.2, 0.95), oriented=True))
        focus_candidates.append(item_colored)

    if not geometries:
        print("Nothing to visualize — both clouds are empty.")
        return

    pts = _collect_points(focus_candidates if focus_item and focus_candidates else [item_colored, env_colored])
    min_xyz = pts.min(axis=0)
    max_xyz = pts.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    scene_extent = float(np.linalg.norm(max_xyz - min_xyz))
    if scene_extent < 1e-6:
        scene_extent = 1.0

    axis_size = 0.12 * scene_extent
    coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size if axis_size > 0 else 0.25, origin=[0.0, 0.0, 0.0]
    )
    geometries.append(coord_axes)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Text Segment 3D", width=1280, height=720)
    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    if render_option:
        render_option.point_size = max(1.0, point_size)
        render_option.background_color = np.array([0.98, 0.98, 0.98])

    view_ctl = vis.get_view_control()
    front = np.array([1.0, 0.0, 0.45])
    front /= np.linalg.norm(front)
    view_ctl.set_front(front.tolist())
    view_ctl.set_lookat(center.tolist())
    view_ctl.set_up([0.0, 0.0, 1.0])
    zoom = np.clip(1.5 / scene_extent, 0.25, 0.85)
    if focus_item:
        zoom = min(0.95, zoom * 1.25)
    view_ctl.set_zoom(float(zoom))

    vis.run()
    vis.destroy_window()


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
        _visualize_clouds(
            item_cloud,
            env_cloud,
            focus_item=args.focus_item,
            hide_env=args.hide_env,
            point_size=args.point_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
