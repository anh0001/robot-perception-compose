"""Utilities for working with the OpenMask3D segmentation server."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

import clip
import open3d as o3d
import requests
from urllib3.exceptions import ReadTimeoutError
from utils import recursive_config
from utils.docker_communication import _get_content
from utils.recursive_config import Config

MODEL, PREPROCESS = clip.load("ViT-L/14@336px", device="cpu")


def zip_point_cloud(path: str) -> str:
    name = os.path.basename(path)
    if os.path.exists(name):
        shutil.rmtree(name)
    output_filename = os.path.join(path, f"{name}.zip")
    with zipfile.ZipFile(output_filename, "w") as zipf:
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".zip"):
                    continue
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, path))
    return output_filename


def _build_server_url(config: Config) -> str:
    """Compose the OpenMask3D server URL from the configuration."""

    server_cfg = config["servers"]["openmask"]
    ip = server_cfg.get("ip", "127.0.0.1")
    port = server_cfg.get("port", 5001)
    route = server_cfg.get("route", "openmask/save_and_predict").lstrip("/")
    return f"http://{ip}:{port}/{route}"


def _features_directory(config: Config, scan_name: str) -> Path:
    base_path = Path(config.get_subpath("openmask_features"))
    return base_path / scan_name


def _features_exist(directory: Path) -> bool:
    expected = {
        directory / "clip_features.npy",
        directory / "scene_MASKS.npy",
        directory / "clip_features_comp.npy",
        directory / "scene_MASKS_comp.npy",
    }
    return all(path.exists() for path in expected)


def get_mask_clip_features(
    config: Config | None = None,
    *,
    overwrite: bool = True,
    timeout: int = 900,
) -> Path:
    """Request mask proposals and CLIP features from the OpenMask3D server.

    Args:
        config: Optional config object (defaults to recursive Config).
        overwrite: Whether to overwrite existing cached features.
        timeout: Request timeout in seconds.

    Returns:
        Path to the directory containing the cached features.
    """

    if config is None:
        config = recursive_config.Config()

    scan_name = config["pre_scanned_graphs"]["high_res"]
    scene_root = Path(config.get_subpath("aligned_point_clouds"))
    directory_path = scene_root / scan_name
    if not directory_path.exists():
        raise FileNotFoundError(
            f"OpenMask scene folder not found: {directory_path}."
        )

    save_dir = _features_directory(config, scan_name)
    if not overwrite and _features_exist(save_dir):
        return save_dir

    zip_path = Path(zip_point_cloud(str(directory_path)))

    params = {
        "name": ("str", scan_name),
        "overwrite": ("bool", overwrite),
        "scene_intrinsic_resolution": ("str", "[1440,1920]"),
    }
    server_address = _build_server_url(config)
    tmp_dir = Path(config.get_subpath("tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zip_path.open("rb") as file_handle:
        try:
            response = requests.post(
                server_address,
                files={"scene": file_handle},
                params=params,
                timeout=timeout,
            )
        except ReadTimeoutError as exc:
            raise TimeoutError("OpenMask request timed out") from exc

    if response.status_code == 200:
        contents = _get_content(response, tmp_dir)
    else:
        message = json.loads(response.content)
        raise RuntimeError(
            f"OpenMask server error ({response.status_code}): {message.get('error')}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    feature_path = save_dir / "clip_features.npy"
    mask_path = save_dir / "scene_MASKS.npy"
    np.save(feature_path, contents["clip_features"])
    np.save(mask_path, contents["scene_MASKS"])

    features = np.load(feature_path)
    masks = np.load(mask_path)
    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    features = features[mask_idx]

    np.save(save_dir / "clip_features_comp.npy", features)
    np.save(save_dir / "scene_MASKS_comp.npy", masks)

    try:
        zip_path.unlink()
    except FileNotFoundError:
        pass

    return save_dir


def ensure_mask_clip_features(
    config: Config | None = None,
    *,
    recompute: bool = False,
) -> Path:
    """Make sure OpenMask3D features exist locally and return their directory."""

    if config is None:
        config = recursive_config.Config()

    scan_name = config["pre_scanned_graphs"]["high_res"]
    save_dir = _features_directory(config, scan_name)

    if recompute or not _features_exist(save_dir):
        return get_mask_clip_features(config=config, overwrite=True)

    return save_dir


def get_mask_points(item: str, config, idx: int = 0, vis_block: bool = False):
    pcd_name = config["pre_scanned_graphs"]["high_res"]
    base_path = config.get_subpath("openmask_features")
    feat_path = os.path.join(base_path, pcd_name, "clip_features_comp.npy")
    mask_path = os.path.join(base_path, pcd_name, "scene_MASKS_comp.npy")
    pcd_path = os.path.join(
        config.get_subpath("aligned_point_clouds"), pcd_name, "scene.ply"
    )

    features = np.load(feat_path)
    masks = np.load(mask_path)
    item = item.lower()

    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    # masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    # features = features[mask_idx]

    text = clip.tokenize([item]).to("cpu")

    # Compute the CLIP feature vector for the specified word
    with torch.no_grad():
        text_features = MODEL.encode_text(text)

    cos_sim = torch.nn.functional.cosine_similarity(
        torch.Tensor(features), text_features, dim=1
    )
    values, indices = torch.topk(cos_sim, idx + 1)
    most_sim_feat_idx = indices[-1].item()
    print(f"{most_sim_feat_idx=}", f"value={values[-1].item()}")
    # idx = 1
    mask = masks[:, most_sim_feat_idx].astype(bool)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pcd_in = pcd.select_by_index(np.where(mask)[0])
    pcd_out = pcd.select_by_index(np.where(~mask)[0])

    if vis_block:
        pcd_in.paint_uniform_color([1, 0, 1])
        o3d.visualization.draw_geometries([pcd_in, pcd_out])

    return pcd_in, pcd_out


########################################################################################
# TESTING
########################################################################################


def visualize():
    item = "cabinet, shelf"
    config = Config()
    for i in range(15):
        print(i, end=", ")
        get_mask_points(item, config, idx=i, vis_block=True)


if __name__ == "__main__":
    # get_mask_clip_features()
    visualize()
