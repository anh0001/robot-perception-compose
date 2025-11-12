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

MODEL, PREPROCESS = clip.load("ViT-L/14@336px", device="cuda")
MODEL_DEVICE = next(MODEL.parameters()).device
FEATURE_CACHE_VERSION = 2
_PROMPT_TEMPLATES: tuple[str, ...] = (
    "{}",
    "a photo of {}",
    "a photo of a {}",
    "an indoor photo of {}",
    "a close-up photo of {}",
    "a detailed scan of {}",
    "a high quality render of {}",
)
_ITEM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "monitor": ("computer monitor", "computer display", "computer screen", "lcd monitor"),
    "screen": ("computer screen", "digital display", "lcd screen"),
    "window": ("glass window", "office window", "sunlit window"),
    "door": ("wooden door", "glass door", "office door"),
    "chair": ("office chair", "desk chair", "rolling chair"),
    "desk": ("office desk", "work desk", "computer desk"),
    "table": ("dining table", "kitchen table", "office table"),
    "cabinet": ("storage cabinet", "wooden cabinet", "office cabinet"),
    "shelf": ("bookshelf", "storage shelf", "wooden shelf"),
    "drawer": ("desk drawer", "cabinet drawer", "pull-out drawer"),
    "keyboard": ("computer keyboard", "pc keyboard"),
    "mouse": ("computer mouse", "pc mouse"),
    "laptop": ("open laptop", "closed laptop", "notebook computer"),
    "plant": ("potted plant", "indoor plant"),
    "whiteboard": ("office whiteboard", "dry erase board"),
    "poster": ("office poster", "wall poster"),
    "sofa": ("office sofa", "couch"),
    "couch": ("living room couch", "office couch"),
    "trash can": ("garbage bin", "trash bin"),
    "trash": ("trash can",),
    "recycling": ("recycling bin",),
    "tv": ("television", "flat screen tv"),
    "microwave": ("kitchen microwave",),
    "fridge": ("refrigerator", "mini fridge"),
}
_QUERY_SEPARATORS: tuple[str, ...] = (",", "|")


def _metadata_path(directory: Path) -> Path:
    return directory / "metadata.json"


def _write_feature_metadata(directory: Path, *, num_masks: int, num_features: int) -> None:
    metadata = {
        "version": FEATURE_CACHE_VERSION,
        "num_masks": int(num_masks),
        "num_features": int(num_features),
    }
    _metadata_path(directory).write_text(json.dumps(metadata, indent=2))


def _load_metadata(directory: Path) -> dict | None:
    metadata_file = _metadata_path(directory)
    if not metadata_file.exists():
        return None
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _needs_cache_upgrade(directory: Path) -> bool:
    metadata = _load_metadata(directory)
    if metadata is None:
        return True
    return metadata.get("version") != FEATURE_CACHE_VERSION


def _write_companion_cache(directory: Path, features: np.ndarray, masks: np.ndarray) -> None:
    np.save(directory / "clip_features_comp.npy", features)
    np.save(directory / "scene_MASKS_comp.npy", masks)
    _write_feature_metadata(
        directory,
        num_masks=masks.shape[1],
        num_features=features.shape[0],
    )


def _rebuild_companion_cache(directory: Path) -> None:
    feature_src = directory / "clip_features.npy"
    mask_src = directory / "scene_MASKS.npy"
    if not feature_src.exists() or not mask_src.exists():
        raise FileNotFoundError(
            f"Cannot rebuild OpenMask cache in {directory}: missing base npy files."
        )
    shutil.copy2(feature_src, directory / "clip_features_comp.npy")
    shutil.copy2(mask_src, directory / "scene_MASKS_comp.npy")
    features = np.load(feature_src, mmap_mode="r")
    masks = np.load(mask_src, mmap_mode="r")
    _write_feature_metadata(
        directory,
        num_masks=masks.shape[1],
        num_features=features.shape[0],
    )


def _split_query_terms(item: str) -> list[str]:
    if not item:
        return []
    segments = [item]
    for separator in _QUERY_SEPARATORS:
        expanded: list[str] = []
        for segment in segments:
            if separator in segment:
                expanded.extend(segment.split(separator))
            else:
                expanded.append(segment)
        segments = expanded
    terms = [segment.strip() for segment in segments if segment.strip()]
    fallback = item.strip()
    return terms or ([fallback] if fallback else ["object"])


def _expand_with_aliases(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for raw_term in terms:
        candidates = [raw_term]
        lookup_key = raw_term.lower()
        candidates.extend(_ITEM_SYNONYMS.get(lookup_key, ()))
        for cand in candidates:
            normalized = cand.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            expanded.append(cand)
    return expanded


def _build_prompt_bank(terms: list[str]) -> list[str]:
    prompts: list[str] = []
    for term in terms:
        for template in _PROMPT_TEMPLATES:
            prompts.append(template.format(term))
    return prompts


def _encode_text_query(item: str) -> torch.Tensor:
    """Return a normalized CLIP embedding that averages prompt variants."""

    terms = _split_query_terms(item)
    terms = _expand_with_aliases(terms)
    prompts = _build_prompt_bank(terms)
    tokens = clip.tokenize(prompts).to(MODEL_DEVICE)
    with torch.no_grad():
        text_features = MODEL.encode_text(tokens)
    text_features = torch.nn.functional.normalize(text_features, dim=1)
    return torch.nn.functional.normalize(text_features.mean(dim=0, keepdim=True), dim=1)


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

    params = [
        ("name", "str"),
        ("name", scan_name),
        ("overwrite", "bool"),
        ("overwrite", "True" if overwrite else "False"),
        ("scene_intrinsic_resolution", "str"),
        ("scene_intrinsic_resolution", "[1440,1920]"),
    ]
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
        try:
            message = json.loads(response.content)
            error_msg = message.get('error', 'Unknown error')
        except (json.JSONDecodeError, AttributeError):
            error_msg = response.content.decode('utf-8', errors='replace')[:500]
        raise RuntimeError(
            f"OpenMask server error ({response.status_code}): {error_msg}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    feature_path = save_dir / "clip_features.npy"
    mask_path = save_dir / "scene_MASKS.npy"
    features = np.asarray(contents["clip_features"])
    masks = np.asarray(contents["scene_MASKS"])
    np.save(feature_path, features)
    np.save(mask_path, masks)
    _write_companion_cache(save_dir, features, masks)

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

    if _needs_cache_upgrade(save_dir):
        _rebuild_companion_cache(save_dir)

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

    feature_tensor = torch.as_tensor(
        features, dtype=torch.float32, device=MODEL_DEVICE
    )
    text_features = _encode_text_query(item)
    text_features = text_features.to(feature_tensor.device, dtype=feature_tensor.dtype)
    cos_sim = torch.nn.functional.cosine_similarity(
        feature_tensor, text_features, dim=1
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
