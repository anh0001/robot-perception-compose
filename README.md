# Text-Driven 3D Scene Segmentation

This repository is a streamlined fork of the Spot-Compose codebase focused solely on open-vocabulary 3D segmentation from iPhone LiDAR scans. The robot control stack, grasp planning, and drawer manipulation routines were removed; what remains is the tooling necessary to:

- capture a dense scene with the iOS 3D Scanner app,
- convert the capture into an [OpenMask3D](https://openmask3d.github.io/) scene,
- query objects by free-form text and extract their point clouds.

The workflow below runs entirely on a workstation—no Spot robot or GraspNet server is required.

## 1. Environment Setup

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The repo targets Python 3.8 and depends on Open3D, PyTorch, Transformers (for CLIP), and Flask/request tooling for talking to the OpenMask3D server. GPU support is optional but recommended for the Docker container that hosts OpenMask3D.

## 2. Capture an iPhone LiDAR Scan

Use the [3D Scanner App](https://apps.apple.com/us/app/3d-scanner-app/id1419913995) and keep an AprilTag or another stable reference in view.

> Note on 3D Scanner settings
> - Use "Normal" mode (default LiDAR) for capture.
> - After finishing the scan, process the textured data using "Custom".
> - Set a small voxel size and set "Simplify" to the lower option to
>   preserve as much detail as possible.
> - Allow processing time up to about 30 minutes; that is typically fine.
>   Longer than ~30 minutes often causes the app to crash. Aim for the most
>   detailed settings that your iPhone can process reliably within this time.
> - Hold the iPhone closer to the desired spot during capture for better scanning results.
> - Before exporting, use the app's crop tool to remove unnecessary
>   background/points so downstream processing is cleaner and faster.

1. Export **All Data** and **Point Cloud/PLY** (High Density, `Z axis up` disabled).
2. Create a folder `data/prescans/<scan-name>/` and unzip *All Data* there.
3. Rename the exported PLY file to `pcd.ply` and place it inside the same folder.

Update `configs/config.yaml`:

```yaml
pre_scanned_graphs:
  high_res: "<scan-name>"
servers:
  openmask:
    ip: "127.0.0.1"
    port: 5001
    route: "openmask/save_and_predict"
```

The `device` field defaults to `cpu`; change it to `cuda` if Open3D with CUDA is installed.

## 3. Prepare an OpenMask3D Scene

The preparation script reprojects the iPhone capture into the folder structure expected by OpenMask3D. Run it once per scan:

```bash
python -m scripts.point_cloud_scripts.full_align \
  --scan-name <scan-name> \
  --skip-autowalk \
  --visualize  # optional debugging windows
```

Output folder: `data/aligned_point_clouds/<scan-name>/`, containing `pose/`, `color/`, `depth/`, `intrinsic/`, `scene.ply`, and `mesh.obj`.

> If you also have a Spot autowalk point cloud, drop it under `data/point_clouds/` and omit `--skip-autowalk` to enable ICP alignment. Otherwise the scene is left in the coordinate frame defined by the AprilTag.

## 4. Launch the OpenMask3D Server

### GPU Setup (Recommended)

The OpenMask3D server runs best with GPU acceleration. First, verify your setup:

```bash
bash scripts/setup_gpu_docker.sh
```

This diagnostic script checks:
- NVIDIA driver installation (`nvidia-smi`)
- Docker GPU access
- NVIDIA Container Toolkit configuration

If the toolkit is missing, the script offers automatic installation.

Once verified, launch the server:

```bash
docker pull craiden/openmask:v1.0
docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0
# inside the container
python3 app.py
```

**VS Code Task**: `Launch OpenMask3D Server (GPU)`

Keep the server running while issuing segmentation queries.

## 5. Segment Objects by Text Query

The CLI automatically requests CLIP features from the server (cached under `data/openmask_features/`), selects the best matching mask, and optionally writes PLY files.

```bash
python -m scripts.temp_scripts.text_segment_3d \
  --item "red bottle" \
  --scan-name <scan-name> \
  --output-dir output/  # optional save location
```

Useful flags:

- `--idx`: inspect lower-ranked matches (`0` is the best cosine-similarity score).
- `--recompute-features`: refreshes the cached masks/features before querying.
- `--no-vis`: run headless (skip Open3D visualization).

The script prints basic statistics about the segmented points and, unless suppressed, launches an Open3D window that overlays the object (magenta) and the remaining scene (gray).

## Repository Layout

```
source/
├── scripts/
│   ├── point_cloud_scripts/
│   │   └── full_align.py        # prepares OpenMask3D scene folders
│   └── temp_scripts/
│       └── text_segment_3d.py   # text-driven point cloud segmentation
├── utils/
│   ├── openmask_interface.py    # talks to the OpenMask server
│   ├── docker_communication.py  # helper for zip uploads/downloads
│   └── recursive_config.py      # hierarchical config loader
configs/
└── data/
    ├── prescans/                # raw iPhone exports
    ├── aligned_point_clouds/    # generated scene folders
    └── openmask_features/       # cached CLIP features + masks
```

Legacy Spot modules remain in the tree for reference but are unused in this workflow.

## Troubleshooting

- **`FileNotFoundError: OpenMask scene folder`** – run `full_align.py` with `--scan-name` matching the value in `config.yaml`.
- **`OpenMask request timed out`** – ensure the Docker container is running and reachable at `ip:port` from your config. Increase the server timeout if processing large scenes.
- **Empty segmentation** – try raising `--idx` to inspect alternative masks or refine the text query (e.g., "mug on table").

## License

The repository inherits the original Spot-Compose license (see `LICENSE`).
