# See-Shell

OCT volumetric point cloud viewer with STL column generation. Browse local or remote scans, filter by intensity/hue/space, and export 3D-printable relief columns.

## Quick Start

```bash
uv sync
uv run python ply_viewer.py
```

## Features

- **Point cloud viewer** — PLY files rendered as RGB point clouds with real-time filtering (intensity, hue, brightness, saturation) and spatial clipping (X/Y/Z range sliders)
- **Remote SSH browsing** — load scans from any SSH host configured in `~/.ssh/config`. Individual PLY files are downloaded just-in-time on click and cached locally
- **STL column generation** — convert any scan to a 3D-printable relief column with configurable pixel size, smoothing, and base height. Supports normal + complement pair generation
- **Parameter export** — save/viewer state (filters, ranges, settings) as JSON for batch processing

## File Overview

| File | Purpose |
|------|---------|
| `ply_viewer.py` | Main GUI — scan gallery, 3D viewer, filters, column generation |
| `data_loader.py` | Data source picker — local folder or remote SSH directory |
| `ssh_utils.py` | SSH operations — parse config, list dirs, find files, download |
| `oct_to_column.py` | CLI pipeline — PLY → surface extraction → polar mesh → STL |
| `pyproject.toml` | Dependencies (numpy, pyvista, pyvistaqt, pyqt6, vtk, scipy, trimesh) |

## Remote Data Setup

1. Ensure your SSH host is in `~/.ssh/config` (e.g. `jingyi-lab`)
2. Launch viewer → click **📂 Load Data…**
3. Switch to **Remote (SSH)** tab → select host → enter path → Browse
4. Click a scan to auto-download and view it

Downloaded files cache at `/tmp/see-shell/{host}/` — no re-downloads.

## Keyboard Shortcuts

None yet — everything is point-and-click.

## Requirements

- Python ≥ 3.10
- PyQt6, PyVista, VTK
- SSH client (for remote loading) — uses system `ssh`/`scp`, no Python SSH deps
