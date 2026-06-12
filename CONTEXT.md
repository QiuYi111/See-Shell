# See-Shell — Domain Glossary

## Core Concepts

**Scan**
A single OCT (Optical Coherence Tomography) volumetric capture, stored as a PLY point cloud file. Each scan contains 3D coordinates (X, Y, Z) and per-point RGB color.

**Filter**
The combination of metric selection, metric range, and spatial clipping applied to a scan. Implemented by the `ScanFilter` module (`scan_filter.py`). A filter produces a masked subset of the original points.

**Metric**
A scalar value derived from per-point RGB: Intensity (luminance), Hue (warm↔cool angle), Brightness (max channel), or Saturation (color purity). Used to select structurally meaningful regions of the scan.

**Viewer**
The interactive PyQt6 application (`ply_viewer.py`) that displays scans, lets the user adjust filters via sliders, and triggers column generation.

**Pipeline**
The headless PLY→STL conversion process (`oct_to_column.py`). Takes a PLY file, filter parameters, and column configuration; produces a watertight STL suitable for 3D printing.

**Column**
A 3D-printable cylindrical relief derived from a filtered scan's top-surface heightfield. Generated as a polar mesh with configurable diameter, base height, and smoothing.

**ColumnConfig**
A dataclass holding all tunable parameters for the pipeline: grid resolution, smoothing sigma, physical calibration (pixel sizes), column geometry, and output paths.

**ScanFilter**
The module (`scan_filter.py`) that owns filter state, axis bounds, metric computation, and mask application. Shared between the Viewer and the Pipeline — both cross the same seam.

**DataSource**
A loading strategy for scans. Local: scan a directory for PLY files. Remote: browse and download via SSH. The Viewer picks a data source through the `DataSourceDialog`.

## File Map

| File | Role |
|------|------|
| `scan_filter.py` | Filter model — metrics, spatial clipping, serialization |
| `ply_viewer.py` | Interactive viewer — Qt GUI, renders filtered point clouds |
| `oct_to_column.py` | Headless pipeline — PLY → heightfield → polar mesh → STL |
| `data_loader.py` | Data source picker dialog — local folder or SSH |
| `ssh_utils.py` | SSH operations — config parsing, remote listing, SCP download |

## Data Flow

```
PLY file → load → filter (ScanFilter) → view (Viewer) / pipeline (Column)
                                                     ↓
                                              STL file (3D-printable)
```
