#!/usr/bin/env python3
"""
oct_to_column.py — OCT thick point cloud → printable column STL pipeline.

Converts an OCT PLY point cloud (as seen in the viewer) into a 3D-printable
relief column via top-surface extraction, polar-grid meshing, and
watertight STL export.

The pipeline reads the same PLY files and applies the exact same
spatial clipping + RGB filter logic as ply_viewer.py, so what you see
in the viewer is exactly what goes into the STL.

Usage:
    uv run python oct_to_column.py --params viewer_params.json
    uv run python oct_to_column.py --ply scan.ply --params viewer_params.json --out column.stl
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
import trimesh
import pyvista as pv

# ---------------------------------------------------------------------------
# 1. RGB → scalar metrics (exact copy from ply_viewer.py)
# ---------------------------------------------------------------------------

FILTER_MODES = ["Intensity", "Hue (warm↔cool)", "Brightness", "Saturation"]


def _rgb_to_hsv_vec(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised RGB → HSV. Returns (hue_0_255, saturation_0_255)."""
    r, g, b = rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    hue = np.zeros(len(rgb))
    mask_d = delta > 0
    mr = mask_d & (cmax == r)
    mg = mask_d & (cmax == g) & ~mr
    mb = mask_d & ~mr & ~mg
    hue[mr] = 60.0 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    hue[mg] = 60.0 * (((b[mg] - r[mg]) / delta[mg]) + 2)
    hue[mb] = 60.0 * (((r[mb] - g[mb]) / delta[mb]) + 4)
    sat = np.where(cmax > 0, delta / cmax, 0.0)
    return hue / 360.0 * 255.0, sat * 255.0


def _rgb_to_metrics(rgb: np.ndarray) -> dict:
    """Compute all four filter metrics from RGB, identical to ply_viewer.py."""
    r, g, b = rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0
    intensity = (r * 0.299 + g * 0.587 + b * 0.114) * 255
    brightness = np.maximum(np.maximum(r, g), b) * 255
    hue, sat = _rgb_to_hsv_vec(rgb)
    return {
        "Intensity": intensity,
        "Hue (warm↔cool)": hue,
        "Brightness": brightness,
        "Saturation": sat,
    }


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

@dataclass
class ColumnConfig:
    """All tunable parameters for the OCT → column pipeline."""

    # Top surface extraction
    grid_size: int = 512
    top_percentile: float = 92.0
    top_window: float = 3.0
    min_points: int = 6

    smoothing_sigma: float = 6.0
    median_filter_size: int = 5

    # Physical Calibration (Default to isotropic 14.7um unless specified)
    x_pixel_size_mm: float = 0.0147       # 14.7 microns/pixel (X)
    y_pixel_size_mm: float = 0.0147       # 14.7 microns/pixel (Y)
    pixel_size_mm: float = 0.0147         # legacy alias for X/Y (used if x/y not set)
    z_pixel_size_mm: float = 0.0147       # 14.7 microns/pixel (Z) - isotropic default
    auto_diameter: bool = True           # set diameter based on ROI

    # Column geometry (mm)
    diameter_mm: float = 40.0            
    base_height_mm: float = 3.0          
    relief_height_mm: float = 5.0        # fallback height
    use_physical_z: bool = True          

    # Mesh & Mode
    ring_segments: int = 512
    n_radial: int = 256
    laplacian_iterations: int = 10         # post-build mesh smoothing passes
    is_complement: bool = False            # Generate a negative (concave) mold

    # Output paths
    output_stl: str = "oct_column.stl"
    output_preview: Optional[str] = None
    output_metadata: Optional[str] = None


# ---------------------------------------------------------------------------
# 3. Data loading
# ---------------------------------------------------------------------------

def load_ply_points(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a PLY point cloud → (points (N,3), rgb (N,3) uint8)."""
    mesh = pv.read(ply_path)
    points = np.asarray(mesh.points).copy()
    rgb = np.asarray(mesh["RGB"]).copy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return points, rgb


def load_viewer_params(params_path: str) -> dict:
    """Load a viewer-params JSON file."""
    with open(params_path, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 4. ROI + colour filter (matches ply_viewer.py exactly)
# ---------------------------------------------------------------------------

def apply_viewer_filter(
    points: np.ndarray,
    rgb: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the exact same spatial + colour filter as the viewer.

    This reproduces the mask logic from ply_viewer.py _rebuild_mesh():
      1. Compute the metric selected by params["filter_mode"]
      2. Apply filter_min / filter_max (0-255 range)
      3. AND with spatial x/y/z clipping
    """
    # --- Colour / intensity filter (exact viewer logic) ---
    metrics = _rgb_to_metrics(rgb)
    filter_mode = params.get("filter_mode", "Intensity")
    lo = params.get("filter_min", 0)
    hi = params.get("filter_max", 255)
    if lo > hi:
        lo, hi = hi, lo

    metric = metrics.get(filter_mode)
    if metric is not None:
        mask = (metric >= lo) & (metric <= hi)
    else:
        mask = np.ones(len(points), dtype=bool)

    # --- Spatial clipping (exact viewer logic) ---
    for ax, ax_min_key, ax_max_key in [
        ("x", "x_min", "x_max"),
        ("y", "y_min", "y_max"),
        ("z", "z_min", "z_max"),
    ]:
        vmin = params.get(ax_min_key)
        vmax = params.get(ax_max_key)
        if vmin is not None and vmax is not None:
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            ax_idx = "xyz".index(ax)
            col = points[:, ax_idx]
            mask &= (col >= vmin) & (col <= vmax)

    return points[mask], rgb[mask]


# ---------------------------------------------------------------------------
# 5. Robust top-Z estimator
# ---------------------------------------------------------------------------

def percentile_top_z(zs: np.ndarray, pct: float) -> float:
    """Return the robust top height for one grid cell via percentile.

    Uses a high percentile (e.g. 90-95) instead of cluster-based extraction.
    With few points per cell (typical in sparse OCT data), cluster methods
    are extremely noisy; percentile estimation is statistically more stable
    and the remaining noise is handled by subsequent Gaussian smoothing.
    """
    if len(zs) == 0:
        return np.nan
    if len(zs) == 1:
        return float(zs[0])
    return float(np.percentile(zs, pct))


def _reject_heightfield_outliers(
    GZ: np.ndarray, mad_threshold: float = 4.0
) -> np.ndarray:
    """Replace cells whose z deviates >mad_threshold*MAD from local median."""
    valid = ~np.isnan(GZ)
    if valid.sum() < 9:
        return GZ
    from scipy.ndimage import median_filter as _mf
    local_med = _mf(np.where(valid, GZ, 0.0), size=5)
    local_med[~valid] = np.nan
    abs_dev = np.abs(GZ - local_med)
    med_abs_dev = float(np.nanmedian(abs_dev[valid]))
    if med_abs_dev < 1e-6:
        return GZ
    outlier = valid & (abs_dev > mad_threshold * 1.4826 * med_abs_dev)
    GZ[outlier] = np.nan
    n_rej = int(outlier.sum())
    if n_rej > 0:
        print(f"  MAD rejected {n_rej} outlier cells "
              f"({100 * n_rej / valid.sum():.1f}%)")
    return GZ


# ---------------------------------------------------------------------------
# 6. Top-surface heightfield extraction
# ---------------------------------------------------------------------------

def extract_top_heightfield(
    points: np.ndarray,
    _extra: np.ndarray,
    config: ColumnConfig,
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Density-aware surface extraction: PLY Y is depth, X/Z are horizontal."""
    N = config.grid_size
    # Mapping: PLY X (0) -> Grid X, PLY Z (2) -> Grid Y, PLY Y (1) -> Height
    x_lateral = points[:, 0]
    y_lateral = points[:, 2] # Z in PLY
    z_depth = points[:, 1]   # Y in PLY (vertical)
    
    # 1. Setup Grid matching the horizontal ROI (X and Z)
    xmin, xmax = x_lateral.min(), x_lateral.max()
    ymin, ymax = y_lateral.min(), y_lateral.max()
    gx = np.linspace(xmin, xmax, N)
    gy = np.linspace(ymin, ymax, N)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.full((N, N), np.nan)
    
    # 2. Use a KDTree for fast radius searching in X/Z plane
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([x_lateral, y_lateral], axis=-1))
    
    # Search radius (0.15mm approx)
    search_radius = 10.0 
    
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    
    batch_size = 10000
    for start in range(0, len(grid_pts), batch_size):
        end = min(start + batch_size, len(grid_pts))
        indices = tree.query_ball_point(grid_pts[start:end], r=search_radius)
        
        for i, idx_list in enumerate(indices):
            if len(idx_list) >= 3:
                # Use mean for a smooth surface trend
                GZ.ravel()[start + i] = np.mean(z_depth[idx_list])
        
        if progress_callback:
            progress_callback(end, len(grid_pts))

    # 3. Minimal Smoothing
    from scipy.ndimage import gaussian_filter
    mask = ~np.isnan(GZ)
    W = mask.astype(float)
    GZ_filled = np.nan_to_num(GZ)
    
    sigma = 1.0
    GZ_smooth = gaussian_filter(GZ_filled * W, sigma=sigma)
    W_smooth = gaussian_filter(W, sigma=sigma)
    GZ = np.divide(GZ_smooth, W_smooth, out=np.full_like(GZ, np.nan), where=W_smooth > 0.1)

    return GX, GY, GZ, None


# ---------------------------------------------------------------------------
# 7. Hole filling & smoothing
# ---------------------------------------------------------------------------

def fill_holes_and_smooth(GZ: np.ndarray, config: ColumnConfig) -> np.ndarray:
    """Aggressively smooth the surface while preserving NaNs at empty areas."""
    if config.smoothing_sigma <= 0:
        return GZ
        
    from scipy.ndimage import gaussian_filter
    
    # 1. Weighted Gaussian: Smooths the data without bleeding into empty space
    mask = ~np.isnan(GZ)
    W = mask.astype(float)
    GZ_filled = np.nan_to_num(GZ)
    
    # Aggressive smoothing: we use the sigma from config
    GZ_smooth = gaussian_filter(GZ_filled * W, sigma=config.smoothing_sigma)
    W_smooth = gaussian_filter(W, sigma=config.smoothing_sigma)
    
    # Only keep results where we have enough weight (original data presence)
    # This prevents 'leaking' the surface into areas with no points
    result = np.divide(GZ_smooth, W_smooth, out=np.full_like(GZ, np.nan), where=W_smooth > 0.3)
    
    return result


# ---------------------------------------------------------------------------
# 8. Column mesh construction (polar grid)
# -----------------------------------------------------------

def build_column_mesh(GX: np.ndarray, GY: np.ndarray, GZ: np.ndarray, config: ColumnConfig) -> trimesh.Trimesh:
    """Build column: relief surface (smoothed) + straight cylinder base (untouched).

    Phase 1 — Build relief top surface as a polar grid, Laplacian-smooth it.
    Phase 2 — Build a perfectly straight cylinder base at base_height_mm.
    Phase 3 — Stitch relief rim → base top ring → bottom.
    """
    N = config.grid_size

    # ── Physical dimensions ──
    gx_min, gx_max = GX.min(), GX.max()
    gy_min, gy_max = GY.min(), GY.max()
    dx_mm = (gx_max - gx_min) * config.x_pixel_size_mm
    dy_mm = (gy_max - gy_min) * config.y_pixel_size_mm
    roi_diag = np.sqrt(dx_mm ** 2 + dy_mm ** 2)

    diameter = roi_diag * 1.1 if config.auto_diameter else config.diameter_mm
    radius = diameter / 2.0

    gz_valid = GZ[~np.isnan(GZ)]
    gz_min_v = float(np.nanmin(gz_valid)) if len(gz_valid) > 0 else 0.0
    gz_max_v = float(np.nanmax(gz_valid)) if len(gz_valid) > 0 else 1.0
    gz_range_v = max(gz_max_v - gz_min_v, 1e-6)
    relief_max_mm = gz_range_v * config.z_pixel_size_mm

    # ── Polar grid setup ──
    M = config.ring_segments
    n_rad = config.n_radial
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    gx0, gy0 = gx_min, gy_min
    gx_step = (gx_max - gx_min) / (N - 1)
    gy_step = (gy_max - gy_min) / (N - 1)

    def get_z_at_mm(px: float, py: float) -> float:
        """Sample relief height from GZ heightfield at physical (mm) coords."""
        gx = (px / config.x_pixel_size_mm) + (gx_min + gx_max) / 2.0
        gy = (py / config.y_pixel_size_mm) + (gy_min + gy_max) / 2.0

        if not (gx_min <= gx <= gx_max and gy_min <= gy <= gy_max):
            return config.base_height_mm

        fi = (gy - gy0) / gy_step
        fj = (gx - gx0) / gx_step
        i0 = int(np.clip(np.floor(fi), 0, N - 2))
        j0 = int(np.clip(np.floor(fj), 0, N - 2))
        di, dj = fi - i0, fj - j0

        v00, v01 = GZ[i0, j0], GZ[i0, j0 + 1]
        v10, v11 = GZ[i0 + 1, j0], GZ[i0 + 1, j0 + 1]

        if any(np.isnan(v) for v in (v00, v01, v10, v11)):
            raw_v = GZ[int(round(fi)), int(round(fj))]
        else:
            raw_v = (v00 * (1 - di) * (1 - dj) + v01 * (1 - di) * dj
                     + v10 * di * (1 - dj) + v11 * di * dj)

        if np.isnan(raw_v):
            return config.base_height_mm

        relief = (gz_max_v - raw_v) * config.z_pixel_size_mm
        if config.is_complement:
            return config.base_height_mm + (relief_max_mm - relief)
        return config.base_height_mm + relief

    # ================================================================
    # Phase 1: Relief top surface (center + n_rad rings, top faces only)
    # ================================================================
    n_relief = 1 + n_rad * M
    rel_verts = np.zeros((n_relief, 3))
    rel_faces = []

    rel_verts[0] = [0, 0, get_z_at_mm(0, 0)]

    for r in range(1, n_rad + 1):
        curr_r = radius * r / n_rad
        base_idx = 1 + (r - 1) * M
        for k in range(M):
            px, py = curr_r * cos_t[k], curr_r * sin_t[k]
            rel_verts[base_idx + k] = [px, py, get_z_at_mm(px, py)]

    # Top fan (center → first ring)
    for k in range(M):
        rel_faces.append([0, 1 + k, 1 + (k + 1) % M])
    # Ring-to-ring quads
    for r in range(1, n_rad):
        i_b = 1 + (r - 1) * M
        o_b = 1 + r * M
        for k in range(M):
            kn = (k + 1) % M
            rel_faces.append([i_b + k, o_b + k, o_b + kn])
            rel_faces.append([i_b + k, o_b + kn, i_b + kn])

    # Laplacian smooth the relief surface ONLY (no base vertices to distort)
    relief_mesh = trimesh.Trimesh(vertices=rel_verts, faces=rel_faces, process=False)
    if config.laplacian_iterations > 0:
        relief_mesh = trimesh.smoothing.filter_laplacian(
            relief_mesh, iterations=config.laplacian_iterations, lamb=0.5,
        )

    # ================================================================
    # Phase 2: Straight cylinder base + stitch to smoothed relief
    # ================================================================
    # Vertex layout:
    #   [0 .. n_relief-1]          smoothed relief top
    #   [n_relief .. n_relief+M-1] base top ring (Z = base_height_mm, flat)
    #   [n_relief+M .. n_relief+2M-1] bottom ring (Z = 0)
    #   [n_relief+2M]              bottom center (Z = 0)

    rim_start = n_relief
    bot_start = rim_start + M
    bot_center_idx = bot_start + M
    total_verts = bot_center_idx + 1

    verts = np.zeros((total_verts, 3))
    faces = []

    verts[:n_relief] = relief_mesh.vertices

    # Base top ring: same XY as column radius, constant Z
    rim_z = config.base_height_mm + (relief_max_mm if config.is_complement else 0.0)
    for k in range(M):
        verts[rim_start + k] = [radius * cos_t[k], radius * sin_t[k], rim_z]

    # Bottom ring: Z = 0
    for k in range(M):
        verts[bot_start + k] = [radius * cos_t[k], radius * sin_t[k], 0.0]

    # Bottom center
    verts[bot_center_idx] = [0, 0, 0.0]

    # ── Faces ──

    # Relief top faces (reuse from phase 1, already correct indices)
    faces.extend(relief_mesh.faces.tolist())

    # Relief rim → base top ring (transition strip)
    relief_rim_start = 1 + (n_rad - 1) * M
    for k in range(M):
        kn = (k + 1) % M
        faces.append([relief_rim_start + k, rim_start + k, rim_start + kn])
        faces.append([relief_rim_start + k, rim_start + kn, relief_rim_start + kn])

    # Vertical sides: base top ring → bottom ring (perfectly straight cylinder)
    for k in range(M):
        kn = (k + 1) % M
        faces.append([rim_start + k, bot_start + k, bot_start + kn])
        faces.append([rim_start + k, bot_start + kn, rim_start + kn])

    # Bottom disk
    for k in range(M):
        faces.append([bot_center_idx, bot_start + (k + 1) % M, bot_start + k])

    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


# ---------------------------------------------------------------------------
# 9. Mesh validation
# ---------------------------------------------------------------------------

def validate_mesh(mesh: trimesh.Trimesh) -> dict:
    """Check mesh for 3D-printability and return a report dict."""
    issues = []

    if not mesh.is_watertight:
        issues.append("Mesh is NOT watertight — may fail slicing.")
    if not mesh.is_winding_consistent:
        issues.append("Face winding is inconsistent.")
    euler = mesh.euler_number
    if euler != 2:
        issues.append(
            f"Euler number is {euler} (expected 2 for a closed sphere-like"
            " solid). There may be internal topology."
        )

    bounds = mesh.bounds
    volume = None
    try:
        volume = float(mesh.volume)
    except Exception:
        issues.append("Could not compute volume (mesh may be non-manifold).")

    is_printable = mesh.is_watertight and mesh.is_winding_consistent and euler == 2

    return {
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(euler),
        "volume": volume,
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "bounds_min": [float(v) for v in bounds[0]],
        "bounds_max": [float(v) for v in bounds[1]],
        "issues": issues,
        "is_printable": is_printable,
    }


# ---------------------------------------------------------------------------
# 10. Mesh repair
# ---------------------------------------------------------------------------

def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Attempt automatic repair of a non-watertight mesh."""
    mesh.process(validate=True)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        mesh.process(validate=True)
    return mesh


# ---------------------------------------------------------------------------
# 11. Preview generation
# ---------------------------------------------------------------------------

def generate_preview(
    mesh: trimesh.Trimesh,
    output_path: str,
    title: str = "OCT Column",
) -> None:
    pv_mesh = pv.wrap(mesh)
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("#1a1a2e")
    plotter.add_mesh(
        pv_mesh,
        color="lightsteelblue",
        smooth_shading=True,
    )
    plotter.add_text(title, position="upper_left", font_size=10, color="white")
    plotter.camera_position = "iso"
    plotter.screenshot(output_path)
    plotter.close()
    print(f"  [preview] Saved → {output_path}")


# ---------------------------------------------------------------------------
# 12. Metadata export
# ---------------------------------------------------------------------------

def save_metadata(
    config: ColumnConfig,
    params: dict,
    mesh_report: dict,
    output_path: str,
    duration: float,
) -> None:
    """Write a JSON file with all pipeline parameters and results."""
    meta = {
        "source_ply": params.get("scan_file", ""),
        "filter_mode": params.get("filter_mode", ""),
        "filter_range": [params.get("filter_min", 0), params.get("filter_max", 255)],
        "grid_size": config.grid_size,
        "top_window": config.top_window,
        "min_points": config.min_points,
        "smooth_sigma": config.smoothing_sigma,
        "median_filter_size": config.median_filter_size,
        "diameter_mm": config.diameter_mm,
        "base_height_mm": config.base_height_mm,
        "relief_height_mm": config.relief_height_mm,
        "ring_segments": config.ring_segments,
        "n_radial": config.n_radial,
        "mesh": mesh_report,
        "processing_time_seconds": round(duration, 3),
    }
    with open(output_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  [metadata] Saved → {output_path}")


# ---------------------------------------------------------------------------
# 13. Pipeline orchestrator
# ---------------------------------------------------------------------------

def pipeline(
    ply_path: str,
    params_path: str,
    config: ColumnConfig,
) -> Tuple[trimesh.Trimesh, dict]:
    t0 = time.time()

    # [1/7] Loading data
    print("[1/7] Loading data...")
    print(f"  PLY:    {ply_path}")
    print(f"  Params: {params_path}")
    points, rgb = load_ply_points(ply_path)
    params = load_viewer_params(params_path)
    print(f"  Points: {points.shape[0]:,}")

    # [2/7] Apply viewer filter (spatial + colour, matches ply_viewer.py)
    filter_mode = params.get("filter_mode", "Intensity")
    filter_lo = params.get("filter_min", 0)
    filter_hi = params.get("filter_max", 255)
    print(f"[2/7] Applying viewer filter...")
    print(f"  Filter: {filter_mode} [{filter_lo}, {filter_hi}]")
    print(f"  ROI  x=[{params['x_min']:.1f}, {params['x_max']:.1f}]  "
          f"y=[{params['y_min']:.1f}, {params['y_max']:.1f}]  "
          f"z=[{params['z_min']:.1f}, {params['z_max']:.1f}]")
    points, rgb = apply_viewer_filter(points, rgb, params)
    print(f"  After filter: {points.shape[0]:,} points")
    if points.shape[0] < 100:
        raise ValueError(
            f"Only {points.shape[0]} points remain after filtering. "
            "Need at least 100 to build a surface."
        )

    # [3/7] Extracting top surface
    print("[3/7] Extracting top surface...")

    def _progress(done, total):
        if done == 0:
            print(f"  Processing ~{total:,} populated cells...")

    GX, GY, GZ, density = extract_top_heightfield(
        points, rgb, config, progress_callback=_progress,
    )
    n_valid = int(np.sum(~np.isnan(GZ)))
    print(f"  Valid cells: {n_valid:,} / {config.grid_size**2:,}  "
          f"({100 * n_valid / config.grid_size**2:.1f} %)")
    if n_valid < 4:
        raise ValueError(
            f"Only {n_valid} valid heightfield cells — cannot proceed."
        )

    # [4/7] Smoothing
    print("[4/7] Smoothing...")
    GZ_smooth = fill_holes_and_smooth(GZ, config)
    print(f"  Z range after smoothing: [{np.nanmin(GZ_smooth):.2f}, {np.nanmax(GZ_smooth):.2f}]")

    # [5/7] Building column mesh
    print("[5/7] Building column mesh...")
    mesh = build_column_mesh(GX, GY, GZ_smooth, config)
    print(f"  Vertices: {mesh.vertices.shape[0]:,}  |  Faces: {mesh.faces.shape[0]:,}")

    # [6/7] Validating mesh
    print("[6/7] Validating mesh...")
    report = validate_mesh(mesh)
    if not report["is_watertight"]:
        print("  ⚠  Mesh not watertight — attempting repair...")
        mesh = repair_mesh(mesh)
        report = validate_mesh(mesh)
    print(f"  Watertight: {report['is_watertight']}  |  "
          f"Printable: {report['is_printable']}  |  "
          f"Euler: {report['euler_number']}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  ⚠  {issue}")

    # [7/7] Exporting
    print("[7/7] Exporting...")
    mesh.export(config.output_stl)
    print(f"  STL → {config.output_stl}")

    if config.output_preview:
        generate_preview(mesh, config.output_preview)

    duration = time.time() - t0
    if config.output_metadata:
        save_metadata(config, params, report, config.output_metadata, duration)

    print(f"\nDone in {duration:.1f}s")
    return mesh, report


def pipeline_from_params(
    ply_path: str,
    params: dict,
    config: ColumnConfig,
) -> Tuple[trimesh.Trimesh, dict]:
    t0 = time.time()

    # [1/7] Loading data
    print("[1/7] Loading data...")
    print(f"  PLY:    {ply_path}")
    points, rgb = load_ply_points(ply_path)
    print(f"  Points: {points.shape[0]:,}")

    # [2/7] Apply viewer filter
    filter_mode = params.get("filter_mode", "Intensity")
    filter_lo = params.get("filter_min", 0)
    filter_hi = params.get("filter_max", 255)
    print(f"[2/7] Applying viewer filter...")
    print(f"  Filter: {filter_mode} [{filter_lo}, {filter_hi}]")
    x_min = params.get('x_min', 0)
    x_max = params.get('x_max', 511)
    y_min = params.get('y_min', 0)
    y_max = params.get('y_max', 511)
    z_min = params.get('z_min', 0)
    z_max = params.get('z_max', 511)
    print(f"  ROI  x=[{x_min:.1f}, {x_max:.1f}]  "
          f"y=[{y_min:.1f}, {y_max:.1f}]  "
          f"z=[{z_min:.1f}, {z_max:.1f}]")
    points, rgb = apply_viewer_filter(points, rgb, params)
    print(f"  After filter: {points.shape[0]:,} points")
    if points.shape[0] < 100:
        raise ValueError(
            f"Only {points.shape[0]} points remain after filtering. "
            "Need at least 100 to build a surface."
        )

    # [3/7] Extracting top surface
    print("[3/7] Extracting top surface...")

    def _progress(done, total):
        if done == 0:
            print(f"  Processing ~{total:,} populated cells...")

    GX, GY, GZ, density = extract_top_heightfield(
        points, rgb, config, progress_callback=_progress,
    )
    n_valid = int(np.sum(~np.isnan(GZ)))
    print(f"  Valid cells: {n_valid:,} / {config.grid_size**2:,}  "
          f"({100 * n_valid / config.grid_size**2:.1f} %)")
    if n_valid < 4:
        raise ValueError(
            f"Only {n_valid} valid heightfield cells — cannot proceed."
        )

    # [4/7] Smoothing
    print("[4/7] Smoothing...")
    GZ_smooth = fill_holes_and_smooth(GZ, config)
    print(f"  Z range after smoothing: [{np.nanmin(GZ_smooth):.2f}, {np.nanmax(GZ_smooth):.2f}]")

    # [5/7] Building column mesh
    print("[5/7] Building column mesh...")
    mesh = build_column_mesh(GX, GY, GZ_smooth, config)
    print(f"  Vertices: {mesh.vertices.shape[0]:,}  |  Faces: {mesh.faces.shape[0]:,}")

    # [6/7] Validating mesh
    print("[6/7] Validating mesh...")
    report = validate_mesh(mesh)
    if not report["is_watertight"]:
        print("  ⚠  Mesh not watertight — attempting repair...")
        mesh = repair_mesh(mesh)
        report = validate_mesh(mesh)
    print(f"  Watertight: {report['is_watertight']}  |  "
          f"Printable: {report['is_printable']}  |  "
          f"Euler: {report['euler_number']}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  ⚠  {issue}")

    # [7/7] Exporting
    print("[7/7] Exporting...")
    mesh.export(config.output_stl)
    print(f"  STL → {config.output_stl}")

    if config.output_preview:
        generate_preview(mesh, config.output_preview)

    duration = time.time() - t0
    if config.output_metadata:
        save_metadata(config, params, report, config.output_metadata, duration)

    print(f"\nDone in {duration:.1f}s")
    return mesh, report


# ---------------------------------------------------------------------------
# 14. CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCT point cloud → printable column STL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  uv run python oct_to_column.py \\\n"
            "    --params viewer_params_....json \\\n"
            "    --out oct_column.stl --preview\n"
        ),
    )

    parser.add_argument("--ply", default=None,
                        help="Path to PLY file (auto-derived from params if omitted)")
    parser.add_argument("--params", required=True, help="Path to viewer params JSON")
    parser.add_argument("--out", default="oct_column.stl",
                        help="Output STL path (default: oct_column.stl)")

    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--top-percentile", type=float, default=92.0)
    parser.add_argument("--top-window", type=float, default=3.0)
    parser.add_argument("--min-points", type=int, default=6)

    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    parser.add_argument("--median-size", type=int, default=5)
    parser.add_argument("--smooth-iters", "--laplacian", type=int, default=10, dest="laplacian")

    # Calibration
    parser.add_argument("--pixel-size-mm", type=float, default=0.0147, help="Lateral pixel size (X/Y)")
    parser.add_argument("--z-pixel-size-mm", type=float, default=0.004, help="Axial pixel size (Z/Depth)")
    parser.add_argument("--no-auto-diameter", action="store_false", dest="auto_diameter")
    parser.add_argument("--physical-z", action="store_true", default=True)

    parser.add_argument("--diameter-mm", type=float, default=40.0)
    parser.add_argument("--base-height-mm", type=float, default=3.0)
    parser.add_argument("--relief-height-mm", type=float, default=5.0)
    parser.add_argument("--ring-segments", type=int, default=512)
    parser.add_argument("--n-radial", type=int, default=256)

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--complement", action="store_true", help="Generate the negative (concave) fit")

    args = parser.parse_args()

    params = load_viewer_params(args.params)

    ply_path = args.ply
    if not ply_path:
        ply_path = params.get("scan_file", "")
    if not os.path.isfile(ply_path):
        print(f"Error: PLY not found: {ply_path}", file=sys.stderr)
        sys.exit(1)

    config = ColumnConfig(
        grid_size=args.grid_size,
        top_percentile=args.top_percentile,
        top_window=args.top_window,
        min_points=args.min_points,
        smoothing_sigma=args.smooth_sigma,
        median_filter_size=args.median_size,
        laplacian_iterations=args.laplacian,
        pixel_size_mm=args.pixel_size_mm,
        z_pixel_size_mm=args.z_pixel_size_mm,
        auto_diameter=args.auto_diameter,
        diameter_mm=args.diameter_mm,
        base_height_mm=args.base_height_mm,
        relief_height_mm=args.relief_height_mm,
        use_physical_z=args.physical_z,
        ring_segments=args.ring_segments,
        n_radial=args.n_radial,
        is_complement=args.complement,
        output_stl=args.out,
        output_preview=args.out.replace(".stl", ".png") if args.preview else None,
        output_metadata=args.out.replace(".stl", "_meta.json"),
    )
    
    # Auto-suffix for complement
    if args.complement and "_complement" not in config.output_stl:
        config.output_stl = config.output_stl.replace(".stl", "_complement.stl")
        if config.output_preview:
            config.output_preview = config.output_preview.replace(".png", "_complement.png")

    pipeline(ply_path, args.params, config)


if __name__ == "__main__":
    main()
