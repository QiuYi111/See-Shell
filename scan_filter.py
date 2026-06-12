#!/usr/bin/env python3
"""Point cloud filter module for See-Shell.

Owns RGB→metric computation, spatial clipping, range masking,
and parameter serialization. Used by both the interactive viewer
and the headless OCT→column pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

FILTER_MODES = ["Intensity", "Hue (warm↔cool)", "Brightness", "Saturation"]

# ---------------------------------------------------------------------------
# RGB → metric helpers
# ---------------------------------------------------------------------------


def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def rgb_to_metrics(rgb: np.ndarray) -> dict:
    """Compute all four filter metrics from an (N,3) uint8 RGB array."""
    r, g, b = rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0
    intensity = (r * 0.299 + g * 0.587 + b * 0.114) * 255
    brightness = np.maximum(np.maximum(r, g), b) * 255
    hue, sat = rgb_to_hsv(rgb)
    return {
        "Intensity": intensity,
        "Hue (warm↔cool)": hue,
        "Brightness": brightness,
        "Saturation": sat,
    }


# ---------------------------------------------------------------------------
# ScanFilter — owns filter state, bounds, and mask computation
# ---------------------------------------------------------------------------


class ScanFilter:
    """Stateful point-cloud filter.

    Usage::

        f = ScanFilter()
        f.set_bounds(points)          # once, after loading a scan
        f.update(mode="Hue (warm↔cool)", filter_min=30, filter_max=200)
        filtered_pts, filtered_rgb = f.apply(points, rgb)
        params_dict = f.to_params(scan_file="scan.ply")
    """

    def __init__(self) -> None:
        # Filter mode + range
        self.mode: str = FILTER_MODES[0]
        self.filter_min: int = 0
        self.filter_max: int = 255

        # Spatial bounds (point-cloud units), set via set_bounds()
        self._bounds: dict[str, tuple[float, float]] = {}

        # Spatial clipping ranges (point-cloud units)
        self.x_min: Optional[float] = None
        self.x_max: Optional[float] = None
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None
        self.z_min: Optional[float] = None
        self.z_max: Optional[float] = None

    # ── Bounds ──

    def set_bounds(self, points: np.ndarray) -> None:
        """Store axis min/max from a raw point cloud."""
        self._bounds = {}
        for i, ax in enumerate("xyz"):
            col = points[:, i]
            self._bounds[ax] = (float(col.min()), float(col.max()))

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._bounds)

    # ── State update ──

    def update(
        self,
        *,
        mode: Optional[str] = None,
        filter_min: Optional[int] = None,
        filter_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> None:
        """Batch-update filter state. Only passed kwargs are changed."""
        if mode is not None:
            self.mode = mode
        if filter_min is not None:
            self.filter_min = filter_min
        if filter_max is not None:
            self.filter_max = filter_max
        if x_min is not None:
            self.x_min = x_min
        if x_max is not None:
            self.x_max = x_max
        if y_min is not None:
            self.y_min = y_min
        if y_max is not None:
            self.y_max = y_max
        if z_min is not None:
            self.z_min = z_min
        if z_max is not None:
            self.z_max = z_max

    # ── Core: apply filter ──

    def apply(
        self, points: np.ndarray, rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply metric + spatial filter. Returns (filtered_pts, filtered_rgb)."""
        # Metric filter
        metrics = rgb_to_metrics(rgb)
        lo, hi = self.filter_min, self.filter_max
        if lo > hi:
            lo, hi = hi, lo

        metric = metrics.get(self.mode)
        mask = (metric >= lo) & (metric <= hi) if metric is not None else np.ones(len(points), dtype=bool)

        # Spatial clipping
        for ax, ax_min, ax_max in [
            ("x", self.x_min, self.x_max),
            ("y", self.y_min, self.y_max),
            ("z", self.z_min, self.z_max),
        ]:
            vmin, vmax = ax_min, ax_max
            if vmin is not None and vmax is not None:
                if vmin > vmax:
                    vmin, vmax = vmax, vmin
                ax_idx = "xyz".index(ax)
                col = points[:, ax_idx]
                mask &= (col >= vmin) & (col <= vmax)

        return points[mask], rgb[mask]

    # ── Serialization ──

    def to_params(self, **extra: object) -> dict:
        """Serialize filter state to a JSON-friendly dict.

        Pass extra kwargs to include scan metadata (e.g. scan_file, scan_label).
        """
        lo, hi = self.filter_min, self.filter_max
        if lo > hi:
            lo, hi = hi, lo
        params: dict = {
            "filter_mode": self.mode,
            "filter_min": lo,
            "filter_max": hi,
        }
        for ax in "xyz":
            vmin = getattr(self, f"{ax}_min")
            vmax = getattr(self, f"{ax}_max")
            if vmin is not None and vmax is not None:
                if vmin > vmax:
                    vmin, vmax = vmax, vmin
            params[f"{ax}_min"] = round(vmin, 4) if vmin is not None else None
            params[f"{ax}_max"] = round(vmax, 4) if vmax is not None else None
        params.update(extra)
        return params

    @staticmethod
    def from_params(params: dict) -> ScanFilter:
        """Construct a ScanFilter from a params dict (e.g. loaded from JSON)."""
        f = ScanFilter()
        f.update(
            mode=params.get("filter_mode", FILTER_MODES[0]),
            filter_min=params.get("filter_min", 0),
            filter_max=params.get("filter_max", 255),
            x_min=params.get("x_min"),
            x_max=params.get("x_max"),
            y_min=params.get("y_min"),
            y_max=params.get("y_max"),
            z_min=params.get("z_min"),
            z_max=params.get("z_max"),
        )
        return f
