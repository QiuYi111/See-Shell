#!/usr/bin/env python3
"""OCT Volume Viewer — Professional volumetric OCT data visualization.

2×2 multi-view layout: 3D volume render, axial B-scan, coronal B-scan, en-face MIP.
Right-side control panel with scan navigation, window/level, opacity, colormap, z-slab,
slice position, and blending mode controls.
"""

import gc
import glob
import os
import re

os.environ["QT_API"] = "pyqt6"

import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QPalette, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QGroupBox,
    QSizePolicy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_DIRS = [
    {
        "path": os.path.join(BASE_DIR, "oct_cloud", "scan_array_20260324_153718"),
        "group": "3x3",
        "grid_size": 3,
    },
    {
        "path": os.path.join(BASE_DIR, "oct_cloud", "scan_array_20260324_235201"),
        "group": "5x5",
        "grid_size": 5,
    },
]
VOL_SHAPE = (512, 512, 512)
DEFAULT_CMAP = "bone"
DEFAULT_OPACITY = "sigmoid_10"
DEFAULT_BLENDING = "composite"
DEFAULT_BG = "#0D0D14"

# ---------------------------------------------------------------------------
# Data Layer
# ---------------------------------------------------------------------------

_GRID_POS_RE = re.compile(r"_(p?\d+)_?(n?\d+)?$")


def _parse_pos_component(tok: str) -> int:
    """Parse a grid-position token like 'p2', 'n1', '0' into an int."""
    if tok.startswith("p"):
        return int(tok[1:])
    if tok.startswith("n"):
        return -int(tok[1:])
    return int(tok)


def collect_scans() -> list[dict]:
    """Glob NPY files from the two scan-array directories.

    Returns a sorted list of dicts with keys: group, suffix, path.
    """
    scans: list[dict] = []
    for info in SCAN_DIRS:
        pattern = os.path.join(info["path"], "volume_pointcloud_*_*.npy")
        for fpath in sorted(glob.glob(pattern)):
            if os.path.getsize(fpath) == 0:
                continue
            basename = os.path.basename(fpath)
            # Filename: volume_pointcloud_YYYYMMDD_HHMMSS_POS1_POS2.npy
            name = basename[: -len(".npy")][len("volume_pointcloud_"):]
            parts = name.split("_")
            pos_tokens = parts[2:]
            suffix = "_".join(pos_tokens)
            scans.append(
                {
                    "group": info["group"],
                    "suffix": suffix,
                    "path": fpath,
                }
            )
    return scans


def load_volume(path: str) -> tuple[pv.ImageData, np.ndarray]:
    """Load NPY point cloud and reconstruct a 512³ volume.

    Returns (pyvista_grid, numpy_volume).
    """
    data = np.load(path)  # (500000, 4): x, y, z, intensity
    vol = np.zeros(VOL_SHAPE, dtype=np.float32)
    x = data[:, 0].astype(np.intp)
    y = data[:, 1].astype(np.intp)
    z = data[:, 2].astype(np.intp)
    vol[x, y, z] = data[:, 3]
    del data
    gc.collect()

    grid = pv.ImageData(dimensions=(VOL_SHAPE[0] + 1, VOL_SHAPE[1] + 1, VOL_SHAPE[2] + 1))
    grid.cell_data["intensity"] = vol.ravel(order="F")
    return grid, vol


# ---------------------------------------------------------------------------
# Dark theme palette
# ---------------------------------------------------------------------------

def apply_dark_palette(app: QApplication) -> None:
    """Apply a professional dark theme to the application."""
    palette = QPalette()
    c_bg = QColor("#1E1E2E")
    c_panel = QColor("#2A2A3C")
    c_text = QColor("#CDD6F4")
    c_dim = QColor("#A6ADC8")
    c_base = QColor("#313244")
    c_button = QColor("#585B70")
    c_highlight = QColor("#89B4FA")
    c_window = QColor("#181825")

    palette.setColor(QPalette.ColorRole.Window, c_bg)
    palette.setColor(QPalette.ColorRole.WindowText, c_text)
    palette.setColor(QPalette.ColorRole.Base, c_base)
    palette.setColor(QPalette.ColorRole.AlternateBase, c_panel)
    palette.setColor(QPalette.ColorRole.ToolTipBase, c_panel)
    palette.setColor(QPalette.ColorRole.ToolTipText, c_text)
    palette.setColor(QPalette.ColorRole.Text, c_text)
    palette.setColor(QPalette.ColorRole.Button, c_button)
    palette.setColor(QPalette.ColorRole.ButtonText, c_text)
    palette.setColor(QPalette.ColorRole.BrightText, c_highlight)
    palette.setColor(QPalette.ColorRole.Link, c_highlight)
    palette.setColor(QPalette.ColorRole.Highlight, c_highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#1E1E2E"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c_dim)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, c_dim)

    app.setPalette(palette)
    app.setStyleSheet(
        """
        QToolTip { background: #2A2A3C; color: #CDD6F4; border: 1px solid #45475A; }
        QGroupBox {
            border: 1px solid #45475A; border-radius: 4px;
            margin-top: 8px; padding-top: 14px;
            font-weight: bold; color: #CDD6F4;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QSlider::groove:horizontal {
            height: 6px; background: #45475A; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 14px; height: 14px; margin: -4px 0;
            background: #89B4FA; border-radius: 7px;
        }
        QComboBox { padding: 4px 8px; border: 1px solid #45475A; border-radius: 3px; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background: #2A2A3C; color: #CDD6F4; selection-background-color: #45475A;
        }
        QPushButton {
            padding: 5px 12px; border: 1px solid #45475A; border-radius: 3px;
            background: #585B70; color: #CDD6F4;
        }
        QPushButton:hover { background: #6C7086; }
        QPushButton:pressed { background: #45475A; }
    """
    )


# ---------------------------------------------------------------------------
# Control Panel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    """Right-side panel with all visualization controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(290)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Scan navigation ---
        grp_scan = QGroupBox("Scan")
        gl = QVBoxLayout(grp_scan)
        self.scan_combo = QComboBox()
        gl.addWidget(self.scan_combo)
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        gl.addLayout(nav_row)
        layout.addWidget(grp_scan)

        # --- Window / Level ---
        grp_wl = QGroupBox("Window / Level")
        wl = QVBoxLayout(grp_wl)
        self.lbl_window = QLabel("Window: 1.000")
        self.slider_window = self._make_slider(0, 1000, 1000)
        wl.addWidget(self.lbl_window)
        wl.addWidget(self.slider_window)
        self.lbl_level = QLabel("Level: 0.000")
        self.slider_level = self._make_slider(0, 1000, 0)
        wl.addWidget(self.lbl_level)
        wl.addWidget(self.slider_level)
        layout.addWidget(grp_wl)

        # --- Opacity ---
        grp_opa = QGroupBox("Opacity")
        ol = QVBoxLayout(grp_opa)
        ol.addWidget(QLabel("Preset:"))
        self.combo_opacity = QComboBox()
        self.combo_opacity.addItems(
            ["sigmoid_6", "sigmoid_10", "sigmoid_15", "linear", "sharp"]
        )
        self.combo_opacity.setCurrentText(DEFAULT_OPACITY)
        ol.addWidget(self.combo_opacity)
        self.lbl_strength = QLabel("Strength: 50")
        self.slider_strength = self._make_slider(1, 100, 50)
        ol.addWidget(self.lbl_strength)
        ol.addWidget(self.slider_strength)
        layout.addWidget(grp_opa)

        # --- Colormap ---
        grp_cm = QGroupBox("Colormap")
        cl = QVBoxLayout(grp_cm)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(
            ["bone", "gray", "viridis", "plasma", "coolwarm", "hot", "cividis", "magma"]
        )
        self.combo_cmap.setCurrentText(DEFAULT_CMAP)
        cl.addWidget(self.combo_cmap)
        layout.addWidget(grp_cm)

        # --- Z-Slab ---
        grp_slab = QGroupBox("Z-Slab")
        sl = QVBoxLayout(grp_slab)
        self.lbl_zmin = QLabel("Z Min: 0")
        self.slider_zmin = self._make_slider(0, 511, 0)
        sl.addWidget(self.lbl_zmin)
        sl.addWidget(self.slider_zmin)
        self.lbl_zmax = QLabel("Z Max: 511")
        self.slider_zmax = self._make_slider(0, 511, 511)
        sl.addWidget(self.lbl_zmax)
        sl.addWidget(self.slider_zmax)
        layout.addWidget(grp_slab)

        # --- Slices ---
        grp_slice = QGroupBox("Slice Position")
        spl = QVBoxLayout(grp_slice)
        self.lbl_slice_z = QLabel("Slice Z (axial): 256")
        self.slider_slice_z = self._make_slider(0, 511, 256)
        spl.addWidget(self.lbl_slice_z)
        spl.addWidget(self.slider_slice_z)
        self.lbl_slice_y = QLabel("Slice Y (coronal): 256")
        self.slider_slice_y = self._make_slider(0, 511, 256)
        spl.addWidget(self.lbl_slice_y)
        spl.addWidget(self.slider_slice_y)
        layout.addWidget(grp_slice)

        # --- Blending ---
        grp_blend = QGroupBox("Blending")
        bl = QVBoxLayout(grp_blend)
        self.combo_blending = QComboBox()
        self.combo_blending.addItems(["composite", "maximum"])
        self.combo_blending.setCurrentText(DEFAULT_BLENDING)
        bl.addWidget(self.combo_blending)
        layout.addWidget(grp_blend)

        # --- Background ---
        grp_bg = QGroupBox("Background")
        bgl = QVBoxLayout(grp_bg)
        self.btn_bg = QPushButton("Choose Background Color")
        bgl.addWidget(self.btn_bg)
        layout.addWidget(grp_bg)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def _make_slider(lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class OCTViewer(QMainWindow):
    """Main application window with 2×2 multi-view layout + control panel."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCT Volume Viewer")
        self.resize(1600, 1000)

        # --- Data state ---
        self.scans = collect_scans()
        self.current_idx = 0
        self.pv_grid: pv.ImageData | None = None
        self.np_vol: np.ndarray | None = None
        self.bg_color = DEFAULT_BG

        # Actor references
        self.vol_actor = None
        self.axial_actor = None
        self.coronal_actor = None
        self.enface_actor = None

        # --- Build UI ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left: 2×2 view grid
        view_widget = QWidget()
        view_grid = QGridLayout(view_widget)
        view_grid.setSpacing(2)
        view_grid.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(view_widget)

        # Create 4 plotters
        self.plotter_3d = pvqt.QtInteractor(view_widget)
        self.plotter_axial = pvqt.QtInteractor(view_widget)
        self.plotter_coronal = pvqt.QtInteractor(view_widget)
        self.plotter_enface = pvqt.QtInteractor(view_widget)

        # Label each view
        for plotter, label, row, col in [
            (self.plotter_3d, "3D Volume", 0, 0),
            (self.plotter_axial, "Axial B-Scan (XY)", 0, 1),
            (self.plotter_coronal, "Coronal B-Scan (XZ)", 1, 0),
            (self.plotter_enface, "En-Face MIP", 1, 1),
        ]:
            view_grid.addWidget(plotter, row, col)
            plotter.set_background(self.bg_color)
            # Add text label overlay
            plotter.add_text(label, position="upper_left", font_size=10, color="white", name="_label")

        # Right: Control panel
        self.ctrl = ControlPanel()
        splitter.addWidget(self.ctrl)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # --- Populate scan combo ---
        for scan in self.scans:
            self.ctrl.scan_combo.addItem(f"{scan['group']} {scan['suffix']}", scan)

        # --- Wire signals ---
        self.ctrl.scan_combo.currentIndexChanged.connect(self._on_scan_changed)
        self.ctrl.btn_prev.clicked.connect(self._go_prev)
        self.ctrl.btn_next.clicked.connect(self._go_next)
        self.ctrl.slider_window.valueChanged.connect(self._on_window_level)
        self.ctrl.slider_level.valueChanged.connect(self._on_window_level)
        self.ctrl.combo_opacity.currentTextChanged.connect(self._on_opacity_or_blend)
        self.ctrl.slider_strength.valueChanged.connect(self._on_opacity_or_blend)
        self.ctrl.combo_cmap.currentTextChanged.connect(self._on_colormap_changed)
        self.ctrl.slider_zmin.valueChanged.connect(self._on_slab_changed)
        self.ctrl.slider_zmax.valueChanged.connect(self._on_slab_changed)
        self.ctrl.slider_slice_z.valueChanged.connect(self._on_slice_changed)
        self.ctrl.slider_slice_y.valueChanged.connect(self._on_slice_changed)
        self.ctrl.combo_blending.currentTextChanged.connect(self._on_opacity_or_blend)
        self.ctrl.btn_bg.clicked.connect(self._on_bg_color)

        # 2D views: disable rotation, use image style
        for p in [self.plotter_axial, self.plotter_coronal, self.plotter_enface]:
            p.interactor_style = "Image"

        # --- Load first scan ---
        if self.scans:
            self._load_current_scan()

    # ------------------------------------------------------------------
    # Scan navigation
    # ------------------------------------------------------------------

    def _go_prev(self):
        if self.current_idx > 0:
            self.ctrl.scan_combo.setCurrentIndex(self.current_idx - 1)

    def _go_next(self):
        if self.current_idx < len(self.scans) - 1:
            self.ctrl.scan_combo.setCurrentIndex(self.current_idx + 1)

    def _on_scan_changed(self, idx: int):
        if idx < 0 or idx >= len(self.scans):
            return
        self.current_idx = idx
        self._load_current_scan()

    # ------------------------------------------------------------------
    # Load / reload
    # ------------------------------------------------------------------

    def _load_current_scan(self):
        scan = self.scans[self.current_idx]
        self.pv_grid = None
        self.np_vol = None
        self.vol_actor = None
        self.axial_actor = None
        self.coronal_actor = None
        self.enface_actor = None
        gc.collect()

        self.setWindowTitle(
            f"OCT Volume Viewer — [{scan['group']} {scan['suffix']}] "
            f"[{self.current_idx + 1}/{len(self.scans)}]"
        )

        self.pv_grid, self.np_vol = load_volume(scan["path"])

        for p in [self.plotter_3d, self.plotter_axial, self.plotter_coronal, self.plotter_enface]:
            p.clear()
            p.set_background(self.bg_color)

        self.plotter_3d.add_text("3D Volume", position="upper_left", font_size=10, color="white", name="_label")
        self.plotter_axial.add_text("Axial B-Scan (XY)", position="upper_left", font_size=10, color="white", name="_label")
        self.plotter_coronal.add_text("Coronal B-Scan (XZ)", position="upper_left", font_size=10, color="white", name="_label")
        self.plotter_enface.add_text("En-Face MIP", position="upper_left", font_size=10, color="white", name="_label")

        self.ctrl.slider_zmin.blockSignals(True)
        self.ctrl.slider_zmax.blockSignals(True)
        self.ctrl.slider_slice_z.blockSignals(True)
        self.ctrl.slider_slice_y.blockSignals(True)
        self.ctrl.slider_zmin.setValue(0)
        self.ctrl.slider_zmax.setValue(511)
        self.ctrl.slider_slice_z.setValue(256)
        self.ctrl.slider_slice_y.setValue(256)
        self.ctrl.slider_zmin.blockSignals(False)
        self.ctrl.slider_zmax.blockSignals(False)
        self.ctrl.slider_slice_z.blockSignals(False)
        self.ctrl.slider_slice_y.blockSignals(False)

        self._rebuild_all_views()

    # ------------------------------------------------------------------
    # View builders
    # ------------------------------------------------------------------

    def _get_clim(self) -> tuple[float, float]:
        level = self.ctrl.slider_level.value() / 1000.0
        window = self.ctrl.slider_window.value() / 1000.0
        clim_min = max(0.0, level)
        clim_max = max(clim_min + 0.01, level + window)
        return (clim_min, clim_max)

    def _get_opacity(self):
        return self.ctrl.combo_opacity.currentText()

    def _get_opacity_unit_distance(self) -> float:
        strength = self.ctrl.slider_strength.value()
        return max(0.5, 100.0 / strength)

    def _get_cmap(self) -> str:
        return self.ctrl.combo_cmap.currentText()

    def _get_blending(self) -> str:
        return self.ctrl.combo_blending.currentText()

    def _get_slab(self) -> tuple[int, int]:
        z_min = min(self.ctrl.slider_zmin.value(), self.ctrl.slider_zmax.value())
        z_max = max(self.ctrl.slider_zmin.value(), self.ctrl.slider_zmax.value())
        return (z_min, z_max)

    def _rebuild_all_views(self):
        """Rebuild all four views from scratch."""
        if self.np_vol is None:
            return
        clim = self._get_clim()
        cmap = self._get_cmap()
        opacity = self._get_opacity()
        blending = self._get_blending()
        z_min, z_max = self._get_slab()
        slice_z = self.ctrl.slider_slice_z.value()
        slice_y = self.ctrl.slider_slice_y.value()

        # --- 3D Volume ---
        self._rebuild_3d(clim, cmap, opacity, blending, z_min, z_max)

        # --- Axial B-Scan (XY at slice_z) ---
        self._rebuild_axial(clim, cmap, slice_z)

        # --- Coronal B-Scan (XZ at slice_y) ---
        self._rebuild_coronal(clim, cmap, slice_y)

        # --- En-Face MIP ---
        self._rebuild_enface(clim, cmap, z_min, z_max)

        self.ctrl.lbl_zmin.setText(f"Z Min: {z_min}")
        self.ctrl.lbl_zmax.setText(f"Z Max: {z_max}")
        self.ctrl.lbl_slice_z.setText(f"Slice Z (axial): {slice_z}")
        self.ctrl.lbl_slice_y.setText(f"Slice Y (coronal): {slice_y}")
        self.ctrl.lbl_window.setText(f"Window: {clim[1] - clim[0]:.3f}")
        self.ctrl.lbl_level.setText(f"Level: {clim[0]:.3f}")
        self.ctrl.lbl_strength.setText(f"Strength: {self.ctrl.slider_strength.value()}")

    def _rebuild_3d(self, clim, cmap, opacity, blending, z_min, z_max):
        if self.pv_grid is None:
            return
        self.plotter_3d.clear()
        self.plotter_3d.set_background(self.bg_color)
        self.plotter_3d.add_text("3D Volume", position="upper_left", font_size=10, color="white", name="_label")

        subset = self.pv_grid.extract_subset([0, VOL_SHAPE[0], 0, VOL_SHAPE[1], z_min, z_max])
        kwargs = {
            "cmap": cmap,
            "clim": list(clim),
            "opacity": opacity,
            "mapper": "smart",
            "name": "vol",
        }
        try:
            self.vol_actor = self.plotter_3d.add_volume(subset, **kwargs, blend_mode=blending)
        except TypeError:
            self.vol_actor = self.plotter_3d.add_volume(subset, **kwargs)

        try:
            self.plotter_3d.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        self.plotter_3d.view_isometric()
        self.plotter_3d.reset_camera()
        self.plotter_3d.render()

    def _make_2d_grid(self, data_2d: np.ndarray) -> pv.ImageData:
        h, w = data_2d.shape
        grid = pv.ImageData(dimensions=(w + 1, h + 1, 1))
        grid.cell_data["intensity"] = data_2d.ravel(order="F").astype(np.float32)
        return grid

    def _rebuild_axial(self, clim, cmap, slice_z):
        self.plotter_axial.clear()
        self.plotter_axial.set_background(self.bg_color)
        self.plotter_axial.add_text("Axial B-Scan (XY)", position="upper_left", font_size=10, color="white", name="_label")
        if self.np_vol is None:
            return
        slice_data = self.np_vol[:, :, slice_z]
        grid = self._make_2d_grid(slice_data)
        self.axial_actor = self.plotter_axial.add_mesh(
            grid, cmap=cmap, clim=list(clim), show_scalar_bar=False, name="axial"
        )
        self.plotter_axial.view_xy()
        self.plotter_axial.reset_camera()
        self.plotter_axial.render()

    def _rebuild_coronal(self, clim, cmap, slice_y):
        self.plotter_coronal.clear()
        self.plotter_coronal.set_background(self.bg_color)
        self.plotter_coronal.add_text("Coronal B-Scan (XZ)", position="upper_left", font_size=10, color="white", name="_label")
        if self.np_vol is None:
            return
        slice_data = self.np_vol[:, slice_y, :]
        grid = self._make_2d_grid(slice_data)
        self.coronal_actor = self.plotter_coronal.add_mesh(
            grid, cmap=cmap, clim=list(clim), show_scalar_bar=False, name="coronal"
        )
        self.plotter_coronal.view_xz()
        self.plotter_coronal.reset_camera()
        self.plotter_coronal.render()

    def _rebuild_enface(self, clim, cmap, z_min, z_max):
        self.plotter_enface.clear()
        self.plotter_enface.set_background(self.bg_color)
        self.plotter_enface.add_text("En-Face MIP", position="upper_left", font_size=10, color="white", name="_label")
        if self.np_vol is None:
            return
        z0 = max(0, z_min)
        z1 = min(VOL_SHAPE[2], z_max + 1)
        if z1 <= z0:
            z1 = z0 + 1
        mip = np.max(self.np_vol[:, :, z0:z1], axis=2)
        grid = self._make_2d_grid(mip)
        self.enface_actor = self.plotter_enface.add_mesh(
            grid, cmap=cmap, clim=list(clim), show_scalar_bar=False, name="enface"
        )
        self.plotter_enface.view_xy()
        self.plotter_enface.reset_camera()
        self.plotter_enface.render()

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_window_level(self):
        clim = self._get_clim()
        self.ctrl.lbl_window.setText(f"Window: {clim[1] - clim[0]:.3f}")
        self.ctrl.lbl_level.setText(f"Level: {clim[0]:.3f}")
        # In-place update on all actors
        if self.vol_actor and hasattr(self.vol_actor, "mapper") and self.vol_actor.mapper:
            self.vol_actor.mapper.scalar_range = list(clim)
            self.plotter_3d.render()
        # Update 2D views in-place
        for plotter, actor in [
            (self.plotter_axial, self.axial_actor),
            (self.plotter_coronal, self.coronal_actor),
            (self.plotter_enface, self.enface_actor),
        ]:
            if actor and hasattr(actor, "mapper") and actor.mapper:
                actor.mapper.scalar_range = list(clim)
                plotter.render()

    def _on_colormap_changed(self, cmap_name: str):
        """Colormap changed — update all views."""
        if self.vol_actor and hasattr(self.vol_actor, "mapper") and self.vol_actor.mapper:
            try:
                self.vol_actor.mapper.lookup_table = pv.LookupTable(cmap=cmap_name)
            except Exception:
                pass
            self.plotter_3d.render()
        for plotter, actor in [
            (self.plotter_axial, self.axial_actor),
            (self.plotter_coronal, self.coronal_actor),
            (self.plotter_enface, self.enface_actor),
        ]:
            if actor and hasattr(actor, "mapper") and actor.mapper:
                try:
                    actor.mapper.lookup_table = pv.LookupTable(cmap=cmap_name)
                except Exception:
                    pass
                plotter.render()

    def _on_opacity_or_blend(self):
        """Opacity preset, strength, or blending changed — rebuild 3D view."""
        if self.np_vol is None:
            return
        clim = self._get_clim()
        cmap = self._get_cmap()
        opacity = self._get_opacity()
        blending = self._get_blending()
        z_min, z_max = self._get_slab()
        self._rebuild_3d(clim, cmap, opacity, blending, z_min, z_max)

    def _on_slab_changed(self):
        """Z-slab changed — rebuild 3D and en-face."""
        if self.np_vol is None:
            return
        clim = self._get_clim()
        cmap = self._get_cmap()
        opacity = self._get_opacity()
        blending = self._get_blending()
        z_min, z_max = self._get_slab()
        self._rebuild_3d(clim, cmap, opacity, blending, z_min, z_max)
        self._rebuild_enface(clim, cmap, z_min, z_max)
        self.ctrl.lbl_zmin.setText(f"Z Min: {z_min}")
        self.ctrl.lbl_zmax.setText(f"Z Max: {z_max}")

    def _on_slice_changed(self):
        """Slice position changed — rebuild corresponding 2D view."""
        if self.np_vol is None:
            return
        clim = self._get_clim()
        cmap = self._get_cmap()
        slice_z = self.ctrl.slider_slice_z.value()
        slice_y = self.ctrl.slider_slice_y.value()
        self._rebuild_axial(clim, cmap, slice_z)
        self._rebuild_coronal(clim, cmap, slice_y)
        self.ctrl.lbl_slice_z.setText(f"Slice Z (axial): {slice_z}")
        self.ctrl.lbl_slice_y.setText(f"Slice Y (coronal): {slice_y}")

    def _on_bg_color(self):
        color = QColorDialog.getColor(self.bg_color, self, "Choose Background Color")
        if color.isValid():
            self.bg_color = color.name()
            for p in [self.plotter_3d, self.plotter_axial, self.plotter_coronal, self.plotter_enface]:
                p.set_background(self.bg_color)
                p.render()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        for p in [self.plotter_3d, self.plotter_axial, self.plotter_coronal, self.plotter_enface]:
            p.close()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    apply_dark_palette(app)
    window = OCTViewer()
    window.show()
    sys.exit(app.exec())
