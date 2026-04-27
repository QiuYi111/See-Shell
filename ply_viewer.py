#!/usr/bin/env python3
import json
import os
import tempfile
os.environ["QT_API"] = "pyqt6"

import glob
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QIcon, QPixmap, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
    QDialog,
)

BASE = os.path.dirname(os.path.abspath(__file__))
OCT_DIR = os.path.join(BASE, "oct_cloud")
FILTER_MODES = ["Intensity", "Hue (warm↔cool)", "Brightness", "Saturation"]
THUMB_SIZE = 160


def _rgb_to_hsv_vec(rgb):
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


def _rgb_to_metrics(rgb):
    r, g, b = rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0
    intensity = (r * 0.299 + g * 0.587 + b * 0.114) * 255
    brightness = np.maximum(np.maximum(r, g), b) * 255
    hue, sat = _rgb_to_hsv_vec(rgb)
    return dict(Intensity=intensity, **{"Hue (warm↔cool)": hue},
                Brightness=brightness, Saturation=sat)


def _mip_thumbnail(npy_path):
    npy = np.load(npy_path)
    img = np.zeros((512, 512), dtype=np.float32)
    x = npy[:, 0].astype(int)
    y = npy[:, 1].astype(int)
    np.maximum.at(img, (x, y), npy[:, 3])
    mx = img.max()
    if mx > 0:
        img = img / mx * 255
    h, w = img.shape
    return QImage(img.astype(np.uint8).tobytes(), w, h, w,
                 QImage.Format.Format_Grayscale8).copy()


def _npy_for_ply(ply_path):
    return ply_path.replace(".ply", ".npy")


def collect(directory: str | None = None):
    """Scan *directory* (defaults to ``OCT_DIR``) for PLY scan files.

    Returns a list of dicts with keys: group, suffix, path, npy.
    """
    base = directory or OCT_DIR
    scans = []
    for f in sorted(glob.glob(os.path.join(base, "**", "*.ply"), recursive=True)):
        if os.path.getsize(f) == 0:
            continue
        reldir = os.path.relpath(os.path.dirname(f), base)
        dirname = os.path.basename(os.path.dirname(f))
        if dirname.startswith("scan_array_"):
            group = dirname.replace("scan_array_", "").split("_")
            group = f"{group[0]} {group[1]}"
        elif dirname.startswith("scan_session_"):
            group = dirname.replace("scan_session_", "").split("_")
            group = f"{group[0]} {group[1]}"
        else:
            group = reldir.replace(os.sep, "/")
        basename = os.path.basename(f)
        parts = basename.replace(".ply", "").split("_")
        suffix = "_".join(parts[-2:]) if len(parts) >= 2 else basename
        npy = _npy_for_ply(f)
        has_npy = os.path.exists(npy) and os.path.getsize(npy) > 0
        scans.append({"group": group, "suffix": suffix,
                      "path": f, "npy": npy if has_npy else None})
    return scans


class ThumbnailLoader(QThread):
    done = pyqtSignal(int, QPixmap)

    def __init__(self, scans):
        super().__init__()
        self.scans = scans

    def run(self):
        for i, s in enumerate(self.scans):
            try:
                if s.get("remote_host") and not os.path.isfile(s["path"]):
                    continue
                if s["npy"]:
                    qimg = _mip_thumbnail(s["npy"])
                else:
                    mesh = pv.read(s["path"])
                    rgb = np.asarray(mesh["RGB"])
                    gray = (rgb.astype(float) @ [0.299, 0.587, 0.114]).astype(np.uint8)
                    qimg = QImage(gray.tobytes(), 1, len(gray), 1,
                                  QImage.Format.Format_Grayscale8)
                px = QPixmap.fromImage(qimg).scaled(
                    THUMB_SIZE, THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                self.done.emit(i, px)
            except Exception:
                pass


class ColumnWorker(QThread):
    """Background worker for STL column generation."""
    done = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, ply_path, params, config, gen_complement=False):
        super().__init__()
        self.ply_path = ply_path
        self.params = params
        self.config = config
        self.gen_complement = gen_complement

    def run(self):
        try:
            from oct_to_column import pipeline_from_params, ColumnConfig
            mesh, report = pipeline_from_params(self.ply_path, self.params, self.config)
            msg = f"✅ {self.config.output_stl}"
            if not report["is_watertight"]:
                msg += " (not watertight)"

            if self.gen_complement:
                comp_config = ColumnConfig(**{**vars(self.config),
                    "is_complement": True,
                    "output_stl": self.config.output_stl.replace(".stl", "_complement.stl"),
                })
                mesh2, report2 = pipeline_from_params(self.ply_path, self.params, comp_config)
                msg += f"\n✅ {comp_config.output_stl}"

            self.done.emit(True, msg)
        except Exception as e:
            self.done.emit(False, f"❌ {e}")


class JITDownloadWorker(QThread):
    done = pyqtSignal(int, str, bool)  # (row, cache_path_or_error, success)

    def __init__(self, row, host, remote_path, cache_dir):
        super().__init__()
        self.row = row
        self.host = host
        self.remote_path = remote_path
        self.cache_dir = cache_dir

    def run(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            basename = os.path.basename(self.remote_path)
            local = os.path.join(self.cache_dir, basename)
            if not os.path.isfile(local) or os.path.getsize(local) == 0:
                from ssh_utils import download_file
                download_file(self.host, self.remote_path, local)
            self.done.emit(self.row, local, True)
        except Exception as exc:
            self.done.emit(self.row, str(exc), False)


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLY Viewer")
        self.resize(1600, 950)
        self.scans = collect()
        self.idx = -1
        self.bg_color = "#0D0D14"
        self.orig_pts = None
        self.orig_rgb = None
        self.metrics = {}
        self.filter_mode = "Intensity"
        self.axis_bounds = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ── Left: gallery ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        self.btn_load_data = QPushButton("📂 Load Data…")
        left_layout.addWidget(self.btn_load_data)
        self.scan_list = QListWidget()
        self.scan_list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.scan_list.setSpacing(4)
        self.scan_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.scan_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.scan_list.setMovement(QListWidget.Movement.Static)
        self.scan_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.scan_list.setFont(QFont("", 8))
        left_layout.addWidget(self.scan_list)
        left.setMinimumWidth(200)
        left.setMaximumWidth(420)
        splitter.addWidget(left)

        # ── Right: viewer + controls ──
        right = QWidget()
        rvbox = QVBoxLayout(right)
        rvbox.setContentsMargins(4, 4, 4, 4)
        rvbox.setSpacing(4)

        self.plotter = QtInteractor(right)
        rvbox.addWidget(self.plotter.interactor, stretch=1)

        # Row 1: info + point size + BG
        row_ctrl = QHBoxLayout()
        self.lbl_info = QLabel("Select a scan")
        row_ctrl.addWidget(self.lbl_info)
        row_ctrl.addStretch()
        row_ctrl.addWidget(QLabel("Size:"))
        self.slider_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_size.setRange(1, 20)
        self.slider_size.setValue(3)
        self.slider_size.setFixedWidth(120)
        row_ctrl.addWidget(self.slider_size)
        self.btn_bg = QPushButton("BG")
        row_ctrl.addWidget(self.btn_bg)
        rvbox.addLayout(row_ctrl)

        # Row 2: Z range
        row_z = QHBoxLayout()
        row_z.addWidget(QLabel("Z min:"))
        self.slider_zmin = QSlider(Qt.Orientation.Horizontal)
        self.slider_zmin.setRange(0, 1000)
        self.slider_zmin.setValue(0)
        row_z.addWidget(self.slider_zmin)
        self.lbl_zmin = QLabel("0")
        self.lbl_zmin.setFixedWidth(36)
        row_z.addWidget(self.lbl_zmin)
        row_z.addWidget(QLabel("Z max:"))
        self.slider_zmax = QSlider(Qt.Orientation.Horizontal)
        self.slider_zmax.setRange(0, 1000)
        self.slider_zmax.setValue(1000)
        row_z.addWidget(self.slider_zmax)
        self.lbl_zmax = QLabel("")
        self.lbl_zmax.setFixedWidth(36)
        row_z.addWidget(self.lbl_zmax)
        self.btn_reset_z = QPushButton("Reset Z")
        row_z.addWidget(self.btn_reset_z)
        rvbox.addLayout(row_z)

        # Row 3: X range
        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("X min:"))
        self.slider_xmin = QSlider(Qt.Orientation.Horizontal)
        self.slider_xmin.setRange(0, 1000)
        self.slider_xmin.setValue(0)
        row_x.addWidget(self.slider_xmin)
        self.lbl_xmin = QLabel("0")
        self.lbl_xmin.setFixedWidth(36)
        row_x.addWidget(self.lbl_xmin)
        row_x.addWidget(QLabel("X max:"))
        self.slider_xmax = QSlider(Qt.Orientation.Horizontal)
        self.slider_xmax.setRange(0, 1000)
        self.slider_xmax.setValue(1000)
        row_x.addWidget(self.slider_xmax)
        self.lbl_xmax = QLabel("")
        self.lbl_xmax.setFixedWidth(36)
        row_x.addWidget(self.lbl_xmax)
        self.btn_reset_x = QPushButton("Reset X")
        row_x.addWidget(self.btn_reset_x)
        rvbox.addLayout(row_x)

        # Row 4: Y range
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Y min:"))
        self.slider_ymin = QSlider(Qt.Orientation.Horizontal)
        self.slider_ymin.setRange(0, 1000)
        self.slider_ymin.setValue(0)
        row_y.addWidget(self.slider_ymin)
        self.lbl_ymin = QLabel("0")
        self.lbl_ymin.setFixedWidth(36)
        row_y.addWidget(self.lbl_ymin)
        row_y.addWidget(QLabel("Y max:"))
        self.slider_ymax = QSlider(Qt.Orientation.Horizontal)
        self.slider_ymax.setRange(0, 1000)
        self.slider_ymax.setValue(1000)
        row_y.addWidget(self.slider_ymax)
        self.lbl_ymax = QLabel("")
        self.lbl_ymax.setFixedWidth(36)
        row_y.addWidget(self.lbl_ymax)
        self.btn_reset_y = QPushButton("Reset Y")
        row_y.addWidget(self.btn_reset_y)
        rvbox.addLayout(row_y)

        # Row 5: color filter
        grid = QHBoxLayout()
        grid.addWidget(QLabel("Filter:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(FILTER_MODES)
        grid.addWidget(self.combo_filter)
        grid.addWidget(QLabel("Min:"))
        self.slider_min = QSlider(Qt.Orientation.Horizontal)
        self.slider_min.setRange(0, 255)
        self.slider_min.setValue(0)
        grid.addWidget(self.slider_min)
        self.lbl_min = QLabel("0")
        self.lbl_min.setFixedWidth(24)
        grid.addWidget(self.lbl_min)
        grid.addWidget(QLabel("Max:"))
        self.slider_max = QSlider(Qt.Orientation.Horizontal)
        self.slider_max.setRange(0, 255)
        self.slider_max.setValue(255)
        grid.addWidget(self.slider_max)
        self.lbl_max = QLabel("255")
        self.lbl_max.setFixedWidth(28)
        grid.addWidget(self.lbl_max)
        self.lbl_filtered = QLabel("")
        grid.addWidget(self.lbl_filtered)
        self.btn_reset_filter = QPushButton("Reset")
        grid.addWidget(self.btn_reset_filter)
        rvbox.addLayout(grid)

        # Row 6: export
        row_export = QHBoxLayout()
        row_export.addStretch()
        self.btn_export = QPushButton("Export Params")
        self.btn_export.setFixedWidth(140)
        row_export.addWidget(self.btn_export)
        self.lbl_export = QLabel("")
        row_export.addWidget(self.lbl_export)
        row_export.addStretch()
        rvbox.addLayout(row_export)

        # Row 7: Column Generation
        grp_column = QGroupBox("Column Generation")
        grp_layout = QVBoxLayout(grp_column)
        grp_layout.setSpacing(4)

        row_px = QHBoxLayout()
        row_px.addWidget(QLabel("Pixel XY (mm):"))
        self.spin_pixel_xy = QDoubleSpinBox()
        self.spin_pixel_xy.setRange(0.0001, 1.0)
        self.spin_pixel_xy.setDecimals(4)
        self.spin_pixel_xy.setSingleStep(0.001)
        self.spin_pixel_xy.setValue(0.0147)
        self.spin_pixel_xy.setFixedWidth(90)
        row_px.addWidget(self.spin_pixel_xy)
        row_px.addWidget(QLabel("Pixel Z (mm):"))
        self.spin_pixel_z = QDoubleSpinBox()
        self.spin_pixel_z.setRange(0.0001, 1.0)
        self.spin_pixel_z.setDecimals(4)
        self.spin_pixel_z.setSingleStep(0.001)
        self.spin_pixel_z.setValue(0.0147)
        self.spin_pixel_z.setFixedWidth(90)
        row_px.addWidget(self.spin_pixel_z)
        row_px.addStretch()
        grp_layout.addLayout(row_px)

        row_smooth = QHBoxLayout()
        row_smooth.addWidget(QLabel("Smooth σ:"))
        self.spin_smooth = QDoubleSpinBox()
        self.spin_smooth.setRange(0.0, 50.0)
        self.spin_smooth.setDecimals(1)
        self.spin_smooth.setSingleStep(0.5)
        self.spin_smooth.setValue(6.0)
        self.spin_smooth.setFixedWidth(70)
        row_smooth.addWidget(self.spin_smooth)
        row_smooth.addWidget(QLabel("Base H (mm):"))
        self.spin_base_h = QDoubleSpinBox()
        self.spin_base_h.setRange(0.5, 50.0)
        self.spin_base_h.setDecimals(1)
        self.spin_base_h.setSingleStep(0.5)
        self.spin_base_h.setValue(3.0)
        self.spin_base_h.setFixedWidth(70)
        row_smooth.addWidget(self.spin_base_h)
        row_smooth.addStretch()
        grp_layout.addLayout(row_smooth)

        row_gen = QHBoxLayout()
        self.chk_complement = QCheckBox("Generate pair (normal + complement)")
        row_gen.addWidget(self.chk_complement)
        row_gen.addStretch()
        self.btn_generate = QPushButton("Generate STL")
        self.btn_generate.setFixedWidth(140)
        row_gen.addWidget(self.btn_generate)
        grp_layout.addLayout(row_gen)

        self.lbl_gen_status = QLabel("")
        self.lbl_gen_status.setWordWrap(True)
        grp_layout.addWidget(self.lbl_gen_status)

        row_save = QHBoxLayout()
        row_save.addStretch()
        self.btn_save_stl = QPushButton("💾 Save STL As…")
        self.btn_save_stl.setFixedWidth(160)
        self.btn_save_stl.setEnabled(False)
        row_save.addWidget(self.btn_save_stl)
        row_save.addStretch()
        grp_layout.addLayout(row_save)

        rvbox.addWidget(grp_column)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ── Signals ──
        self.scan_list.currentRowChanged.connect(self._on_select)
        self.btn_load_data.clicked.connect(self._on_load_data)
        self.slider_size.valueChanged.connect(self._on_size)
        self.btn_bg.clicked.connect(self._on_bg)
        self.combo_filter.currentTextChanged.connect(self._on_filter_mode)
        self.slider_min.valueChanged.connect(self._on_filter)
        self.slider_max.valueChanged.connect(self._on_filter)
        self.btn_reset_filter.clicked.connect(self._reset_filter)
        self.slider_zmin.valueChanged.connect(self._on_spatial)
        self.slider_zmax.valueChanged.connect(self._on_spatial)
        self.slider_xmin.valueChanged.connect(self._on_spatial)
        self.slider_xmax.valueChanged.connect(self._on_spatial)
        self.slider_ymin.valueChanged.connect(self._on_spatial)
        self.slider_ymax.valueChanged.connect(self._on_spatial)
        self.btn_reset_z.clicked.connect(lambda: self._reset_axis("z"))
        self.btn_reset_x.clicked.connect(lambda: self._reset_axis("x"))
        self.btn_reset_y.clicked.connect(lambda: self._reset_axis("y"))
        self.btn_export.clicked.connect(self._on_export)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_save_stl.clicked.connect(self._on_save_stl)

        for s in self.scans:
            item = QListWidgetItem(f"{s['group']}\n{s['suffix']}")
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.scan_list.addItem(item)

        self.plotter.set_background(self.bg_color)

        self.thumb_loader = ThumbnailLoader(self.scans)
        self.thumb_loader.done.connect(self._set_thumbnail)
        self.thumb_loader.start()

        if self.scans:
            self.scan_list.setCurrentRow(0)

    # ── helpers ──

    def _slider_to_axis_val(self, axis, slider_val):
        lo, hi = self.axis_bounds.get(axis, (0.0, 1.0))
        return lo + (hi - lo) * slider_val / 1000.0

    def _axis_val_to_slider(self, axis, val):
        lo, hi = self.axis_bounds.get(axis, (0.0, 1.0))
        return int(1000.0 * (val - lo) / max(hi - lo, 1e-9))

    # ── slots ──

    def _set_thumbnail(self, idx, pixmap):
        if 0 <= idx < self.scan_list.count():
            self.scan_list.item(idx).setIcon(QIcon(pixmap))

    def _on_select(self, row):
        if row < 0 or row >= len(self.scans):
            return
        self.idx = row
        s = self.scans[row]

        if s.get("remote_host") and not os.path.isfile(s["path"]):
            self.lbl_info.setText(f"Downloading {s['suffix']}…")
            self.scan_list.setEnabled(False)
            cache_dir = os.path.dirname(s["path"])
            self._jit_worker = JITDownloadWorker(
                row, s["remote_host"], s["remote_path"], cache_dir)
            self._jit_worker.done.connect(self._on_jit_done)
            self._jit_worker.start()
            return

        self._load_mesh(s, row)

    def _on_jit_done(self, row, path_or_err, success):
        self.scan_list.setEnabled(True)
        if not success:
            self.lbl_info.setText(f"Download failed: {path_or_err}")
            return
        s = self.scans[row]
        s["path"] = path_or_err
        self._load_mesh(s, row)

    def _load_mesh(self, s, row):
        mesh = pv.read(s["path"])
        n = mesh.n_points

        self.orig_pts = np.asarray(mesh.points).copy()
        self.orig_rgb = np.asarray(mesh["RGB"]).copy()
        self.metrics = _rgb_to_metrics(self.orig_rgb)

        for i, ax in enumerate("xyz"):
            col = self.orig_pts[:, i]
            self.axis_bounds[ax] = (float(col.min()), float(col.max()))

        for attr in ("slider_zmin", "slider_xmin", "slider_ymin"):
            getattr(self, attr).blockSignals(True)
            getattr(self, attr).setValue(0)
            getattr(self, attr).blockSignals(False)
        for attr in ("slider_zmax", "slider_xmax", "slider_ymax"):
            getattr(self, attr).blockSignals(True)
            getattr(self, attr).setValue(1000)
            getattr(self, attr).blockSignals(False)

        self._rebuild_mesh()

        self.setWindowTitle(
            f"PLY Viewer — {s['group']} {s['suffix']}  ({n:,} pts)  [{row+1}/{len(self.scans)}]")
        self.lbl_info.setText(f"{s['group']} {s['suffix']} — {n:,} pts")

    def _rebuild_mesh(self):
        if self.orig_pts is None:
            return

        metric = self.metrics.get(self.filter_mode)
        lo = self.slider_min.value()
        hi = self.slider_max.value()
        if lo > hi:
            lo, hi = hi, lo
        mask = (metric >= lo) & (metric <= hi) if metric is not None else np.ones(len(self.orig_pts), dtype=bool)

        for ax, smin_attr, smax_attr in [
            ("z", "slider_zmin", "slider_zmax"),
            ("x", "slider_xmin", "slider_xmax"),
            ("y", "slider_ymin", "slider_ymax"),
        ]:
            blo, bhi = self.axis_bounds.get(ax, (0, 1))
            ax_idx = "xyz".index(ax)
            vmin = self._slider_to_axis_val(ax, getattr(self, smin_attr).value())
            vmax = self._slider_to_axis_val(ax, getattr(self, smax_attr).value())
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            col = self.orig_pts[:, ax_idx]
            mask &= (col >= vmin) & (col <= vmax)

        pts = self.orig_pts[mask]
        rgb = self.orig_rgb[mask]
        vis = int(mask.sum())
        total = len(self.orig_pts)

        self.plotter.clear()
        self.plotter.set_background(self.bg_color)
        if vis > 0:
            cloud = pv.PolyData(pts)
            cloud["RGB"] = rgb
            self.plotter.add_mesh(
                cloud, scalars=None, rgb=True,
                point_size=self.slider_size.value(),
                render_points_as_spheres=True, name="cloud")
            self.plotter.reset_camera()
        self.plotter.render()

        self.lbl_filtered.setText(f"{vis:,}/{total:,} ({100*vis/max(1,total):.0f}%)")
        self.lbl_min.setText(str(lo))
        self.lbl_max.setText(str(hi))
        self._update_axis_labels()

    def _update_axis_labels(self):
        for ax, smin_attr, smax_attr, lmin_attr, lmax_attr in [
            ("z", "slider_zmin", "slider_zmax", "lbl_zmin", "lbl_zmax"),
            ("x", "slider_xmin", "slider_xmax", "lbl_xmin", "lbl_xmax"),
            ("y", "slider_ymin", "slider_ymax", "lbl_ymin", "lbl_ymax"),
        ]:
            vmin = self._slider_to_axis_val(ax, getattr(self, smin_attr).value())
            vmax = self._slider_to_axis_val(ax, getattr(self, smax_attr).value())
            getattr(self, lmin_attr).setText(f"{vmin:.1f}")
            getattr(self, lmax_attr).setText(f"{vmax:.1f}")

    def _on_size(self, val):
        actor = self.plotter.renderer.actors.get("cloud")
        if actor:
            actor.GetProperty().SetPointSize(val)
            self.plotter.render()

    def _on_bg(self):
        c = QColorDialog.getColor(QColor(self.bg_color), self, "Background")
        if c.isValid():
            self.bg_color = c.name()
            self.plotter.set_background(self.bg_color)
            self.plotter.render()

    def _on_filter_mode(self, mode):
        self.filter_mode = mode
        self._rebuild_mesh()

    def _on_filter(self):
        self._rebuild_mesh()

    def _on_spatial(self):
        self._rebuild_mesh()

    def _reset_filter(self):
        self.slider_min.blockSignals(True)
        self.slider_max.blockSignals(True)
        self.slider_min.setValue(0)
        self.slider_max.setValue(255)
        self.slider_min.blockSignals(False)
        self.slider_max.blockSignals(False)
        self._rebuild_mesh()

    def _reset_axis(self, ax):
        smin_attr = f"slider_{ax}min"
        smax_attr = f"slider_{ax}max"
        getattr(self, smin_attr).blockSignals(True)
        getattr(self, smax_attr).blockSignals(True)
        getattr(self, smin_attr).setValue(0)
        getattr(self, smax_attr).setValue(1000)
        getattr(self, smin_attr).blockSignals(False)
        getattr(self, smax_attr).blockSignals(False)
        self._rebuild_mesh()

    def _get_current_params(self):
        s = self.scans[self.idx] if self.idx >= 0 else None
        lo = self.slider_min.value()
        hi = self.slider_max.value()
        if lo > hi:
            lo, hi = hi, lo
        params = {
            "scan_file": s["path"] if s else None,
            "scan_group": s["group"] if s else None,
            "scan_suffix": s["suffix"] if s else None,
            "point_size": self.slider_size.value(),
            "background_color": self.bg_color,
            "filter_mode": self.filter_mode,
            "filter_min": lo,
            "filter_max": hi,
        }
        for ax in "xyz":
            vmin = self._slider_to_axis_val(ax, getattr(self, f"slider_{ax}min").value())
            vmax = self._slider_to_axis_val(ax, getattr(self, f"slider_{ax}max").value())
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            params[f"{ax}_min"] = round(vmin, 4)
            params[f"{ax}_max"] = round(vmax, 4)
        return params

    def _on_generate(self):
        if self.idx < 0:
            self.lbl_gen_status.setText("Select a scan first")
            return
        s = self.scans[self.idx]
        ply_path = s["path"]
        if not os.path.isfile(ply_path):
            self.lbl_gen_status.setText(f"PLY not found: {ply_path}")
            return

        self.btn_generate.setEnabled(False)
        self.lbl_gen_status.setText("Generating...")

        from oct_to_column import ColumnConfig
        params = self._get_current_params()
        suffix = params.get("scan_suffix", "output")
        out_stl = os.path.join(BASE, f"column_{suffix}.stl")

        config = ColumnConfig(
            pixel_size_mm=self.spin_pixel_xy.value(),
            z_pixel_size_mm=self.spin_pixel_z.value(),
            smoothing_sigma=self.spin_smooth.value(),
            base_height_mm=self.spin_base_h.value(),
            is_complement=False,
            output_stl=out_stl,
        )

        gen_comp = self.chk_complement.isChecked()
        self._column_worker = ColumnWorker(ply_path, params, config, gen_comp)
        self._column_worker.done.connect(self._on_gen_done)
        self._column_worker.start()

    def _on_gen_done(self, success, msg):
        self.btn_generate.setEnabled(True)
        self.lbl_gen_status.setText(msg)
        if success:
            self.btn_save_stl.setEnabled(True)

    def _on_export(self):
        params = self._get_current_params()
        default_name = f"viewer_params_{params.get('scan_group','unknown').replace(' ','_')}_{params.get('scan_suffix','')}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization Parameters", default_name,
            "JSON Files (*.json);;All Files (*)")
        if path:
            with open(path, "w") as f:
                json.dump(params, f, indent=2)
            self.lbl_export.setText(f"Saved → {os.path.basename(path)}")

    def _on_load_data(self):
        from data_loader import DataSourceDialog
        dialog = DataSourceDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        host, path = dialog.result_data
        if not path:
            return

        new_scans = []
        if host is None:
            if not os.path.isdir(path):
                return
            new_scans = collect(path)
        else:
            try:
                from ssh_utils import find_remote_files
                remote_files = find_remote_files(host, path, ".ply")
            except RuntimeError as exc:
                self.lbl_info.setText(f"SSH error: {exc}")
                return
            if not remote_files:
                return
            cache_root = os.path.join(tempfile.gettempdir(), "see-shell", host)
            for rp in remote_files:
                reldir = os.path.relpath(os.path.dirname(rp), path)
                dirname = os.path.basename(os.path.dirname(rp))
                if dirname.startswith("scan_array_"):
                    parts = dirname.replace("scan_array_", "").split("_")
                    group = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else dirname
                elif dirname.startswith("scan_session_"):
                    parts = dirname.replace("scan_session_", "").split("_")
                    group = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else dirname
                else:
                    group = reldir.replace(os.sep, "/")
                basename = os.path.basename(rp)
                suffix_parts = basename.replace(".ply", "").split("_")
                suffix = "_".join(suffix_parts[-2:]) if len(suffix_parts) >= 2 else basename
                new_scans.append({
                    "group": group, "suffix": suffix,
                    "path": os.path.join(cache_root, os.path.relpath(rp, path)),
                    "npy": None,
                    "remote_host": host, "remote_path": rp,
                })

        if not new_scans:
            return

        self.scans = new_scans
        self.idx = -1
        self.scan_list.clear()
        for s in self.scans:
            label = f"{s['group']}\n{s['suffix']}"
            if s.get("remote_host"):
                label = f"☁ {label}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.scan_list.addItem(item)

        self.thumb_loader = ThumbnailLoader(self.scans)
        self.thumb_loader.done.connect(self._set_thumbnail)
        self.thumb_loader.start()
        if self.scans:
            self.scan_list.setCurrentRow(0)

    def _on_save_stl(self):
        if self.idx < 0:
            return
        s = self.scans[self.idx]
        default_name = f"column_{s['suffix']}.stl"
        src = os.path.join(BASE, default_name)
        if not os.path.isfile(src):
            comp_src = src.replace(".stl", "_complement.stl")
            if os.path.isfile(comp_src):
                src = comp_src
                default_name = os.path.basename(comp_src)
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Save STL", "No STL file found. Generate one first.")
                return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save STL As", default_name,
            "STL Files (*.stl);;All Files (*)")
        if path:
            import shutil
            shutil.copy2(src, path)
            self.lbl_gen_status.setText(f"Saved → {os.path.basename(path)}")

    def closeEvent(self, e):
        if self.thumb_loader.isRunning():
            self.thumb_loader.quit()
            self.thumb_loader.wait(2000)
        if hasattr(self, '_column_worker') and self._column_worker.isRunning():
            self._column_worker.quit()
            self._column_worker.wait(2000)
        if hasattr(self, '_jit_worker') and self._jit_worker.isRunning():
            self._jit_worker.quit()
            self._jit_worker.wait(2000)
        self.plotter.close()
        e.accept()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec())
