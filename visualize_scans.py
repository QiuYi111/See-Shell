#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from pathlib import Path

BASE_DIR = Path('/Users/qiujingyi.7/See-Shell/oct_cloud')


def parse_grid_idx(s):
    if s.startswith('p'): return int(s[1:])
    elif s.startswith('n'): return -int(s[1:])
    return int(s)


def collect_scans():
    scans = []
    for group, dirname in [('3x3', 'scan_array_20260324_153718'),
                            ('5x5', 'scan_array_20260324_235201')]:
        d = BASE_DIR / dirname
        if not d.exists():
            continue
        for ply in sorted(d.glob('volume_pointcloud_*.ply')):
            if ply.stat().st_size == 0:
                continue
            parts = ply.stem.split('_')
            scans.append(dict(
                group=group,
                suffix=f"{parts[-2]}_{parts[-1]}",
                path=ply,
            ))
    return scans


HIDE = np.array([0.0, 0.0, -1e7])


class Viewer:
    def __init__(self, scans):
        self.scans = scans
        self.idx = 0
        self.full_pts = np.zeros((0, 3))
        self.full_cols = np.zeros((0, 3))
        self.z_lo = self.z_hi = 0.0
        self.z_min_pct = 0.0
        self.z_max_pct = 1.0
        self.alpha = 1.0
        self._added = False

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("OCT Viewer", 1400, 900)
        self.pcd = o3d.geometry.PointCloud()

        self.vis.register_key_callback(262, lambda v: self._nav(1))
        self.vis.register_key_callback(263, lambda v: self._nav(-1))
        self.vis.register_key_callback(ord('W'), lambda v: self._adj_z(0.02))
        self.vis.register_key_callback(ord('S'), lambda v: self._adj_z(-0.02))
        self.vis.register_key_callback(ord('A'), lambda v: self._adj_alpha(-0.1))
        self.vis.register_key_callback(ord('D'), lambda v: self._adj_alpha(0.1))
        self.vis.register_key_callback(ord('Q'), lambda v: self._adj_size(-0.5))
        self.vis.register_key_callback(ord('E'), lambda v: self._adj_size(0.5))
        self.vis.register_key_callback(ord('R'), lambda v: self._reset())
        self.vis.register_key_callback(ord('1'), lambda v: self._adj_zmin(0.02))
        self.vis.register_key_callback(ord('2'), lambda v: self._adj_zmin(-0.02))

        self._load()
        self.vis.run()
        self.vis.destroy_window()

    def _load(self):
        s = self.scans[self.idx]
        print(f'[{self.idx+1}/{len(self.scans)}] {s["group"]} {s["suffix"]}',
              end=' ', flush=True)
        data = o3d.io.read_point_cloud(str(s['path']))
        self.full_pts = np.asarray(data.points).copy()
        self.full_cols = np.asarray(data.colors).copy()
        self.z_lo = self.full_pts[:, 2].min()
        self.z_hi = self.full_pts[:, 2].max()
        self.z_min_pct = 0.0
        self.z_max_pct = 1.0
        self.alpha = 1.0
        print(f'{len(self.full_pts):,} pts  z=[{self.z_lo:.0f}..{self.z_hi:.0f}]')

        if self._added:
            self.vis.remove_geometry(self.pcd, reset_bounding_box=False)

        self._apply_filter()
        self.vis.add_geometry(self.pcd, reset_bounding_box=True)
        self._added = True
        self.vis.get_render_option().point_size = 3.0
        self.vis.get_render_option().background_color = np.array([0.05, 0.05, 0.08])
        self._status()

    def _apply_filter(self):
        z_lo = self.z_lo + self.z_min_pct * (self.z_hi - self.z_lo)
        z_hi = self.z_lo + self.z_max_pct * (self.z_hi - self.z_lo)
        mask = (self.full_pts[:, 2] >= z_lo) & (self.full_pts[:, 2] <= z_hi)

        pts = self.full_pts.copy()
        pts[~mask] = HIDE

        cols = self.full_cols * self.alpha
        cols[~mask] = 0.0

        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(cols)

    def _update(self):
        self._apply_filter()
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self._status()

    def _status(self):
        z_lo = self.z_lo + self.z_min_pct * (self.z_hi - self.z_lo)
        z_hi = self.z_lo + self.z_max_pct * (self.z_hi - self.z_lo)
        mask = (self.full_pts[:, 2] >= z_lo) & (self.full_pts[:, 2] <= z_hi)
        s = self.scans[self.idx]
        n_vis = int(mask.sum())
        print(f'  {s["group"]} {s["suffix"]}  z:[{z_lo:.0f},{z_hi:.0f}]  '
              f'a:{self.alpha:.2f}  {n_vis:,}/{len(self.full_pts):,}  '
              f'[{self.idx+1}/{len(self.scans)}]')

    def _nav(self, d):
        self.idx = (self.idx + d) % len(self.scans)
        self._load()

    def _adj_z(self, d):
        self.z_max_pct = np.clip(self.z_max_pct + d, self.z_min_pct + 0.02, 1.0)
        self._update()

    def _adj_zmin(self, d):
        self.z_min_pct = np.clip(self.z_min_pct + d, 0.0, self.z_max_pct - 0.02)
        self._update()

    def _adj_alpha(self, d):
        self.alpha = np.clip(self.alpha + d, 0.05, 1.0)
        self._update()

    def _adj_size(self, d):
        ro = self.vis.get_render_option()
        ro.point_size = max(0.5, ro.point_size + d)
        print(f'  pt_size={ro.point_size:.1f}')

    def _reset(self):
        self.z_min_pct = 0.0
        self.z_max_pct = 1.0
        self.alpha = 1.0
        self._update()


HELP = """
Controls:
  Right/Left   Next / Prev scan
  W / S        Z max  +/-
  1 / 2        Z min  +/-
  D / A        Alpha  +/-
  E / Q        Point size  +/-
  R            Reset all
"""


if __name__ == '__main__':
    scans = collect_scans()
    print(f'{len(scans)} scans found')
    print(HELP)
    Viewer(scans)
