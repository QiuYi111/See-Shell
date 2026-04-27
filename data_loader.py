#!/usr/bin/env python3
"""Data source picker for See-Shell — choose local folder or remote SSH path.

Returns ``(host: str | None, path: str)`` via ``dialog.result_data``.
  - Local:  ``(None, "/local/folder/path")``
  - Remote: ``("jingyi-lab", "/mnt/data/oct")``
"""

import os
os.environ["QT_API"] = "pyqt6"

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QTabWidget, QVBoxLayout, QWidget,
)

try:
    from ssh_utils import parse_ssh_config, list_remote_dir
    _SSH = True
except ImportError:
    _SSH = False

_STYLE = """
QPushButton { padding: 5px 12px; border: 1px solid #45475A; border-radius: 3px;
              background: #585B70; color: #CDD6F4; }
QPushButton:hover { background: #6C7086; }
QPushButton:pressed { background: #45475A; }
QPushButton:disabled { background: #313244; color: #6C7086; }
QLineEdit { padding: 4px 8px; border: 1px solid #45475A; border-radius: 3px;
            background: #313244; color: #CDD6F4; }
QComboBox { padding: 4px 8px; border: 1px solid #45475A; border-radius: 3px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #2A2A3C; color: #CDD6F4;
                              selection-background-color: #45475A; }
QListWidget { background: #313244; color: #CDD6F4; border: 1px solid #45475A;
              border-radius: 3px; }
QListWidget::item:selected { background: #45475A; }
QTabWidget::pane { border: 1px solid #45475A; background: #1E1E2E; }
QTabBar::tab { background: #313244; color: #CDD6F4; padding: 6px 16px;
               border: 1px solid #45475A; border-bottom: none;
               border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: #2A2A3C; border-bottom: 2px solid #89B4FA; }
QLabel { color: #CDD6F4; }
"""


class DataSourceDialog(QDialog):
    """Pick a data source: local folder or remote SSH directory.

    After accept, read ``dialog.result_data`` → ``(host | None, path)``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Data")
        self.resize(600, 420)
        self.result_data: tuple[str | None, str] = (None, "")

        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#1E1E2E"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#CDD6F4"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#313244"))
        pal.setColor(QPalette.ColorRole.Text, QColor("#CDD6F4"))
        pal.setColor(QPalette.ColorRole.Button, QColor("#585B70"))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor("#CDD6F4"))
        pal.setColor(QPalette.ColorRole.Highlight, QColor("#89B4FA"))
        self.setPalette(pal)
        self.setStyleSheet(_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        tabs = QTabWidget()
        tabs.addTab(self._local_tab(), "Local Folder")
        tabs.addTab(self._remote_tab() if _SSH else self._no_ssh_tab(), "Remote (SSH)")
        self._tabs = tabs
        layout.addWidget(tabs)

        row = QHBoxLayout()
        row.addStretch()
        self._btn_ok = QPushButton("Open")
        self._btn_ok.setFixedWidth(100)
        self._btn_ok.setEnabled(False)
        self._btn_ok.clicked.connect(self._accept)
        row.addWidget(self._btn_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedWidth(100)
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)
        layout.addLayout(row)

    # ── Local tab ──

    def _local_tab(self):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(8, 12, 8, 8)

        row = QHBoxLayout()
        self._local_path = QLineEdit()
        self._local_path.setReadOnly(True)
        self._local_path.setPlaceholderText("Select a folder…")
        row.addWidget(self._local_path, stretch=1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_local)
        row.addWidget(btn)
        vbox.addLayout(row)

        self._local_summary = QLabel("")
        self._local_summary.setStyleSheet("color: #A6ADC8;")
        vbox.addWidget(self._local_summary)
        vbox.addStretch()
        return tab

    def _browse_local(self):
        path = QFileDialog.getExistingDirectory(self, "Select OCT Data Folder")
        if not path:
            return
        self._local_path.setText(path)
        import glob
        ply = [f for f in glob.glob(os.path.join(path, "**", "*.ply"), recursive=True)
               if os.path.getsize(f) > 0]
        self._local_summary.setText(f"Found {len(ply)} PLY file(s)")
        self._btn_ok.setEnabled(len(ply) > 0)

    # ── Remote tab ──

    def _remote_tab(self):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(8, 12, 8, 8)

        row_host = QHBoxLayout()
        row_host.addWidget(QLabel("Host:"))
        self._combo_host = QComboBox()
        for entry in parse_ssh_config():
            label = entry["host"]
            if entry.get("hostname"):
                label += f"  ({entry['hostname']})"
            self._combo_host.addItem(label, entry)
        row_host.addWidget(self._combo_host, stretch=1)
        vbox.addLayout(row_host)

        row_path = QHBoxLayout()
        row_path.addWidget(QLabel("Path:"))
        self._remote_path = QLineEdit()
        self._remote_path.setPlaceholderText("/mnt/data/oct")
        self._remote_path.returnPressed.connect(self._browse_remote)
        row_path.addWidget(self._remote_path, stretch=1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_remote)
        row_path.addWidget(btn_browse)
        vbox.addLayout(row_path)

        self._remote_list = QListWidget()
        self._remote_list.itemDoubleClicked.connect(self._remote_dblclick)
        vbox.addWidget(self._remote_list, stretch=1)

        self._remote_status = QLabel("")
        self._remote_status.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        vbox.addWidget(self._remote_status)

        self._remote_dir: str = ""
        return tab

    def _no_ssh_tab(self):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        lbl = QLabel("ssh_utils not found. Remote unavailable.")
        lbl.setStyleSheet("color: #F38BA8;")
        vbox.addWidget(lbl)
        vbox.addStretch()
        return tab

    def _host(self) -> str:
        entry = self._combo_host.currentData()
        return entry["host"] if isinstance(entry, dict) else ""

    def _browse_remote(self):
        host = self._host()
        path = self._remote_path.text().strip() or "/"
        self._remote_status.setText("Loading…")
        self._remote_list.clear()
        try:
            entries = list_remote_dir(host, path)
        except RuntimeError as exc:
            self._remote_status.setText(f"Error: {exc}")
            return
        self._remote_dir = path
        self._remote_status.setText(f"{len(entries)} items  —  double-click to open, then press Open")

        parent = QListWidgetItem("📁 ..")
        parent.setData(Qt.ItemDataRole.UserRole, dict(name="..", is_dir=True))
        self._remote_list.addItem(parent)

        for e in sorted(entries, key=lambda x: (not x["is_dir"], x["name"].lower())):
            prefix = "📁 " if e["is_dir"] else "📄 "
            self._remote_list.addItem(f"{prefix}{e['name']}")

        self._btn_ok.setEnabled(True)

    def _remote_dblclick(self, item):
        text = item.text()
        if ".." in text:
            import os.path
            parent = os.path.dirname(self._remote_dir.rstrip("/"))
            if parent:
                self._remote_path.setText(parent)
                self._browse_remote()
            return
        name = text.lstrip("📁📄 ")
        new_path = f"{self._remote_dir.rstrip('/')}/{name}"
        if text.startswith("📁"):
            self._remote_path.setText(new_path)
            self._browse_remote()

    # ── Accept ──

    def _accept(self):
        idx = self._tabs.currentIndex()
        if idx == 0:
            path = self._local_path.text().strip()
            if path:
                self.result_data = (None, path)
                self.accept()
        elif idx == 1:
            host = self._host()
            path = self._remote_dir or self._remote_path.text().strip()
            if host and path:
                self.result_data = (host, path)
                self.accept()


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    d = DataSourceDialog()
    if d.exec() == QDialog.DialogCode.Accepted:
        print(d.result_data)
