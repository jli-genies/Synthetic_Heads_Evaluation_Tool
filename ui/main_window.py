"""Main window for the Synthetic Heads asset tagging tool."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import QDir, Qt
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

try:
    from .render_panel import IMAGE_EXTENSIONS, RenderPanel
    from .tag_panel import TagPanel
except ImportError:  # Allow running this file directly.
    from render_panel import IMAGE_EXTENSIONS, RenderPanel
    from tag_panel import TagPanel


ASSET_EXTENSIONS = {".blend", ".fbx", ".obj", ".gltf", ".glb", ".usd", ".usda", ".usdc"}
SELECTABLE_EXTENSIONS = ASSET_EXTENSIONS | IMAGE_EXTENSIONS


class MainWindow(QMainWindow):
    """Coordinates file browsing, render previews, and sidecar tag files."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Synthetic Heads — Asset Tagger")
        self.resize(1440, 860)
        self.setMinimumSize(1050, 650)

        self.project_root = Path(__file__).resolve().parents[1]
        self.root_path: Path | None = None
        self.current_asset: Path | None = None
        self.assets: list[Path] = []

        self.file_model = QFileSystemModel(self)
        self.file_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)

        self.root_button = QPushButton("Choose root folder…")
        self.root_button.setObjectName("rootButton")
        self.root_button.clicked.connect(self.choose_root_folder)

        self.root_label = QLabel("No folder selected")
        self.root_label.setObjectName("rootLabel")
        self.root_label.setWordWrap(True)

        self.tree = QTreeView()
        self.tree.setModel(self.file_model)
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(16)
        for column in range(1, 4):
            self.tree.hideColumn(column)
        selection_model = self.tree.selectionModel()
        if selection_model is None:
            raise RuntimeError("Tree view has no selection model after setModel().")
        selection_model.currentChanged.connect(self._tree_selection_changed)

        browser = QWidget()
        browser.setObjectName("browserPanel")
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(12, 12, 12, 12)
        browser_layout.addWidget(self.root_button)
        browser_layout.addWidget(self.root_label)
        browser_layout.addWidget(self.tree, 1)

        self.render_panel = RenderPanel()
        self.render_panel.previous_requested.connect(lambda: self._navigate(-1))
        self.render_panel.next_requested.connect(lambda: self._navigate(1))
        self.render_panel.download_requested.connect(self.download_asset)

        self.tag_panel = TagPanel(self.project_root / "tag_schema.json")
        self.tag_panel.submit_requested.connect(self.save_tags)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(browser)
        splitter.addWidget(self.render_panel)
        splitter.addWidget(self.tag_panel)
        splitter.setSizes([250, 670, 430])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        self.setCentralWidget(splitter)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Choose a folder containing head assets.")
        self.setStyleSheet(STYLE_SHEET)

    def choose_root_folder(self) -> None:
        start = str(self.root_path or self.project_root)
        selected = QFileDialog.getExistingDirectory(self, "Choose asset root folder", start)
        if selected:
            self.set_root_folder(Path(selected))

    def set_root_folder(self, root_path: Path) -> None:
        self.root_path = root_path.resolve()
        self.root_label.setText(str(self.root_path))
        self.root_label.setToolTip(str(self.root_path))
        root_index = self.file_model.setRootPath(str(self.root_path))
        self.tree.setRootIndex(root_index)
        self.assets = sorted(
            path
            for path in self.root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SELECTABLE_EXTENSIONS
        )
        self._select_asset(None)
        self.status_bar.showMessage(f"Found {len(self.assets)} selectable assets.")

    def _tree_selection_changed(self, current, _previous) -> None:
        path = Path(self.file_model.filePath(current))
        if path.is_file() and path.suffix.lower() in SELECTABLE_EXTENSIONS:
            self._select_asset(path)

    def _select_asset(self, asset_path: Path | None) -> None:
        self.current_asset = asset_path
        self.render_panel.set_asset(asset_path)
        self.tag_panel.set_asset(asset_path.name if asset_path else None)

        if asset_path:
            self.tag_panel.set_tags(self._load_tags(asset_path))
            self.status_bar.showMessage(f"Selected {asset_path.name}")
        else:
            self.tag_panel.clear()

        try:
            index = self.assets.index(asset_path) if asset_path else -1
        except ValueError:
            index = -1
        self.render_panel.set_navigation_enabled(index > 0, 0 <= index < len(self.assets) - 1)

    def _navigate(self, offset: int) -> None:
        if self.current_asset not in self.assets:
            return
        target_index = self.assets.index(self.current_asset) + offset
        if not 0 <= target_index < len(self.assets):
            return
        target = self.assets[target_index]
        model_index = self.file_model.index(str(target))
        self.tree.setCurrentIndex(model_index)
        self.tree.scrollTo(model_index)

    def save_tags(self, tags: dict) -> None:
        if not self.current_asset:
            return

        payload = {
            "schema_version": self.tag_panel.schema.get("schema_version"),
            "asset": self.current_asset.name,
            "tags": tags,
        }
        tag_path = self._tag_path(self.current_asset)
        try:
            tag_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            QMessageBox.critical(self, "Unable to save tags", str(error))
            return

        self.tag_panel.status_label.setText(f"Saved: {tag_path.name}")
        self.status_bar.showMessage(f"Tags saved to {tag_path.name}", 5000)

    def download_asset(self) -> None:
        if not self.current_asset:
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Download selected asset",
            self.current_asset.name,
            f"Asset (*{self.current_asset.suffix});;All files (*)",
        )
        if not destination:
            return
        try:
            shutil.copy2(self.current_asset, destination)
        except OSError as error:
            QMessageBox.critical(self, "Unable to download asset", str(error))
            return
        self.status_bar.showMessage(f"Copied asset to {destination}", 5000)

    @staticmethod
    def _tag_path(asset_path: Path) -> Path:
        return asset_path.with_name(f"{asset_path.name}.tags.json")

    def _load_tags(self, asset_path: Path) -> dict:
        tag_path = self._tag_path(asset_path)
        if not tag_path.exists():
            return {}
        try:
            payload = json.loads(tag_path.read_text(encoding="utf-8"))
            return payload.get("tags", {})
        except (OSError, json.JSONDecodeError) as error:
            self.status_bar.showMessage(f"Could not read {tag_path.name}: {error}", 7000)
            return {}


STYLE_SHEET = """
QMainWindow, QWidget {
    background: #f5f7fa;
    color: #202936;
    font-family: "Segoe UI";
    font-size: 13px;
}
QWidget#browserPanel {
    background: #eef1f5;
}
QLabel#sectionTitle {
    font-size: 22px;
    font-weight: 650;
    color: #162033;
    padding: 2px 0 6px;
}
QLabel#assetName {
    color: #556176;
    font-size: 14px;
    padding-bottom: 4px;
}
QLabel#rootLabel, QLabel#tagStatus {
    color: #667085;
    font-size: 12px;
    padding: 3px;
}
QTreeView {
    background: #ffffff;
    border: 1px solid #d5dae3;
    border-radius: 7px;
    padding: 4px;
}
QTreeView::item {
    min-height: 25px;
}
QTreeView::item:selected {
    background: #dce8ff;
    color: #173f85;
}
QFrame#renderTile {
    background: #ffffff;
    border: 1px solid #cdd3dd;
    border-radius: 8px;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d9dee7;
    border-radius: 7px;
    font-weight: 600;
    margin-top: 11px;
    padding: 12px 8px 7px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QComboBox {
    background: #ffffff;
    border: 1px solid #bfc7d4;
    border-radius: 5px;
    min-height: 28px;
    padding: 0 7px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #bfc7d4;
    border-radius: 6px;
    min-height: 32px;
    padding: 0 13px;
}
QPushButton:hover {
    background: #edf3ff;
    border-color: #7da2e8;
}
QPushButton:disabled {
    color: #9aa3b2;
    background: #eef0f3;
}
QPushButton#primaryButton, QPushButton#rootButton {
    background: #315fbd;
    border-color: #315fbd;
    color: white;
    font-weight: 600;
}
QPushButton#primaryButton:hover, QPushButton#rootButton:hover {
    background: #264e9f;
}
QSplitter::handle {
    background: #cbd1da;
    width: 1px;
}
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Synthetic Heads Asset Tagger")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
