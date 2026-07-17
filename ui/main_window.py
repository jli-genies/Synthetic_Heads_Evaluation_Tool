"""Main window for the Synthetic Heads asset tagging tool."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
)

try:
    from .asset_tree import AssetTree
    from .render_panel import RenderPanel
    from .tag_panel import TagPanel
except ImportError:  # Allow running this file directly.
    from asset_tree import AssetTree
    from render_panel import RenderPanel
    from tag_panel import TagPanel


class MainWindow(QMainWindow):
    """Coordinates asset browsing, render previews, and sidecar tag files."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Synthetic Heads — Asset Tagger")
        self.resize(1440, 860)
        self.setMinimumSize(1050, 650)

        self.project_root = Path(__file__).resolve().parents[1]
        self.current_asset: Path | None = None

        self.asset_tree = AssetTree(project_root=self.project_root)
        self.asset_tree.asset_selected.connect(self._select_asset)
        self.asset_tree.status_message.connect(self.status_bar_message)
        self.asset_tree.render_finished.connect(self._on_render_finished)

        self.render_panel = RenderPanel(project_root=self.project_root)
        self.render_panel.previous_requested.connect(lambda: self._navigate(-1))
        self.render_panel.next_requested.connect(lambda: self._navigate(1))
        self.render_panel.download_requested.connect(self.download_asset)

        self.tag_panel = TagPanel(self.project_root / "tag_schema.json")
        self.tag_panel.submit_requested.connect(self.save_tags)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.asset_tree)
        splitter.addWidget(self.render_panel)
        splitter.addWidget(self.tag_panel)
        splitter.setSizes([280, 650, 430])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        self.setCentralWidget(splitter)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Choose a folder containing .glb / .fbx assets.")
        self.setStyleSheet(STYLE_SHEET)

    def status_bar_message(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def _on_render_finished(self, asset_path: Path) -> None:
        # Refresh previews once Blender writes renders/<stem>/.
        if self.current_asset and self.current_asset.resolve() == Path(asset_path).resolve():
            self.render_panel.set_asset(self.current_asset)

    def _select_asset(self, asset_path: Path | None) -> None:
        self.current_asset = Path(asset_path) if asset_path else None
        # #region agent log
        try:
            import json
            import time
            from pathlib import Path as _P

            _log = _P(__file__).resolve().parents[1] / "debug-c44263.log"
            with _log.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "sessionId": "c44263",
                            "runId": "pre-fix",
                            "hypothesisId": "A",
                            "location": "main_window.py:_select_asset",
                            "message": "main window selecting asset",
                            "data": {
                                "asset_path": str(asset_path) if asset_path else None,
                                "current_asset": str(self.current_asset)
                                if self.current_asset
                                else None,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
        # #endregion
        self.render_panel.set_asset(self.current_asset)
        self.tag_panel.set_asset(self.current_asset.name if self.current_asset else None)

        if self.current_asset:
            self.tag_panel.set_tags(self._load_tags(self.current_asset))
            self.status_bar.showMessage(f"Selected {self.current_asset.name}")
        else:
            self.tag_panel.clear()

        assets = self.asset_tree.assets
        try:
            index = assets.index(self.current_asset) if self.current_asset else -1
        except ValueError:
            # Paths may differ by resolve(); compare resolved forms.
            index = -1
            if self.current_asset:
                resolved = self.current_asset.resolve()
                for i, path in enumerate(assets):
                    if path.resolve() == resolved:
                        index = i
                        break
        self.render_panel.set_navigation_enabled(index > 0, 0 <= index < len(assets) - 1)

    def _navigate(self, offset: int) -> None:
        assets = self.asset_tree.assets
        if not self.current_asset or not assets:
            return
        resolved = self.current_asset.resolve()
        try:
            current_index = next(i for i, path in enumerate(assets) if path.resolve() == resolved)
        except StopIteration:
            return
        target_index = current_index + offset
        if not 0 <= target_index < len(assets):
            return
        self.asset_tree.select_asset(assets[target_index])

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
QTreeView, QTreeWidget {
    background: #ffffff;
    border: 1px solid #d5dae3;
    border-radius: 7px;
    padding: 4px;
}
QTreeView::item, QTreeWidget::item {
    min-height: 25px;
}
QTreeView::item:selected, QTreeWidget::item:selected {
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
