"""Render preview and asset navigation widgets."""

from __future__ import annotations

import json
import time
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "debug-c44263.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "c44263",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# #endregion


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PREVIEW_SIZE = 512  # Fixed square display size for each render tile.
VIEW_NAMES = (
    ("Front view", ("front",)),
    ("Side view R", ("side_r", "right", "profile_r")),
)


class RenderTile(QFrame):
    """A fixed-square preview tile for one rendered view."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setObjectName("renderTile")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.image = QLabel(title)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setWordWrap(True)
        self.image.setFixedSize(PREVIEW_SIZE - 16, PREVIEW_SIZE - 16)
        self.image.setStyleSheet("color: #7b8494; font-size: 14px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.image, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_image(self, path: Path | None, empty_text: str) -> None:
        pixmap = QPixmap(str(path)) if path else QPixmap()
        # #region agent log
        _agent_log(
            "D",
            "render_panel.py:RenderTile.set_image",
            "pixmap load result",
            {
                "path": str(path) if path else None,
                "path_exists": bool(path and path.is_file()),
                "pixmap_null": pixmap.isNull(),
                "empty_text": empty_text,
                "label_size": [self.image.width(), self.image.height()],
                "tile_size": [self.width(), self.height()],
            },
        )
        # #endregion
        if pixmap.isNull():
            self._pixmap = None
            self.image.setPixmap(QPixmap())
            self.image.setText(empty_text)
            self.image.setToolTip("")
            return

        self._pixmap = pixmap
        self.image.setText("")
        self.image.setToolTip(str(path))
        self._update_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap:
            self.image.setPixmap(
                self._pixmap.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class RenderPanel(QWidget):
    """Displays front and side-right renders and emits navigation requests."""

    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    download_requested = pyqtSignal()

    def __init__(
        self,
        project_root: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self._asset_path: Path | None = None

        title = QLabel("Renders")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.asset_name = QLabel("Select an asset")
        self.asset_name.setObjectName("assetName")
        self.asset_name.setAlignment(Qt.AlignmentFlag.AlignCenter)

        grid = QHBoxLayout()
        grid.setSpacing(10)
        grid.addStretch(1)
        self.tiles: list[RenderTile] = []
        for view_name, _ in VIEW_NAMES:
            tile = RenderTile(view_name)
            self.tiles.append(tile)
            grid.addWidget(tile, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addStretch(1)

        self.download_button = QPushButton("Download selected asset")
        self.download_button.setObjectName("primaryButton")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_requested)

        self.previous_button = QPushButton("←  Back")
        self.next_button = QPushButton("Forward  →")
        self.previous_button.clicked.connect(self.previous_requested)
        self.next_button.clicked.connect(self.next_requested)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)

        navigation = QHBoxLayout()
        navigation.addWidget(self.previous_button)
        navigation.addStretch()
        navigation.addWidget(self.next_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addWidget(title)
        layout.addWidget(self.asset_name)
        layout.addStretch(1)
        layout.addLayout(grid)
        layout.addStretch(1)
        layout.addWidget(self.download_button)
        layout.addLayout(navigation)

    @property
    def asset_path(self) -> Path | None:
        return self._asset_path

    def set_asset(self, asset_path: str | Path | None) -> None:
        self._asset_path = Path(asset_path) if asset_path else None
        self.download_button.setEnabled(bool(self._asset_path and self._asset_path.is_file()))
        # #region agent log
        _agent_log(
            "A",
            "render_panel.py:RenderPanel.set_asset",
            "set_asset called",
            {
                "input": str(asset_path) if asset_path else None,
                "resolved": str(self._asset_path) if self._asset_path else None,
                "is_file": bool(self._asset_path and self._asset_path.is_file()),
                "stem": self._asset_path.stem if self._asset_path else None,
                "project_root": str(self.project_root),
                "cache_dir": str(self.project_root / "renders" / self._asset_path.stem)
                if self._asset_path
                else None,
                "cache_exists": (
                    (self.project_root / "renders" / self._asset_path.stem).is_dir()
                    if self._asset_path
                    else False
                ),
                "panel_size": [self.width(), self.height()],
            },
        )
        # #endregion

        if not self._asset_path:
            self.asset_name.setText("Select an asset")
            for tile, (view_name, _) in zip(self.tiles, VIEW_NAMES):
                tile.set_image(None, view_name)
            return

        self.asset_name.setText(self._asset_path.stem)
        images = self._find_render_images(self._asset_path)
        for tile, (view_name, keywords) in zip(self.tiles, VIEW_NAMES):
            match = next(
                (image for image in images if any(word in image.stem.lower() for word in keywords)),
                None,
            )
            # #region agent log
            _agent_log(
                "C",
                "render_panel.py:RenderPanel.set_asset",
                "view match result",
                {
                    "view_name": view_name,
                    "keywords": list(keywords),
                    "match": str(match) if match else None,
                    "images_found": [str(p) for p in images],
                    "image_stems": [p.stem for p in images],
                },
            )
            # #endregion
            tile.set_image(match, f"{view_name}\nNo render found")

    def set_navigation_enabled(self, has_previous: bool, has_next: bool) -> None:
        self.previous_button.setEnabled(has_previous)
        self.next_button.setEnabled(has_next)

    def _find_render_images(self, asset_path: Path) -> list[Path]:
        if asset_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [asset_path]

        # Prefer the centralized repo cache, then fall back to legacy local folders.
        search_directories = [
            self.project_root / "renders" / asset_path.stem,
            asset_path.parent / "renders" / asset_path.stem,
            asset_path.parent / f"{asset_path.stem}_renders",
            asset_path.parent / "renders",
            asset_path.parent / asset_path.stem,
            asset_path.parent,
        ]
        images: list[Path] = []
        dir_status = []
        for directory in search_directories:
            exists = directory.is_dir()
            dir_status.append({"dir": str(directory), "exists": exists})
            if not exists:
                continue
            for candidate in directory.iterdir():
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                    if directory != asset_path.parent or asset_path.stem.lower() in candidate.stem.lower():
                        images.append(candidate)
        result = sorted(set(images))
        # #region agent log
        _agent_log(
            "B",
            "render_panel.py:RenderPanel._find_render_images",
            "search directories and results",
            {
                "asset_path": str(asset_path),
                "stem": asset_path.stem,
                "directories": dir_status,
                "images": [str(p) for p in result],
            },
        )
        # #endregion
        return result
