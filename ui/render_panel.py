"""Render preview and asset navigation widgets."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIEW_NAMES = (
    ("Front view", ("front",)),
    ("Side view L", ("side_l", "left", "profile_l")),
    ("Side view R", ("side_r", "right", "profile_r")),
    ("Back", ("back", "rear")),
)


class RenderTile(QFrame):
    """A labeled image tile that scales its pixmap with the window."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setObjectName("renderTile")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(220, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.image = QLabel(title)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setWordWrap(True)
        self.image.setStyleSheet("color: #7b8494; font-size: 14px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.image)

    def set_image(self, path: Path | None, empty_text: str) -> None:
        pixmap = QPixmap(str(path)) if path else QPixmap()
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
            size = self.image.size()
            self.image.setPixmap(
                self._pixmap.scaled(
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class RenderPanel(QWidget):
    """Displays four standard head renders and emits navigation requests."""

    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    download_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._asset_path: Path | None = None

        title = QLabel("Renders")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.asset_name = QLabel("Select an asset")
        self.asset_name.setObjectName("assetName")
        self.asset_name.setAlignment(Qt.AlignmentFlag.AlignCenter)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.tiles: list[RenderTile] = []
        for index, (view_name, _) in enumerate(VIEW_NAMES):
            tile = RenderTile(view_name)
            self.tiles.append(tile)
            grid.addWidget(tile, index // 2, index % 2)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

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
        layout.addLayout(grid, 1)
        layout.addWidget(self.download_button)
        layout.addLayout(navigation)

    @property
    def asset_path(self) -> Path | None:
        return self._asset_path

    def set_asset(self, asset_path: str | Path | None) -> None:
        self._asset_path = Path(asset_path) if asset_path else None
        self.download_button.setEnabled(bool(self._asset_path and self._asset_path.is_file()))

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
            tile.set_image(match, f"{view_name}\nNo render found")

    def set_navigation_enabled(self, has_previous: bool, has_next: bool) -> None:
        self.previous_button.setEnabled(has_previous)
        self.next_button.setEnabled(has_next)

    @staticmethod
    def _find_render_images(asset_path: Path) -> list[Path]:
        if asset_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [asset_path]

        search_directories = [
            asset_path.parent,
            asset_path.parent / "renders",
            asset_path.parent / asset_path.stem,
            asset_path.parent / f"{asset_path.stem}_renders",
        ]
        images: list[Path] = []
        for directory in search_directories:
            if not directory.is_dir():
                continue
            for candidate in directory.iterdir():
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                    if directory != asset_path.parent or asset_path.stem.lower() in candidate.stem.lower():
                        images.append(candidate)
        return sorted(set(images))