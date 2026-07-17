"""Asset browser: scan a folder for .glb/.fbx and load/render selected assets."""

from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6.QtCore import Qt, QProcess, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


ASSET_EXTENSIONS = {".glb", ".fbx"}
PATH_ROLE = Qt.ItemDataRole.UserRole
RENDERED_ROLE = Qt.ItemDataRole.UserRole + 1


def find_blender_executable() -> Path | None:
    """Locate a Blender binary (PATH first, then common Windows installs)."""
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)

    program_files = Path(r"C:\Program Files\Blender Foundation")
    if program_files.is_dir():
        installs = sorted(program_files.glob("Blender */blender.exe"), reverse=True)
        if installs:
            return installs[0]
    return None


def collect_assets(root: Path) -> list[Path]:
    """Return sorted .glb / .fbx files under ``root`` (recursive)."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS
    )


class AssetTree(QWidget):
    """Left-panel asset tree with folder scan and Blender Load/render."""

    asset_selected = pyqtSignal(object)  # Path | None
    status_message = pyqtSignal(str)
    assets_changed = pyqtSignal()
    render_finished = pyqtSignal(object)  # Path of rendered asset

    def __init__(self, project_root: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.root_path: Path | None = None
        self.assets: list[Path] = []
        self._folder_items: dict[Path, QTreeWidgetItem] = {}
        self._process: QProcess | None = None
        self._pending_asset: Path | None = None
        self._render_queue: list[Path] = []
        self._batch_total = 0
        self._batch_failed = 0

        self.setObjectName("browserPanel")
        self.setMinimumWidth(220)

        self.root_button = QPushButton("Choose root folder…")
        self.root_button.setObjectName("rootButton")
        self.root_button.clicked.connect(self.choose_root_folder)

        self.root_label = QLabel("No folder selected")
        self.root_label.setObjectName("rootLabel")
        self.root_label.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(16)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)

        self.load_button = QPushButton("Load / render selected")
        self.load_button.setObjectName("primaryButton")
        self.load_button.setEnabled(False)
        self.load_button.setToolTip(
            "Run Blender (render_head.py) for the selected .glb/.fbx into renders/<stem>/."
        )
        self.load_button.clicked.connect(self.load_selected_asset)

        self.load_folder_button = QPushButton("Load / render folder")
        self.load_folder_button.setObjectName("primaryButton")
        self.load_folder_button.setEnabled(False)
        self.load_folder_button.setToolTip(
            "Queue every .glb/.fbx in this folder. Assets that already have "
            "front + side_r previews are skipped unless you choose to re-render."
        )
        self.load_folder_button.clicked.connect(self.load_folder_assets)

        self.cancel_button = QPushButton("Cancel batch")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_batch)

        self.hint_label = QLabel(
            "Choose a folder to list assets. Use Load / render folder for the whole set, "
            "or select one asset and Load / render selected."
        )
        self.hint_label.setObjectName("rootLabel")
        self.hint_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.root_button)
        layout.addWidget(self.root_label)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.load_folder_button)
        layout.addWidget(self.load_button)
        layout.addWidget(self.cancel_button)

    @property
    def current_asset(self) -> Path | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        path = item.data(0, PATH_ROLE)
        return Path(path) if path else None

    def choose_root_folder(self) -> None:
        start = str(self.root_path or self.project_root)
        selected = QFileDialog.getExistingDirectory(self, "Choose asset root folder", start)
        if selected:
            self.set_root_folder(Path(selected))

    def set_root_folder(self, root_path: Path) -> None:
        if self._is_busy():
            QMessageBox.warning(self, "Render in progress", "Cancel the batch before changing folders.")
            return
        self.root_path = root_path.resolve()
        self.root_label.setText(str(self.root_path))
        self.root_label.setToolTip(str(self.root_path))
        self.assets = collect_assets(self.root_path)
        self._rebuild_tree()
        self.asset_selected.emit(None)
        self.assets_changed.emit()
        self.load_folder_button.setEnabled(bool(self.assets))
        self.status_message.emit(f"Found {len(self.assets)} .glb/.fbx assets.")

    def select_asset(self, asset_path: Path | None) -> None:
        """Programmatically select an asset in the tree (e.g. back/forward)."""
        if asset_path is None:
            self.tree.clearSelection()
            self.tree.setCurrentItem(None)
            self._update_action_buttons()
            self.asset_selected.emit(None)
            return

        target = asset_path.resolve()
        for item in self._iter_asset_items():
            path = item.data(0, PATH_ROLE)
            if path and Path(path).resolve() == target:
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                return

    def load_selected_asset(self) -> None:
        """Render the selected asset with Blender, then keep/refresh it in the tree."""
        asset = self.current_asset
        if asset is None:
            QMessageBox.information(self, "No asset selected", "Select a .glb or .fbx first.")
            return
        if self._is_busy():
            QMessageBox.warning(self, "Render in progress", "Wait for the current Load to finish.")
            return
        self._start_queue([asset])

    def load_folder_assets(self) -> None:
        """Queue renders for every asset in the chosen folder (skip existing by default)."""
        if not self.assets:
            QMessageBox.information(self, "No assets", "Choose a root folder with .glb/.fbx files first.")
            return
        if self._is_busy():
            QMessageBox.warning(self, "Render in progress", "Wait for the current batch to finish.")
            return

        missing = [asset for asset in self.assets if not self._has_renders(asset)]
        already = len(self.assets) - len(missing)

        if not missing:
            answer = QMessageBox.question(
                self,
                "All assets already rendered",
                f"All {len(self.assets)} assets already have previews in renders/.\n\n"
                "Re-render the entire folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            queue = list(self.assets)
        else:
            message = (
                f"Found {len(self.assets)} assets.\n"
                f"{len(missing)} need rendering"
                + (f"; {already} already have previews and will be skipped." if already else ".")
                + "\n\nStart batch render for missing assets?"
            )
            if already:
                message += "\n\nChoose Yes = missing only, No = cancel.\nUse Yes after clearing renders/ to force all."
            answer = QMessageBox.question(
                self,
                "Load / render folder",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            queue = missing

        self._start_queue(queue)

    def cancel_batch(self) -> None:
        """Stop queued renders; the current Blender job is left to finish."""
        skipped = len(self._render_queue)
        self._render_queue.clear()
        self.cancel_button.setEnabled(False)
        if skipped:
            self.status_message.emit(
                f"Batch cancelled ({skipped} queued assets skipped). Current render may still finish."
            )
        else:
            self.status_message.emit("No queued assets left to cancel.")

    def refresh_render_status(self) -> None:
        for item in self._iter_asset_items():
            path = item.data(0, PATH_ROLE)
            if not path:
                continue
            asset = Path(path)
            rendered = self._has_renders(asset)
            item.setData(0, RENDERED_ROLE, rendered)
            item.setText(0, self._item_label(asset, rendered))
            item.setForeground(
                0,
                QBrush(QColor("#1b7a3d" if rendered else "#202936")),
            )

    def _has_renders(self, asset: Path) -> bool:
        cache = self.project_root / "renders" / asset.stem
        return (cache / "front.png").is_file() and (cache / "side_r.png").is_file()

    @staticmethod
    def _item_label(asset: Path, rendered: bool) -> str:
        mark = "[x] " if rendered else "[ ] "
        return f"{mark}{asset.name}"

    def _is_busy(self) -> bool:
        return bool(self._render_queue) or (
            self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning
        )

    def _update_action_buttons(self) -> None:
        busy = self._is_busy()
        self.load_button.setEnabled(self.current_asset is not None and not busy)
        self.load_folder_button.setEnabled(bool(self.assets) and not busy)
        self.cancel_button.setEnabled(busy and bool(self._render_queue))
        self.root_button.setEnabled(not busy)

    def _start_queue(self, assets: list[Path]) -> None:
        if not assets:
            return
        blender = find_blender_executable()
        if blender is None:
            QMessageBox.critical(
                self,
                "Blender not found",
                "Could not find blender.exe. Install Blender or add it to PATH.",
            )
            return

        scene = self.project_root / "blender" / "cameraSetup.blend"
        script = self.project_root / "blender" / "render_head.py"
        if not scene.is_file() or not script.is_file():
            QMessageBox.critical(
                self,
                "Missing render files",
                f"Expected:\n{scene}\n{script}",
            )
            return

        self._render_queue = list(assets)
        self._batch_total = len(assets)
        self._batch_failed = 0
        self._update_action_buttons()
        self.status_message.emit(f"Queued {self._batch_total} asset(s) for rendering…")
        self._start_next_in_queue()

    def _start_next_in_queue(self) -> None:
        if not self._render_queue:
            self._update_action_buttons()
            return

        asset = self._render_queue.pop(0)
        blender = find_blender_executable()
        scene = self.project_root / "blender" / "cameraSetup.blend"
        script = self.project_root / "blender" / "render_head.py"
        output_dir = self.project_root / "renders" / asset.stem
        assert blender is not None

        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "--background",
            str(scene),
            "--python",
            str(script),
            "--",
            str(asset),
            "--output-dir",
            str(output_dir),
            "--views",
            "front",
            "side_r",
        ]

        done = self._batch_total - len(self._render_queue)
        self._pending_asset = asset
        self.select_asset(asset)
        self.status_message.emit(f"Rendering {done}/{self._batch_total}: {asset.name}…")
        self._update_action_buttons()

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.finished.connect(self._on_render_finished)
        self._process.errorOccurred.connect(self._on_render_error)
        self._process.start(str(blender), args)

    def _rebuild_tree(self) -> None:
        self.tree.clear()
        self._folder_items.clear()
        if not self.root_path:
            return

        root_item = QTreeWidgetItem([self.root_path.name])
        root_item.setFlags(root_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.addTopLevelItem(root_item)
        self._folder_items[self.root_path] = root_item

        for asset in self.assets:
            parent = self._folder_item_for(asset.parent)
            rendered = self._has_renders(asset)
            item = QTreeWidgetItem([self._item_label(asset, rendered)])
            item.setData(0, PATH_ROLE, str(asset))
            item.setData(0, RENDERED_ROLE, rendered)
            item.setToolTip(0, str(asset))
            item.setForeground(0, QBrush(QColor("#1b7a3d" if rendered else "#202936")))
            parent.addChild(item)

        self.tree.expandAll()
        self._update_action_buttons()

    def _folder_item_for(self, folder: Path) -> QTreeWidgetItem:
        folder = folder.resolve()
        if folder in self._folder_items:
            return self._folder_items[folder]
        assert self.root_path is not None
        if folder == self.root_path:
            return self._folder_items[self.root_path]

        parent_item = self._folder_item_for(folder.parent)
        item = QTreeWidgetItem([folder.name])
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        parent_item.addChild(item)
        self._folder_items[folder] = item
        return item

    def _iter_asset_items(self):
        def walk(item: QTreeWidgetItem):
            if item.data(0, PATH_ROLE):
                yield item
            for index in range(item.childCount()):
                child = item.child(index)
                if child is not None:
                    yield from walk(child)

        for index in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(index)
            if top is not None:
                yield from walk(top)

    def _on_current_item_changed(self, current: QTreeWidgetItem | None, _previous) -> None:
        asset = None
        if current is not None:
            path = current.data(0, PATH_ROLE)
            if path:
                asset = Path(path)
        self._update_action_buttons()
        self.asset_selected.emit(asset)

    def _on_render_finished(self, exit_code: int, _status) -> None:
        asset = self._pending_asset
        output = ""
        if self._process is not None:
            raw = self._process.readAllStandardOutput()
            output = raw.data().decode("utf-8", errors="replace")
            self._process.deleteLater()
        self._process = None
        self._pending_asset = None

        if exit_code != 0 or asset is None:
            self._batch_failed += 1
            details = output.strip() or f"Exit code {exit_code}"
            # In a batch, keep going; only pop a dialog for single-asset loads.
            if not self._render_queue and self._batch_total <= 1:
                QMessageBox.critical(self, "Render failed", details[-2000:])
            self.status_message.emit(
                f"Render failed for {asset.name if asset else 'asset'} "
                f"({self._batch_failed} failure(s) so far)."
            )
        else:
            if self.root_path and asset.resolve() not in {path.resolve() for path in self.assets}:
                self.assets = collect_assets(self.root_path)
                self._rebuild_tree()
                self.select_asset(asset)
            else:
                self.refresh_render_status()

            self.status_message.emit(f"Rendered {asset.name} → renders/{asset.stem}/")
            self.render_finished.emit(asset)
            self.asset_selected.emit(asset)
            self.assets_changed.emit()

        if self._render_queue:
            self._start_next_in_queue()
            return

        self._update_action_buttons()
        if self._batch_total > 1:
            ok = self._batch_total - self._batch_failed
            self.status_message.emit(
                f"Batch complete: {ok}/{self._batch_total} rendered"
                + (f", {self._batch_failed} failed." if self._batch_failed else ".")
            )
        self._batch_total = 0
        self._batch_failed = 0

    def _on_render_error(self, error) -> None:
        self.status_message.emit(f"Blender process error: {error}")
