"""Main window for the Synthetic Heads asset tagging tool."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QThread, Qt, pyqtSignal
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

# Association layer lives at project root (sibling of ui/).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from segmentation.geniesam_client import (  # noqa: E402
    build_local_payload,
    expected_local_output,
    invoke_geniesam,
)
from segmentation.propose_tags import merge_tags, propose_tags_from_isat  # noqa: E402


class GenieSamHttpWorker(QThread):
    """Background POST to local GenieSAM Docker (use_local)."""

    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        endpoint_url: str,
        payload: dict,
        timeout_s: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._endpoint_url = endpoint_url
        self._payload = payload
        self._timeout_s = timeout_s

    def run(self) -> None:
        try:
            result = invoke_geniesam(
                self._endpoint_url,
                self._payload,
                timeout_s=self._timeout_s,
            )
            if result.get("status") != "success":
                message = result.get("message") or json.dumps(result)[:2000]
                self.failed.emit(str(message))
                return
            self.succeeded.emit(result)
        except Exception as error:  # noqa: BLE001 - surface any transport/API failure
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    """Coordinates asset browsing, render previews, and sidecar tag files."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Synthetic Heads — Asset Tagger")
        self.resize(1440, 860)
        self.setMinimumSize(1050, 650)

        self.project_root = Path(__file__).resolve().parents[1]
        self.current_asset: Path | None = None
        self._seg_process: QProcess | None = None
        self._seg_worker: GenieSamHttpWorker | None = None
        self._seg_asset: Path | None = None
        self._seg_output_json: Path | None = None

        self.asset_tree = AssetTree(project_root=self.project_root)
        self.asset_tree.setMinimumWidth(240)
        self.asset_tree.asset_selected.connect(self._select_asset)
        self.asset_tree.status_message.connect(self.status_bar_message)
        self.asset_tree.render_finished.connect(self._on_render_finished)

        self.render_panel = RenderPanel(project_root=self.project_root)
        self.render_panel.setMinimumWidth(520)
        self.render_panel.previous_requested.connect(lambda: self._navigate(-1))
        self.render_panel.next_requested.connect(lambda: self._navigate(1))
        self.render_panel.segment_requested.connect(self.run_segmentation)

        self.tag_panel = TagPanel(self.project_root / "tag_schema.json")
        self.tag_panel.setMinimumWidth(320)
        self.tag_panel.submit_requested.connect(self.save_tags)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        self.splitter.setOpaqueResize(True)
        self.splitter.addWidget(self.asset_tree)
        self.splitter.addWidget(self.render_panel)
        self.splitter.addWidget(self.tag_panel)
        self.splitter.setSizes([280, 720, 400])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, False)
        self.setCentralWidget(self.splitter)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Choose a folder containing .glb / .fbx assets.")
        self.setStyleSheet(STYLE_SHEET)

    def status_bar_message(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def _on_render_finished(self, asset_path: Path) -> None:
        if self.current_asset and self.current_asset.resolve() == Path(asset_path).resolve():
            self.render_panel.set_asset(self.current_asset)

    def _select_asset(self, asset_path: Path | None) -> None:
        self.current_asset = Path(asset_path) if asset_path else None
        # #region agent log
        try:
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
            index = -1
            if self.current_asset:
                resolved = self.current_asset.resolve()
                for i, path in enumerate(assets):
                    if path.resolve() == resolved:
                        index = i
                        break
        self.render_panel.set_navigation_enabled(index > 0, 0 <= index < len(assets) - 1)
        if self._seg_process is not None or self._seg_worker is not None:
            self.render_panel.set_segment_enabled(False)

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
            tag_path.parent.mkdir(parents=True, exist_ok=True)
            tag_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            QMessageBox.critical(self, "Unable to save tags", str(error))
            return

        relative = tag_path.relative_to(self.project_root)
        self.tag_panel.status_label.setText(f"Saved: {relative}")
        self.status_bar.showMessage(f"Tags saved to {relative}", 5000)

    def run_segmentation(self) -> None:
        if not self.current_asset:
            return
        if self._seg_process is not None or self._seg_worker is not None:
            self.status_bar.showMessage("Segmentation already running…", 4000)
            return

        front_png = self.project_root / "renders" / self.current_asset.stem / "front.png"
        if not front_png.is_file():
            QMessageBox.warning(
                self,
                "Render required",
                f"No front render found at:\n{front_png}\n\nRender the asset first.",
            )
            return

        try:
            config = self._load_segmentation_config()
        except (OSError, json.JSONDecodeError, ValueError) as error:
            QMessageBox.critical(self, "Segmentation config error", str(error))
            return

        mode = str(config.get("mode") or "http_local").strip().lower()
        if mode == "http_local":
            self._run_segmentation_http(front_png, config)
        elif mode == "process":
            self._run_segmentation_process(front_png, config)
        else:
            QMessageBox.critical(
                self,
                "Unknown segmentation mode",
                f"Unsupported mode {mode!r}. Use \"http_local\" or \"process\".",
            )

    def _run_segmentation_http(self, front_png: Path, config: dict) -> None:
        assert self.current_asset is not None
        endpoint = str(config.get("endpoint_url") or "http://127.0.0.1:8080/invocations")
        host_root_raw = (config.get("host_renders_root") or "").strip()
        host_root = (
            Path(host_root_raw).expanduser()
            if host_root_raw
            else (self.project_root / "renders")
        )
        container_root = str(config.get("container_renders_root") or "/data/renders")
        timeout_s = float(config.get("timeout_s") or 600)

        output_dir = self.project_root / "renders" / self.current_asset.stem / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)

        categories = config.get("categories")
        try:
            payload = build_local_payload(
                image_host=front_png,
                output_dir_host=output_dir,
                host_renders_root=host_root,
                container_renders_root=container_root,
                categories=categories if isinstance(categories, list) else None,
                request_id=f"{self.current_asset.stem}-front",
            )
        except ValueError as error:
            QMessageBox.critical(self, "Path mapping error", str(error))
            return

        self._seg_output_json = expected_local_output(output_dir)
        self._seg_asset = self.current_asset
        self.render_panel.set_segment_enabled(False)
        self.status_bar.showMessage(
            f"Calling GenieSAM ({endpoint}) for {self.current_asset.name}…"
        )

        worker = GenieSamHttpWorker(endpoint, payload, timeout_s, parent=self)
        worker.succeeded.connect(self._on_segmentation_http_success)
        worker.failed.connect(self._on_segmentation_http_failure)
        worker.finished.connect(self._on_segmentation_http_finished)
        self._seg_worker = worker
        worker.start()

    def _on_segmentation_http_success(self, _result: dict) -> None:
        self._apply_segmentation_result()

    def _on_segmentation_http_failure(self, message: str) -> None:
        QMessageBox.critical(self, "Segmentation failed", message[-2000:])
        self.status_bar.showMessage("Segmentation failed", 7000)

    def _on_segmentation_http_finished(self) -> None:
        self._seg_worker = None
        self.render_panel.set_segment_enabled(True)

    def _run_segmentation_process(self, front_png: Path, config: dict) -> None:
        assert self.current_asset is not None
        try:
            python_exe = self._resolve_python(config)
        except ValueError as error:
            QMessageBox.critical(self, "Python path error", str(error))
            return

        checkpoint = self._resolve_checkpoint(config)
        geniesam_root = Path(config.get("geniesam_root") or "").expanduser()
        if not geniesam_root.is_dir():
            QMessageBox.critical(
                self,
                "GenieSAM not found",
                f"geniesam_root is missing or invalid:\n{geniesam_root}\n\n"
                "Update segmentation_config.json.",
            )
            return
        if not checkpoint:
            QMessageBox.critical(
                self,
                "Checkpoint required",
                "Set checkpoint in segmentation_config.json or SAM3_CHECKPOINT env var "
                "to a local sam3.pth file.",
            )
            return

        output_dir = self.project_root / "renders" / self.current_asset.stem / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)
        script = self.project_root / "tools" / "run_geniesam.py"
        self._seg_output_json = output_dir / "front.json"
        self._seg_asset = self.current_asset

        args = [
            str(script),
            "--image",
            str(front_png),
            "--output-dir",
            str(output_dir),
            "--geniesam-root",
            str(geniesam_root),
            "--checkpoint",
            str(checkpoint),
            "--device",
            str(config.get("device") or "cuda"),
            "--image-size",
            str(int(config.get("image_size") or 1008)),
            "--basename",
            "front",
        ]
        categories = config.get("categories")
        if isinstance(categories, list) and categories:
            args.append("--categories")
            args.extend(str(c) for c in categories)

        self.render_panel.set_segment_enabled(False)
        self.status_bar.showMessage(f"Running GenieSAM on {self.current_asset.name}…")

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(self._on_segmentation_finished)
        process.errorOccurred.connect(self._on_segmentation_error)
        self._seg_process = process
        process.start(str(python_exe), args)

    def _on_segmentation_finished(self, exit_code: int, _status) -> None:
        process = self._seg_process
        self._seg_process = None
        self.render_panel.set_segment_enabled(True)

        log_text = ""
        if process is not None:
            try:
                raw = process.readAllStandardOutput()
                log_text = raw.data().decode("utf-8", errors="replace")
            except Exception:
                log_text = ""

        if exit_code != 0 or not self._seg_output_json or not self._seg_output_json.is_file():
            detail = log_text.strip() or f"exit code {exit_code}"
            QMessageBox.critical(
                self,
                "Segmentation failed",
                f"GenieSAM did not produce output.\n\n{detail[-2000:]}",
            )
            self.status_bar.showMessage("Segmentation failed", 7000)
            return

        self._apply_segmentation_result()

    def _apply_segmentation_result(self) -> None:
        asset = self._seg_asset
        output_json = self._seg_output_json
        if not output_json or not output_json.is_file():
            QMessageBox.critical(
                self,
                "Segmentation failed",
                f"Expected ISAT JSON missing:\n{output_json}",
            )
            self.status_bar.showMessage("Segmentation failed", 7000)
            return

        try:
            proposed = propose_tags_from_isat(output_json)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "Tag proposal failed", str(error))
            self.status_bar.showMessage("Segmentation OK, tag proposal failed", 7000)
            return

        existing = self.tag_panel.tags()
        if not (asset and self.current_asset and asset.resolve() == self.current_asset.resolve()):
            if asset:
                existing = self._load_tags(asset)

        merged, filled = merge_tags(existing, proposed)
        if asset and self.current_asset and asset.resolve() == self.current_asset.resolve():
            self.tag_panel.set_tags(merged)
            self.tag_panel.status_label.setText(
                f"AI suggested {filled} field(s) — review and Submit"
            )
        self.status_bar.showMessage(
            f"Suggested {filled} field(s) from segmentation — review and Submit",
            8000,
        )

    def _on_segmentation_error(self, error) -> None:
        self._seg_process = None
        self.render_panel.set_segment_enabled(True)
        QMessageBox.critical(
            self,
            "Segmentation process error",
            f"Failed to start GenieSAM process: {error}",
        )
        self.status_bar.showMessage("Segmentation process error", 7000)

    def _load_segmentation_config(self) -> dict:
        path = self.project_root / "segmentation_config.json"
        if not path.is_file():
            raise ValueError(f"Missing {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve_python(self, config: dict) -> Path:
        configured = (config.get("python_exe") or "").strip()
        if configured:
            path = Path(configured).expanduser()
            if path.is_file():
                return path
            raise ValueError(f"python_exe not found: {path}")
        return Path(sys.executable)

    def _resolve_checkpoint(self, config: dict) -> Path | None:
        configured = (config.get("checkpoint") or "").strip()
        env_ckpt = (os.environ.get("SAM3_CHECKPOINT") or "").strip()
        raw = configured or env_ckpt
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path if path.is_file() else None

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

    def _tag_path(self, asset_path: Path) -> Path:
        """Centralized cache: ``tags/<asset_filename>.tags.json``."""
        return self.project_root / "tags" / f"{asset_path.name}.tags.json"

    def _load_tags(self, asset_path: Path) -> dict:
        tag_path = self._tag_path(asset_path)
        legacy_path = asset_path.with_name(f"{asset_path.name}.tags.json")
        if not tag_path.exists() and legacy_path.is_file():
            tag_path = legacy_path
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
    width: 6px;
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
