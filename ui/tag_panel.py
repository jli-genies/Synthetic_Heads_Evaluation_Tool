"""Schema-driven controls for assigning tags to an asset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Face-region categories the range-anomaly model scores against (see
# landmarks/ratios.py's FEATURE_REGIONS -- these ids must stay in sync with
# that mapping). Labels are pulled from tag_schema.json's own categories at
# build time, not hardcoded here, so this tab and the Attributes tab never
# drift on wording for the same region.
FACE_REGIONS: tuple[str, ...] = (
    "eyes_shape",
    "brow_shape",
    "nose_shape",
    "cheeks_shape",
    "mouth_shape",
    "lips_shape",
    "chin_shape",
    "jaw_shape",
    "head_shape",
)

JOINT_FEATURE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Not specified", ""),
    ("Good", "good"),
    ("Bad", "bad"),
)


class CheckableComboBox(QComboBox):
    """Compact multi-select control using checkable combo-box items."""

    values_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setModel(QStandardItemModel(self))

        line_edit = cast(QLineEdit, self.lineEdit())
        line_edit.setReadOnly(True)
        line_edit.setPlaceholderText("Select one or more…")

        view = cast(QListView, self.view())
        view.pressed.connect(self._toggle_item)

    def _item_model(self) -> QStandardItemModel:
        return cast(QStandardItemModel, self.model())

    def add_option(self, label: str, value: str) -> None:
        item = QStandardItem(label)
        item.setData(value, Qt.ItemDataRole.UserRole)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self._item_model().appendRow(item)

    def values(self) -> list[str]:
        model = self._item_model()
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for row in range(model.rowCount())
            if (item := model.item(row)) is not None
            and item.checkState() == Qt.CheckState.Checked
        ]

    def set_values(self, values: list[str]) -> None:
        selected = set(values)
        model = self._item_model()
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue
            state = (
                Qt.CheckState.Checked
                if item.data(Qt.ItemDataRole.UserRole) in selected
                else Qt.CheckState.Unchecked
            )
            item.setCheckState(state)
        self._update_text()

    def hidePopup(self) -> None:  # noqa: N802 - Qt API name
        self._update_text()
        super().hidePopup()

    def _toggle_item(self, index) -> None:
        item = self._item_model().itemFromIndex(index)
        if item is None:
            return
        item.setCheckState(
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        self._update_text()
        self.values_changed.emit()

    def _update_text(self) -> None:
        model = self._item_model()
        labels = [
            item.text()
            for row in range(model.rowCount())
            if (item := model.item(row)) is not None
            and item.checkState() == Qt.CheckState.Checked
        ]
        line_edit = cast(QLineEdit, self.lineEdit())
        line_edit.setText(", ".join(labels))


class TagPanel(QWidget):
    """Builds tag fields from ``tag_schema.json`` and returns their values."""

    submit_requested = pyqtSignal(dict)

    def __init__(self, schema_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.controls: dict[tuple[str, str], QComboBox] = {}
        self.region_controls: dict[str, QComboBox] = {}
        self.region_info_labels: dict[str, QLabel] = {}
        self.overall_control: QComboBox | None = None
        self.setMinimumWidth(300)

        title = QLabel("Tag Attributes")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Select an asset to begin tagging.")
        self.status_label.setObjectName("tagStatus")
        self.status_label.setWordWrap(True)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_attributes_tab(), "Attributes")
        self.tabs.addTab(self._build_face_proportions_tab(), "Face Proportions")

        self.submit_button = QPushButton("Submit / update attributes")
        self.submit_button.setObjectName("primaryButton")
        self.submit_button.setEnabled(False)
        self.submit_button.clicked.connect(lambda: self.submit_requested.emit(self.tags()))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addWidget(title)
        layout.addWidget(self.status_label)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.submit_button)

    def _build_attributes_tab(self) -> QWidget:
        fields_widget = QWidget()
        fields_layout = QVBoxLayout(fields_widget)
        fields_layout.setContentsMargins(4, 4, 4, 4)
        fields_layout.setSpacing(10)
        self._build_fields(fields_layout)
        fields_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(fields_widget)
        return scroll

    def _region_label(self, region_id: str) -> str:
        """Human-readable label for a face region, pulled from tag_schema.json's
        own categories so this tab never drifts from the Attributes tab's wording."""
        for category in self.schema.get("categories", []):
            if category.get("id") == region_id:
                return category.get("label", region_id)
        return region_id

    def _build_face_proportions_tab(self) -> QWidget:
        """Overall good/bad verdict, plus a per-region deviation breakdown.

        The overall verdict is the actual bucket decision (pre-filled from the
        model's single RMS-combined score against its tuned threshold, then
        reviewer-correctable) -- region ratings below are diagnostic only.
        Tallying flagged regions is deliberately NOT how the verdict is
        derived; see ui/sort_assets.py's module docstring for why.
        """
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(10)

        hint = QLabel(
            "Overall verdict defaults to good until Evaluate proportions runs or you "
            "set it yourself. Per-region ratings below explain why, but don't decide it."
        )
        hint.setObjectName("tagStatus")
        hint.setWordWrap(True)
        content_layout.addWidget(hint)

        verdict_group = QGroupBox("Overall verdict")
        verdict_form = QFormLayout(verdict_group)
        verdict_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.overall_control = QComboBox()
        for label, value in JOINT_FEATURE_OPTIONS:
            self.overall_control.addItem(label, value)
        self.overall_control.setCurrentIndex(self.overall_control.findData("good"))
        self.overall_control.setMinimumWidth(190)
        verdict_form.addRow("This asset is", self.overall_control)
        content_layout.addWidget(verdict_group)

        group = QGroupBox("Face regions (diagnostic)")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        for region in FACE_REGIONS:
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)

            info_label = QLabel("Not yet evaluated")
            info_label.setObjectName("tagStatus")
            self.region_info_labels[region] = info_label
            row_layout.addWidget(info_label)

            control = QComboBox()
            for label, value in JOINT_FEATURE_OPTIONS:
                control.addItem(label, value)
            control.setCurrentIndex(control.findData("good"))
            control.setMinimumWidth(190)
            self.region_controls[region] = control
            row_layout.addWidget(control)

            form.addRow(self._region_label(region), row)

        content_layout.addWidget(group)
        content_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def set_asset(self, asset_name: str | None) -> None:
        self.submit_button.setEnabled(bool(asset_name))
        self.status_label.setText(
            f"Editing: {asset_name}" if asset_name else "Select an asset to begin tagging."
        )

    def tags(self) -> dict[str, dict[str, str | list[str]]]:
        values: dict[str, dict[str, str | list[str]]] = {}
        for (category_id, field_id), control in self.controls.items():
            if isinstance(control, CheckableComboBox):
                value: str | list[str] = control.values()
            else:
                value = control.currentData() or ""
            if value:
                values.setdefault(category_id, {})[field_id] = value
        return values

    def set_tags(self, tags: dict[str, Any] | None) -> None:
        tags = tags or {}
        for (category_id, field_id), control in self.controls.items():
            value = tags.get(category_id, {}).get(field_id, [])
            if isinstance(control, CheckableComboBox):
                control.set_values(value if isinstance(value, list) else [value])
            else:
                index = control.findData(value)
                control.setCurrentIndex(max(0, index))

    def default_region_features(self) -> dict[str, str]:
        """All face regions rated ``good``."""
        return {region: "good" for region in FACE_REGIONS}

    def region_features(self) -> dict[str, str]:
        """Return good/bad diagnostic ratings for every face region.

        Unset controls fall back to ``good`` so exports always include a full map.
        """
        values: dict[str, str] = {}
        for region, control in self.region_controls.items():
            value = control.currentData() or ""
            values[region] = value if value in {"good", "bad"} else "good"
        return values

    def set_region_features(self, features: dict[str, Any] | None) -> None:
        """Populate per-region diagnostic ratings. Missing regions default to ``good``."""
        features = features or {}
        for region, control in self.region_controls.items():
            value = features.get(region, "good")
            if not isinstance(value, str) or value not in {"good", "bad"}:
                value = "good"
            index = control.findData(value)
            control.setCurrentIndex(max(0, index))

    def set_region_scores(
        self, scores: dict[str, float] | None, top_contributors: dict[str, str] | None
    ) -> None:
        """Show each region's deviation magnitude and top-contributing measurement.

        Purely informational -- doesn't touch the good/bad controls themselves.
        """
        scores = scores or {}
        top_contributors = top_contributors or {}
        for region, label in self.region_info_labels.items():
            score = scores.get(region)
            if score is None:
                label.setText("Not yet evaluated")
                continue
            contributor = top_contributors.get(region, "n/a")
            label.setText(f"z={score:.2f} · top: {contributor}")

    def default_overall_verdict(self) -> str:
        return "good"

    def overall_verdict(self) -> str:
        """The reviewer's single overall Good/Bad call -- this, not a region
        tally, is what ui/sort_assets.py buckets the asset by."""
        if self.overall_control is None:
            return self.default_overall_verdict()
        value = self.overall_control.currentData() or ""
        return value if value in {"good", "bad"} else "good"

    def set_overall_verdict(self, verdict: str | None) -> None:
        if self.overall_control is None:
            return
        if not isinstance(verdict, str) or verdict not in {"good", "bad"}:
            verdict = "good"
        index = self.overall_control.findData(verdict)
        self.overall_control.setCurrentIndex(max(0, index))

    def clear(self) -> None:
        self.set_tags({})
        self.set_region_features(self.default_region_features())
        self.set_overall_verdict(self.default_overall_verdict())
        self.set_region_scores(None, None)

    def _load_schema(self) -> dict[str, Any]:
        try:
            return json.loads(self.schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to load tag schema at {self.schema_path}: {error}") from error

    def _build_fields(self, parent_layout: QVBoxLayout) -> None:
        for category in self.schema.get("categories", []):
            group = QGroupBox(category["label"])
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

            for field in category.get("fields", []):
                if field.get("type") == "multi_select":
                    control: QComboBox = CheckableComboBox()
                    for option in field.get("options", []):
                        cast(CheckableComboBox, control).add_option(option["label"], option["value"])
                else:
                    control = QComboBox()
                    control.addItem("Not specified", "")
                    for option in field.get("options", []):
                        control.addItem(option["label"], option["value"])

                control.setMinimumWidth(190)
                self.controls[(category["id"], field["id"])] = control
                form.addRow(field["label"], control)

            parent_layout.addWidget(group)
