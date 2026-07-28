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

# HeadGen chaos_joints.json bind markers used for per-feature quality ratings.
CHAOS_JOINT_MARKERS: tuple[str, ...] = (
    "FaceBind",
    "JawBind",
    "LeftBrowBind",
    "RightBrowBind",
    "LeftCheekBind",
    "RightCheekBind",
    "LeftCheekBoneBind",
    "RightCheekBoneBind",
    "LeftEyeSocketBind",
    "RightEyeSocketBind",
    "MouthBind",
    "MouthInnerBind",
    "NoseBind",
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
        self.joint_controls: dict[str, QComboBox] = {}
        self.setMinimumWidth(300)

        title = QLabel("Tag Attributes")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Select an asset to begin tagging.")
        self.status_label.setObjectName("tagStatus")
        self.status_label.setWordWrap(True)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_attributes_tab(), "Attributes")
        self.tabs.addTab(self._build_joint_features_tab(), "Joint Features")

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

    def _build_joint_features_tab(self) -> QWidget:
        """Good/bad ratings for each HeadGen chaos joint feature."""
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(10)

        hint = QLabel(
            "Rate each chaos joint feature as good or bad relative to the "
            "authored ground-truth head."
        )
        hint.setObjectName("tagStatus")
        hint.setWordWrap(True)
        content_layout.addWidget(hint)

        group = QGroupBox("Chaos joint markers")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        for marker in CHAOS_JOINT_MARKERS:
            control = QComboBox()
            for label, value in JOINT_FEATURE_OPTIONS:
                control.addItem(label, value)
            control.setMinimumWidth(190)
            self.joint_controls[marker] = control
            form.addRow(marker, control)

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

    def joint_features(self) -> dict[str, str]:
        """Return good/bad ratings keyed by chaos joint marker (omit unset)."""
        values: dict[str, str] = {}
        for marker, control in self.joint_controls.items():
            value = control.currentData() or ""
            if value:
                values[marker] = value
        return values

    def set_joint_features(self, features: dict[str, Any] | None) -> None:
        """Populate joint feature ratings. Missing markers reset to unset."""
        features = features or {}
        for marker, control in self.joint_controls.items():
            value = features.get(marker, "")
            if not isinstance(value, str):
                value = ""
            index = control.findData(value)
            control.setCurrentIndex(max(0, index))

    def clear(self) -> None:
        self.set_tags({})
        self.set_joint_features(None)

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
