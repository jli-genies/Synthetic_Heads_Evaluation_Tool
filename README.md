# Synthetic Heads Evaluation Tool

PyQt6 GUI for reviewing synthetic head assets and tagging them against a shared attribute schema (`tag_schema.json`). Optional Blender renders help with visualization, and a trained range-anomaly model can pre-fill per-joint good/bad ratings.

## Requirements

- Python 3.10+ recommended
- [PyQt6](https://pypi.org/project/PyQt6/) (`requirements.txt`)
- [Blender](https://www.blender.org/) 5.1+ on `PATH` (or a standard Windows install) for rendering and for the Evaluate joints button
- Optional: `landmarks/model_range.joblib` trained (`tools/train_range_classifier.py`) for the Evaluate joints button to have something to score with

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

From the repository root:

```bash
python ui\main_window.py
```

Load an asset folder in the GUI to populate the asset tree.

## Workflow

1. **Select assets** — Load a folder of `.glb` / `.fbx` heads in the asset tree.
2. **Render (recommended)** — Use **Load/Render** (single asset) or **Render folder** to generate previews. Output lands in `renders/<asset_stem>/` (`front.png`, `side_r.png`).
3. **Tag** — Click through assets and set attributes in the tag panel (schema-driven fields).
4. **Save** — Use **Submit / Update attributes** to write two sidecars beside the mesh under `<asset_stem>/`: `<stem>_tags.json` (attributes) and `<stem>_joint_eval.json` (joint features). The source `.fbx` / `.glb` is also copied into that folder if missing.
5. **Review** — Open the per-asset folder next to each source mesh to inspect saved tags.

### Optional: model-assisted joint evaluation

The **Evaluate joints** button runs the trained range-anomaly model
(`landmarks/model_range.joblib`, built by `tools/train_range_classifier.py`)
against the selected asset and pre-fills the Joint Features tab's per-marker
good/bad ratings, which you then review and correct before Submit. It shells
out to `tools/predict_asset_range.py`, which needs Blender on `PATH` (or a
standard Windows install) to extract the asset's landmarks. If no model has
been trained yet, the button reports that clearly instead of guessing.

## Project layout

| Path | Role |
|------|------|
| `ui/` | PyQt main window, asset tree, render + tag panels |
| `blender/` | `cameraSetup.blend` + `render_head.py` multiview stills |
| `landmarks/` | Distance/ratio math, feature-row builder, trained models |
| `tools/train_range_classifier.py`, `tools/predict_asset_range.py` | Range-anomaly model training + single-asset scoring |
| `tools/train_classifier.py`, `tools/predict_asset.py` | Gradient-boosted classifier training + single-asset scoring |
| `tag_schema.json` | Tag categories and field definitions |
| `lists/` | `good.json` / `bad.json` asset-name lists from joint eval |
| `renders/` | Cached front/side previews and per-asset model output (`joint_eval_predicted.json`) |
| `<asset_dir>/<stem>/` | Per-asset attribute + joint eval JSON beside the source mesh |

## Notes

- Renders are optional for tagging but required for preview and for segmentation that reads `front.png`.
- Tag output lives beside each source mesh as `<stem>/<stem>_tags.json` (attributes) and `<stem>/<stem>_joint_eval.json` (joint features). On save, the mesh is copied into that folder; the asset tree still lists the original file and hides the packed copy.
