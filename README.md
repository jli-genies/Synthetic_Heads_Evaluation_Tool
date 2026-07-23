# Synthetic Heads Evaluation Tool

PyQt6 GUI for reviewing synthetic head assets and tagging them against a shared attribute schema (`tag_schema.json`). Optional Blender renders and GenieSAM segmentation help with visualization and partial auto-fill.

## Requirements

- Python 3.10+ recommended
- [PyQt6](https://pypi.org/project/PyQt6/) (`requirements.txt`)
- [Blender](https://www.blender.org/) 5.1+ on `PATH` (or a standard Windows install) for rendering
- Optional: a working GenieSAM install + SAM3 checkpoint for segmentation-assisted tagging (`segmentation_config.json`)

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
4. **Save** — Use **Submit / Update attributes** to write sidecar JSON under `tags/` (`<asset_filename>.tags.json`).
5. **Review** — Open the `tags/` folder to inspect or share saved tags.

### Optional: segmentation assist

If GenieSAM is configured locally (see `segmentation_config.json`), the tool can run segmentation and propose values for some tags. This requires a working GenieSAM checkout, checkpoint path, and (for HTTP mode) a reachable endpoint.

## Project layout

| Path | Role |
|------|------|
| `ui/` | PyQt main window, asset tree, render + tag panels |
| `blender/` | `cameraSetup.blend` + `render_head.py` multiview stills |
| `segmentation/` | GenieSAM client, heuristics, tag proposals |
| `tools/run_geniesam.py` | CLI wrapper for local GenieSAM → ISAT JSON |
| `tag_schema.json` | Tag categories and field definitions |
| `segmentation_config.json` | GenieSAM / endpoint / checkpoint settings |
| `renders/` | Cached front/side (and optional segmentation) previews |
| `tags/` | Saved per-asset tag JSON |

## Notes

- Renders are optional for tagging but required for preview and for segmentation that reads `front.png`.
- Tag output is keyed by asset filename under `tags/`, independent of where the source mesh lives.
