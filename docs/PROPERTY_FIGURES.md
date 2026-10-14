# Property figures at manuscript size

The property renderer draws nine properties in a 3 × 3 grid using every row of the recorded final-survivor plotting population. It copies the input population and reference summary into the output and verifies cohort counts, means, medians, sample standard deviations, minima and maxima before saving figures.

## Inputs and dimensions

The completed run's inputs are under `figures/<run_id>/publication/raw_data/`. Ground-truth figures use `gt_final_unique.csv`; generated figures use `generated_final_unique_by_size.csv`. Both use `property_summary.csv` and the enclosing `figure_manifest.json`.

`provenance/manuscript_layout.json` records the manuscript text width, figure placement factors and measurement-source hashes. The measured text width is 430.00462 TeX points, with 72.27 TeX points per inch.

| Population | Manuscript placement | PDF width |
|---|---|---:|
| Ground truth | `SI.tex`, `FIGS/distribution_test.png`, `1\textwidth` | 5.949974 in |
| Generated | `results.tex`, `FIGS/distribution.png`, `0.87\textwidth` | 5.176477 in |

Labels are at least 8 points at the recorded physical widths. PDFs embed TrueType fonts; PNGs use 600 dpi. The 100 dpi previews show the same layout. The reviewed outputs are in `figures/reviewed_20261002/property_gt/` and `figures/reviewed_20261002/property_generated/`; `figures/reviewed_20261002/publication/` contains the combined manuscript figure set.

## Render both populations

Create or activate the recorded environment using the [archive README](../README.md). Set `KINASE_RUNTIME_ENV` to its absolute prefix. Start in the extracted archive root and choose a new directory beside the archive for the renderings:

```bash
set -euo pipefail
export KINASE_ARCHIVE="$PWD"
export KG_PYTHON="${KINASE_RUNTIME_ENV:?Set the recorded environment prefix}/bin/python"
export LD_LIBRARY_PATH="$KINASE_RUNTIME_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
RUN_ID=$("$KG_PYTHON" -c 'import json; r=json.load(open("provenance/run_manifest.json")); assert r["status"] == "complete"; print(r["run_id"])')
PUBLICATION="$KINASE_ARCHIVE/figures/$RUN_ID/publication"
FIGURE_WORK="$KINASE_ARCHIVE/../kinase-property-figures-01"
mkdir "$FIGURE_WORK"
export MPLCONFIGDIR="$FIGURE_WORK/matplotlib-cache"

"$KG_PYTHON" -B "$KINASE_ARCHIVE/scripts/render_property_panels.py" \
  --population "$PUBLICATION/raw_data/gt_final_unique.csv" \
  --reference-summary "$PUBLICATION/raw_data/property_summary.csv" \
  --source-manifest "$PUBLICATION/figure_manifest.json" \
  --layout-spec "$KINASE_ARCHIVE/provenance/manuscript_layout.json" \
  --cohort gt --output "$FIGURE_WORK/property_gt"

"$KG_PYTHON" -B "$KINASE_ARCHIVE/scripts/render_property_panels.py" \
  --population "$PUBLICATION/raw_data/generated_final_unique_by_size.csv" \
  --reference-summary "$PUBLICATION/raw_data/property_summary.csv" \
  --source-manifest "$PUBLICATION/figure_manifest.json" \
  --layout-spec "$KINASE_ARCHIVE/provenance/manuscript_layout.json" \
  --cohort generated --output "$FIGURE_WORK/property_generated"
```

The generated table contains final survivors deduplicated by normalized molecular identity within each linker-size cohort. The renderer uses linker lengths 8–13, labels the six sample counts and saves them in `raw_data/cohort_sizes.csv`. Ground-truth data use the recorded final unique population. Violin distributions and medians use Matplotlib's recorded calculation; bars show means with sample standard deviations (`ddof=1`).

Each output contains the PDF, PNG, preview, copied renderer, raw tables and input manifests. `summary_comparison.json` records agreement for all nine properties; `layout_validation.json` records font sizes and text bounds. `figure_manifest.json` records the renderer hash, software versions, dimensions and output hashes. Measurement-source paths in the layout JSON identify the recorded provenance; rendering reads the layout values and the files passed on the command line.

See [Figure reproduction](FIGURE_REPRODUCTION.md) for the other figures and [LigPlot panel preparation](CONTACT_PANEL_PREPARATION.md) for interaction diagrams at their final manuscript size.
