#!/usr/bin/env python3
"""Render the recorded final-survivor populations at manuscript print size."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

PROPERTIES = (
    ("rotatable_bond_count", "Rotatable bonds", True),
    ("ring_count", "Rings", True),
    ("fused_ring_count", "Fused ring systems", True),
    ("heteroaromatic_ring_count", "Heteroaromatic rings", True),
    ("aromatic_ring_count", "Aromatic rings", True),
    ("logp", "logP", False),
    ("qed", "QED", False),
    ("sa_score", "SA score", False),
    ("molecular_weight", "Molecular weight (Da)", False),
)
COLORS = ("#7c98c7", "#80b6a2", "#b5ad70", "#cc9c71", "#b98caa", "#a99fca")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def copy_exact(source, destination):
    with Path(source).open("rb") as src, Path(destination).open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha256(source) != sha256(destination):
        raise ValueError(f"Copy checksum differs: {source}")


def check_population(table, cohort):
    id_column = "gt_id" if cohort == "gt" else "candidate_id"
    required = {id_column, "canonical_smiles", "after_2d", *[p[0] for p in PROPERTIES]}
    if cohort == "generated":
        required.update({"normalized_identity", "linker_size"})
    if required.difference(table.columns):
        raise ValueError(f"Missing columns: {sorted(required.difference(table.columns))}")
    if table[id_column].duplicated().any():
        raise ValueError("Repeated record identifiers")
    if not table.after_2d.astype(str).str.lower().isin(["1", "1.0", "true"]).all():
        raise ValueError("Population includes a row outside the final survivors")
    identity = "normalized_identity" if "normalized_identity" in table else "canonical_smiles"
    if table[identity].astype(str).eq("").any():
        raise ValueError("Empty molecular identity")
    keys = ["linker_size", identity] if cohort == "generated" else [identity]
    if table.duplicated(keys).any():
        raise ValueError("Input is not the recorded deduplicated plotting population")
    if cohort == "generated":
        sizes = pd.to_numeric(table.linker_size, errors="raise")
        if not sizes.isin(range(8, 14)).all():
            raise ValueError("Generated linker size outside the recorded 8--13 panel")
    return identity


def summarize(table, cohort):
    groups = [(str(s), table.loc[table.linker_size.astype(int) == s]) for s in range(8, 14)] if cohort == "generated" else [("GT", table)]
    summaries = []
    for column, _, _ in PROPERTIES:
        for label, group in groups:
            values = pd.to_numeric(group[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values: {column}/{label}")
            record = {"cohort": label, "property": column, "n": len(values)}
            if len(values):
                record.update(mean=float(values.mean()), median=float(np.median(values)),
                              sd=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                              minimum=float(values.min()), maximum=float(values.max()))
            summaries.append(record)
    return groups, summaries


def compare_summary(summaries, reference, cohort):
    reference = reference.loc[reference.cohort.astype(str).eq("GT")] if cohort == "gt" else reference.loc[reference.cohort.astype(str).isin([str(s) for s in range(8, 14)])]
    expected = {(str(r["cohort"]), r["property"]): r for r in reference.to_dict("records")}
    if len(expected) != len(reference) or set(expected) != {(r["cohort"], r["property"]) for r in summaries}:
        raise ValueError("Reference summary property/cohort keys differ")
    comparisons = []
    for row in summaries:
        other = expected[(row["cohort"], row["property"])]
        if int(row["n"]) != int(other["n"]):
            raise ValueError("Reference cohort count differs")
        errors = {}
        for field in ("mean", "median", "sd", "minimum", "maximum"):
            if row["n"]:
                actual, previous = float(row[field]), float(other[field])
                if not math.isclose(actual, previous, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"Statistic differs: {row['cohort']}/{row['property']}/{field}")
                errors[field] = abs(actual - previous)
        comparisons.append({"cohort": row["cohort"], "property": row["property"], "n": row["n"], "absolute_differences": errors})
    return comparisons


def verify_source_manifest(population, reference, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    by_sha = {entry["sha256"] for entry in manifest.get("artifacts", [])}
    for path in (population, reference):
        if sha256(path) not in by_sha:
            raise ValueError(f"Input is not bound by the supplied source figure manifest: {path}")
    if manifest.get("status") not in (None, "complete"):
        raise ValueError("Source figure manifest does not describe completed figures")
    return manifest


def validate_layout(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems, boxes, minimum_font = [], [], 100.0
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        bounds = item.get_window_extent(renderer)
        # Invisible empty-axis offset strings are excluded by the text test.
        if bounds.x0 < -0.5 or bounds.y0 < -0.5 or bounds.x1 > fig.bbox.width + 0.5 or bounds.y1 > fig.bbox.height + 0.5:
            problems.append({"text": item.get_text(), "reason": "outside_canvas"})
        minimum_font = min(minimum_font, item.get_fontsize())
        boxes.append({"text": item.get_text(), "font_points": item.get_fontsize(), "bounds_pixels": list(bounds.bounds)})
    for axis in axes.flat:
        ticks = [t for t in axis.get_xticklabels() if t.get_visible() and t.get_text()]
        for left, right in zip(ticks, ticks[1:]):
            if left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)):
                problems.append({"text": left.get_text() + "/" + right.get_text(), "reason": "overlapping_x_ticks"})
    if minimum_font < 8:
        problems.append({"reason": "font_below_8_points", "minimum": minimum_font})
    if problems:
        raise ValueError(f"Print layout validation failed: {problems}")
    return {"minimum_font_points": minimum_font, "text_boxes": boxes, "problems": problems}


def render(groups, cohort, width_inches):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6})
    height = 6.5 if cohort == "generated" else 6.3
    fig, axes = plt.subplots(3, 3, figsize=(width_inches, height))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.16 if cohort == "generated" else 0.075,
                        top=0.915, wspace=0.53, hspace=0.60)
    for axis, (column, title, discrete) in zip(axes.flat, PROPERTIES):
        axis.set_title(title, fontsize=8.25, pad=6)
        for index, (_, group) in enumerate(groups):
            values = pd.to_numeric(group[column], errors="raise").to_numpy(dtype=float)
            color = COLORS[index] if cohort == "generated" else "#bf88a8"
            if len(values):
                if discrete:
                    axis.bar(index, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0,
                             color=color, capsize=2, width=0.7,
                             error_kw={"elinewidth": 0.8, "capthick": 0.8})
                elif len(values) > 1 and np.ptp(values) > 0:
                    parts = axis.violinplot(values, positions=[index], widths=0.85, showmedians=True)
                    for body in parts["bodies"]:
                        body.set_facecolor(color); body.set_alpha(0.8)
                    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                        parts[key].set_color("#333333"); parts[key].set_linewidth(0.8)
                else:
                    axis.scatter([index], [values[0]], color=color, s=16)
        axis.set_xticks(range(len(groups)) if cohort == "generated" else [])
        if cohort == "generated":
            axis.set_xticklabels([label for label, _ in groups])
        axis.set_xlim(-0.65, len(groups) - 0.35)
        axis.tick_params(axis="both", length=2.5, pad=2)
        axis.spines[["top", "right"]].set_visible(False)
        if discrete:
            axis.set_ylabel("Mean ± SD", labelpad=3)
    if cohort == "gt":
        fig.suptitle(f"Ground-truth final survivors (n = {len(groups[0][1])})", fontsize=9, y=0.976)
        fig.text(0.5, 0.025, "Bars: mean ± sample SD.  Violins: distributions and medians.", ha="center", fontsize=8)
    else:
        fig.suptitle("Generated final survivors by linker length", fontsize=9, y=0.976)
        fig.text(0.5, 0.102, "Linker heavy atoms and cohort size (n)", ha="center", fontsize=8)
        for index, (label, group) in enumerate(groups):
            x = 0.13 + index * 0.15
            fig.text(x, 0.070, label, ha="center", fontsize=8)
            fig.text(x, 0.045, f"n = {len(group)}", ha="center", fontsize=8)
        fig.text(0.5, 0.014, "Bars: mean ± sample SD.  Violins: distributions and medians.", ha="center", fontsize=8)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True, help="Recorded final unique population CSV; copied byte-exact")
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--layout-spec", type=Path, required=True)
    parser.add_argument("--cohort", choices=["gt", "generated"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, absent output directory")
    args = parser.parse_args()
    for path in (args.population, args.reference_summary, args.source_manifest, args.layout_spec):
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
    source = verify_source_manifest(args.population, args.reference_summary, args.source_manifest)
    layout = json.loads(args.layout_spec.read_text())
    textwidth = float(layout["textwidth_tex_points"])
    multiplier = float(layout["width_multipliers"][args.cohort])
    width = textwidth / 72.27 * multiplier
    table = pd.read_csv(args.population, keep_default_na=False)
    identity = check_population(table, args.cohort)
    groups, summaries = summarize(table, args.cohort)
    comparison = compare_summary(summaries, pd.read_csv(args.reference_summary), args.cohort)
    fig, axes = render(groups, args.cohort, width)
    layout_check = validate_layout(fig, axes)
    args.output.mkdir(parents=True, exist_ok=False)
    raw, figures = args.output / "raw_data", args.output / "FIGS"
    raw.mkdir(); figures.mkdir()
    copies = [(args.population, raw / args.population.name),
              (args.reference_summary, raw / args.reference_summary.name),
              (args.source_manifest, raw / "source_figure_manifest.json"),
              (args.layout_spec, args.output / "manuscript_layout.json"),
              (Path(__file__), args.output / Path(__file__).name)]
    for source_path, destination in copies:
        copy_exact(source_path, destination)
    with (raw / "rendered_property_summary.csv").open("x", newline="") as handle:
        pd.DataFrame(summaries).to_csv(handle, index=False)
    with (raw / "cohort_sizes.csv").open("x", newline="") as handle:
        pd.DataFrame([{"cohort": label, "n": len(group)} for label, group in groups]).to_csv(handle, index=False)
    write_json(args.output / "summary_comparison.json", {"status": "pass", "tolerance": {"relative": 1e-12, "absolute": 1e-12}, "comparisons": comparison})
    write_json(args.output / "layout_validation.json", layout_check)
    basename = "distribution_test" if args.cohort == "gt" else "distribution"
    for extension, dpi in [("pdf", 600), ("png", 600)]:
        with (figures / f"{basename}.{extension}").open("xb") as handle:
            fig.savefig(handle, format=extension, dpi=dpi, bbox_inches=None, facecolor="white")
    with (figures / f"{basename}_print_preview.png").open("xb") as handle:
        fig.savefig(handle, format="png", dpi=100, bbox_inches=None, facecolor="white")
    plt.close(fig)
    input_records = [{"role": role, "path": str(path.resolve()), "sha256": sha256(path)} for role, path in
                     [("population", args.population), ("reference_summary", args.reference_summary),
                      ("source_manifest", args.source_manifest), ("layout_spec", args.layout_spec)]]
    result = {"schema_version": 1, "status": "complete", "cohort": args.cohort,
              "source_run_id": source.get("run_id"), "population_records": len(table),
              "population_identity_column": identity, "selection": "Use every row of the recorded final unique plotting population; no selection changes",
              "figure_width_inches": width, "figure_width_pdf_points": width * 72,
              "manuscript_width_multiplier": multiplier, "minimum_font_points": layout_check["minimum_font_points"],
              "script_sha256": sha256(__file__), "inputs": input_records,
              "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
              "arguments": sys.argv[1:],
              "artifacts": [{"path": str(p.relative_to(args.output)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                            for p in sorted(args.output.rglob("*")) if p.is_file()]}
    write_json(args.output / "figure_manifest.json", result)
    print(json.dumps({"status": "complete", "output": str(args.output), "records": len(table), "width_inches": width}))


if __name__ == "__main__":
    main()
