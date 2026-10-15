#!/usr/bin/env python3
"""Render figures and their exact plotting tables from an evaluated run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

PROPERTIES = (
    ("rotatable_bond_count", "Rotatable bonds", True),
    ("ring_count", "Rings", True),
    ("fused_ring_count", "Fused ring systems", True),
    ("heteroaromatic_ring_count", "Heteroaromatic rings", True),
    ("aromatic_ring_count", "Aromatic rings", True),
    ("logp", "logP", False), ("qed", "QED", False),
    ("sa_score", "SA score", False), ("molecular_weight", "Molecular weight (Da)", False),
)
COLORS = ("#7c98c7", "#80b6a2", "#b5ad70", "#cc9c71", "#b98caa", "#a99fca")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def copy_file(source, target):
    with Path(source).open("rb") as incoming, Path(target).open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def write_csv(table, path):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        table.to_csv(handle, index=False)


def flag(series):
    values = series.astype(str).str.lower()
    if not values.isin(["0", "1", "true", "false", "0.0", "1.0"]).all():
        raise ValueError(f"Non-binary flag values in {series.name}")
    return values.isin(["1", "true", "1.0"])


def normalized_population(table):
    """Retain source SMILES and add a parse-normalized isomeric identity key."""
    from kinase_gap.fragmentation import canonical_isomeric_identity
    result = table.copy()
    identities = []
    for value in result.canonical_smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError("Final survivor has an unparsable canonical SMILES")
        identities.append(canonical_isomeric_identity(molecule))
    result["normalized_identity"] = identities
    result["plot_identity_normalization_status"] = "parsed_and_canonicalized"
    return result


def read_evaluations(path, id_column):
    table = pd.read_csv(path, keep_default_na=False)
    required = [id_column, "canonical_smiles", "after_posebusters", "after_lipinski", "after_2d"]
    for column in required:
        if column not in table:
            raise ValueError(f"Missing {column}: {path}")
    if table[id_column].duplicated().any():
        raise ValueError(f"Repeated {id_column}: {path}")
    previous = pd.Series(True, index=table.index)
    for name in ("after_posebusters", "after_lipinski", "after_2d"):
        current = flag(table[name])
        if (current & ~previous).any():
            raise ValueError(f"Non-sequential survivor flags: {path}: {name}")
        previous = current
    if "final_pass" in table and not (flag(table.final_pass) == previous).all():
        raise ValueError(f"final_pass differs from after_2d: {path}")
    return table


def save_figure(figure, directory, basename):
    for extension in ("png", "pdf"):
        with (directory / f"{basename}.{extension}").open("xb") as handle:
            figure.savefig(handle, format=extension, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def image_figure(image_path, directory, basename):
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        with (directory / f"{basename}.png").open("xb") as handle:
            rgb.save(handle, format="PNG")
        with (directory / f"{basename}.pdf").open("xb") as handle:
            rgb.save(handle, format="PDF", resolution=300)


def property_plots(table, directory, basename, group_by_size):
    groups = [(str(size), table.loc[table.linker_size.astype(int) == size]) for size in range(8, 14)] if group_by_size else [("GT", table)]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    summaries = []
    for axis, (column, title, discrete) in zip(axes.flat, PROPERTIES):
        axis.set_title(title, fontsize=11)
        for i, (label, cohort) in enumerate(groups):
            values = pd.to_numeric(cohort[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {column} in {label}")
            color = COLORS[i % len(COLORS)] if group_by_size else "#bf88a8"
            if len(values):
                if discrete:
                    axis.bar(i, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0,
                             color=color, capsize=3, width=0.7)
                elif len(values) > 1 and np.ptp(values) > 0:
                    parts = axis.violinplot(values, positions=[i], widths=0.85, showmedians=True)
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.8)
                    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                        parts[key].set_color("#333333")
                else:
                    axis.scatter([i], [values[0]], color=color, s=35)
                summaries.append({"cohort": label, "property": column, "n": len(values),
                                  "mean": float(values.mean()), "median": float(np.median(values)),
                                  "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                                  "minimum": float(values.min()), "maximum": float(values.max())})
            else:
                summaries.append({"cohort": label, "property": column, "n": 0})
        axis.set_xticks(range(len(groups)))
        axis.set_xticklabels([f"{label}\nn={len(cohort)}" for label, cohort in groups], fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        if discrete:
            axis.set_ylabel("Mean ± sample SD")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    save_figure(fig, directory, basename)
    return summaries


def funnel_table(generated, gt):
    rows = []
    for label, cohort in [("GT", gt)] + [(str(size), generated.loc[generated.linker_size.astype(int) == size]) for size in range(8, 14)]:
        previous = len(cohort)
        for stage, column in [("Input", None), ("PoseBusters", "after_posebusters"), ("Lipinski", "after_lipinski"), ("MOSES", "after_2d")]:
            count = len(cohort) if column is None else int(flag(cohort[column]).sum())
            rows.append({"cohort": label, "stage": stage, "count": count, "attempts": len(cohort),
                         "cumulative_rate": count / len(cohort) if len(cohort) else None,
                         "conditional_rate": count / previous if previous else None})
            previous = count
    return pd.DataFrame(rows)


def draw_funnel(table, directory):
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    stages = ["Input", "PoseBusters", "Lipinski", "MOSES"]
    for i, label in enumerate(["GT"] + [str(size) for size in range(8, 14)]):
        subset = table.loc[table.cohort == label].set_index("stage").reindex(stages)
        axis.plot(stages, 100 * subset.cumulative_rate, "o-", linewidth=1.8,
                  color="#333333" if label == "GT" else COLORS[i - 1], label=label)
    axis.set_ylabel("Cumulative survival (%)")
    axis.set_ylim(0, 103)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(title="Linker atoms", frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, directory, "filter_survival")


def draw_workflow(directory):
    fig, axis = plt.subplots(figsize=(13, 5.5))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 5.5)
    axis.axis("off")
    boxes = [(0.2, 3.2, "1  KLIFS structures", "Ligands and 10 Å protein pockets\nGroup-disjoint train, validation and test sets"),
             (4.5, 3.2, "2  Fragmentation", "Two retained fragments and a linker\nAnchor atoms follow cut-bond endpoints"),
             (8.8, 3.2, "3  Model fine-tuning", "MOAD checkpoint initialization\nValidation-based hyperparameter selection"),
             (8.8, 0.3, "4  Linker generation", "Retained fragments and pocket context\nLinker lengths 8–13 heavy atoms"),
             (4.5, 0.3, "5  Molecular evaluation", "PoseBusters → Lipinski → MOSES\nChemical properties and diversity"),
             (0.2, 0.3, "6  Structural analysis", "Pose overlays and atom perturbations\nProtein–ligand contact diagrams")]
    for x, y, title, body in boxes:
        axis.add_patch(FancyBboxPatch((x, y), 4.0, 1.65, boxstyle="round,pad=0.06", facecolor="#eaf0f7", edgecolor="#506784"))
        axis.text(x + 2, y + 1.23, title, ha="center", va="center", fontsize=12, weight="bold")
        axis.text(x + 2, y + 0.62, body, ha="center", va="center", fontsize=9.3, linespacing=1.6)
    for start, end in [((4.24, 4), (4.43, 4)), ((8.54, 4), (8.73, 4)), ((10.8, 3.1), (10.8, 2.05)), ((8.74, 1.1), (8.56, 1.1)), ((4.44, 1.1), (4.26, 1.1))]:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20, color="#506784"))
    save_figure(fig, directory, "workflow")


def draw_mmpa(dataset, directory, raw):
    path = dataset / "KLIFS_test_table.csv"
    table = pd.read_csv(path, keep_default_na=False)
    cohort = table.loc[table.pdb.str.lower() == "5lqf"].sort_values(["structure_id", "fragmentation_id"])
    if cohort.empty:
        raise ValueError("No CDK1/5LQF test context for the fragmentation figure")
    row = cohort.iloc[0]
    molecules = [Chem.MolFromSmiles(row.canonical_smiles), Chem.MolFromSmiles(row.mmpa_fragments_with_dummies), Chem.MolFromSmiles(row.mmpa_linker_with_dummies)]
    if any(molecule is None for molecule in molecules):
        raise ValueError("CDK1 fragmentation SMILES could not be parsed")
    legends = ["Native CDK1 ligand (5LQF)", "Two retained fragments", "Removed linker"]
    for molecule in molecules:
        rdDepictor.Compute2DCoords(molecule)
    image = Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(700, 420), legends=legends)
    with (directory / "mmpa.png").open("xb") as handle:
        image.save(handle, "PNG")
    with (directory / "mmpa.pdf").open("xb") as handle:
        image.convert("RGB").save(handle, "PDF", resolution=300)
    write_json(raw / "mmpa_5lqf.json", {"selection_rule": "First test row sorted by structure_id and fragmentation_id with PDB 5LQF",
                                      "source_table_sha256": sha256(path), "context": row.to_dict(),
                                      "dummy_atom_meaning": "Cut-bond endpoint"})


def structural_figures(structural, directory, raw):
    manifest_path = structural / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    copy_file(manifest_path, raw / "structural_manifest.json")
    selection_records = []
    for target in ("5lqf", "4ag8"):
        record = manifest["targets"][target]
        for filename, basename in [("morgan_similarity_grid.png", "atomic_molgrid"), ("qed_atom_deletion_grid.png", "qed_molgrid")]:
            if not record["candidates"]:
                continue
            image_figure(structural / "targets" / target / filename, directory,
                         basename + ("_type2" if target == "4ag8" else ""))
        for rank, candidate in enumerate(record["candidates"], start=1):
            selection_records.append({"target": target, "panel": rank, "candidate_id": candidate["candidate_id"],
                                      "source_structure_id": candidate["source_structure_id"],
                                      "fragmentation_id": candidate["fragmentation_id"], "directory": candidate["directory"]})
            destination = raw / target / f"candidate_{rank:02d}"
            destination.mkdir(parents=True)
            for name in ("candidate_aligned.sdf", "fragment_mask.json", "morgan_similarity_weights.json", "qed_atom_deletion_weights.json", "alignment.json"):
                copy_file(structural / candidate["directory"] / name, destination / name)
    write_csv(pd.DataFrame(selection_records), raw / "structural_selection.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--structural-dir", type=Path)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.run_manifest.read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Run manifest must declare status complete after evaluation validation")
    for path in (args.evaluation_dir, args.dataset_dir, args.generation_dir, args.structural_dir, args.run_manifest):
        if path is not None and "kinase-abr" in str(path.resolve()):
            raise ValueError("Excluded source path")
    generated_path = args.evaluation_dir / "evaluations.csv"
    gt_path = args.evaluation_dir / "gt_evaluations.csv"
    generated = read_evaluations(generated_path, "candidate_id")
    gt = read_evaluations(gt_path, "gt_id")
    if len(generated) != metadata["generation"]["attempts"]:
        raise ValueError("Evaluation count differs from complete run manifest")
    generated_population = normalized_population(generated.loc[flag(generated.after_2d)]).sort_values("candidate_id").drop_duplicates(["linker_size", "normalized_identity"])
    gt_population = normalized_population(gt.loc[flag(gt.after_2d)]).sort_values("gt_id").drop_duplicates("normalized_identity")
    for population in (generated_population, gt_population):
        if population.canonical_smiles.eq("").any():
            raise ValueError("Empty SMILES in final-pass plotting population")
    args.output.mkdir(parents=True, exist_ok=False)
    figures, raw = args.output / "FIGS", args.output / "raw_data"
    figures.mkdir(); raw.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    write_csv(generated_population, raw / "generated_final_unique_by_size.csv")
    write_csv(gt_population, raw / "gt_final_unique.csv")
    summaries = property_plots(generated_population, figures, "distribution", True)
    summaries += property_plots(gt_population, figures, "distribution_test", False)
    write_csv(pd.DataFrame(summaries), raw / "property_summary.csv")
    funnel = funnel_table(generated, gt)
    write_csv(funnel, raw / "funnel.csv")
    draw_funnel(funnel, figures)
    draw_workflow(figures)
    draw_mmpa(args.dataset_dir, figures, raw)
    from generation_schematic import draw_generation_schematic
    draw_generation_schematic(args.generation_dir, args.evaluation_dir, args.dataset_dir, figures, raw)
    if args.structural_dir:
        structural_figures(args.structural_dir, figures, raw)
    for name in ("metrics.csv", "gt_metrics.csv", "gt_funnel.csv"):
        source = args.evaluation_dir / "metrics" / name
        if source.is_file():
            copy_file(source, raw / name)
    copy_file(args.run_manifest, raw / "run_manifest.json")
    copy_file(Path(__file__), args.output / Path(__file__).name)
    copy_file(Path(__file__).with_name("generation_schematic.py"), args.output / "generation_schematic.py")
    write_json(args.output / "figure_manifest.json", {
        "schema_version": 1, "run_manifest_sha256": sha256(args.run_manifest),
        "inputs": {"evaluations": {"sha256": sha256(generated_path), "records": len(generated)},
                   "gt_evaluations": {"sha256": sha256(gt_path), "records": len(gt)}},
        "populations": {"generated_properties": "Final survivors, deduplicated by parse-normalized canonical isomeric SMILES separately within each linker-size cohort; source SMILES preserved",
                        "gt_properties": "Final GT survivors, deduplicated by parse-normalized canonical isomeric SMILES; source SMILES preserved",
                        "funnel": "All generated attempts or GT fragmentation rows; cumulative rates use the original cohort count"},
        "artifacts": [{"path": str(p.relative_to(args.output)), "sha256": sha256(p), "bytes": p.stat().st_size}
                      for p in sorted(args.output.rglob("*")) if p.is_file()]})


if __name__ == "__main__":
    main()
