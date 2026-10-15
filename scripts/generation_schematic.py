#!/usr/bin/env python3
"""Select and draw CDK1 examples with their verified conditioning fragments."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from build_publication_assets import flag, save_figure, sha256, write_csv, write_json

CHECKS = ("anchor_attachment_pass", "retained_stereo_pass",
          "retained_nonanchor_explicit_h_match", "retained_nonanchor_total_h_match")
GREEN = (0.65, 0.83, 0.62)
MAGENTA = (0.83, 0.58, 0.78)


def select_examples(survivors):
    table = survivors.loc[survivors.pdb.str.lower().eq("5lqf")].copy()
    for column in ("final_pass", "connected", "novel") + CHECKS:
        if column not in table:
            raise ValueError(f"Generation illustration requires {column}")
        table = table.loc[flag(table[column])]
    table["linker_size"] = pd.to_numeric(table.linker_size, errors="raise").astype(int)
    table = table.loc[table.linker_size.between(8, 13)]
    table["qed"] = pd.to_numeric(table.qed, errors="raise")
    table["sa_score"] = pd.to_numeric(table.sa_score, errors="raise")
    table["mw_distance_400"] = abs(pd.to_numeric(table.molecular_weight, errors="raise") - 400)
    sizes = set(range(8, 14))
    context_sizes = table.groupby("fragmentation_id").linker_size.agg(set)
    common = sorted(context_sizes.index[context_sizes.apply(lambda values: values == sizes)])
    if common:
        table = table.loc[table.fragmentation_id.eq(common[0])]
        mode = "shared_fragment_pair"
    else:
        mode = "individual_fragment_pairs"
    selected = table.sort_values(["linker_size", "qed", "sa_score", "mw_distance_400", "candidate_id"],
                                 ascending=[True, False, True, True, True]).drop_duplicates("linker_size")
    return selected, {"mode": mode, "eligible_shared_contexts": common,
                      "selection_rule": "Lexicographically first fragmentation ID with eligible final survivors at all six sizes; within each size, descending QED, ascending SA score, absolute molecular-weight distance from 400 Da, then candidate ID. When no shared context spans all six sizes, apply the same within-size rank across contexts.",
                      "eligibility": ["PDB 5LQF", "final survivor", "connected", "novel"] + list(CHECKS)}


def selected_sdf_records(path, identifiers):
    from kinase_gap.structural_bundle import molecule_identifier, parse_sdf_bytes
    wanted, found, block = set(identifiers), {}, []
    with path.open("rb") as handle:
        for line in handle:
            block.append(line)
            if line.strip() != b"$$$$":
                continue
            name = block[0].decode("utf-8").strip()
            if name in wanted:
                if name in found:
                    raise ValueError(f"Repeated selected candidate in SDF: {name}")
                payload = b"".join(block)
                molecules = parse_sdf_bytes(payload, label=name)
                if len(molecules) != 1 or molecule_identifier(molecules[0], label=name) != name:
                    raise ValueError(f"Candidate SDF identity mismatch: {name}")
                found[name] = (molecules[0], payload)
            block = []
    if block and b"".join(block).strip():
        raise ValueError("Incomplete final SDF record")
    if set(found) != wanted:
        raise ValueError(f"Selected candidate SDFs missing: {sorted(wanted - set(found))}")
    return found


def molecule_image(molecule, retained=None, width=650, height=410):
    molecule = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(molecule)
    if retained is None:
        retained = set(range(molecule.GetNumAtoms()))
    colors = {i: GREEN if i in retained else MAGENTA for i in range(molecule.GetNumAtoms())}
    return Draw.MolToImage(molecule, size=(width, height), highlightAtoms=list(colors), highlightAtomColors=colors)


def draw_generation_schematic(generation, evaluation, dataset, directory, raw):
    from kinase_gap.structural_bundle import load_selected_fragments, validate_fragment_prefix
    source = evaluation / "metrics/final_survivors.csv"
    survivors = pd.read_csv(source, keep_default_na=False)
    selected, selection = select_examples(survivors)
    if selected.empty:
        raise ValueError("No verified CDK1 final survivor for the generation illustration")
    sdf = generation / "candidates.sdf"
    records = selected_sdf_records(sdf, selected.candidate_id)
    context_path = dataset / "KLIFS_test_table.csv"
    contexts = pd.read_csv(context_path, keep_default_na=False, dtype=str).set_index("fragmentation_id", drop=False)
    rows = selected.astype(str).to_dict("records")
    fragments, fragment_hash = load_selected_fragments(dataset / "KLIFS_test_frag.sdf", {"5lqf": rows})
    inputs, outputs, measurements = [], [], []
    destination = raw / "generation_examples"
    destination.mkdir()
    for row in rows:
        identifier = row["candidate_id"]
        molecule, payload = records[identifier]
        mask = validate_fragment_prefix(molecule, fragments[int(row["source_uuid"])], row)
        if len(mask["generated_linker_atom_indices"]) != int(row["linker_size"]):
            raise ValueError(f"Candidate linker atom count mismatch: {identifier}")
        context = contexts.loc[row["fragmentation_id"]]
        if isinstance(context, pd.DataFrame):
            raise ValueError("Fragmentation ID is not unique in the dataset")
        if int(context.uuid) != int(row["source_uuid"]):
            raise ValueError("Source UUID differs between candidate and test table")
        input_mol = Chem.MolFromSmiles(context.mmpa_fragments_with_dummies)
        if input_mol is None:
            raise ValueError("Unparsable input fragment pair")
        inputs.append(molecule_image(input_mol))
        outputs.append(molecule_image(molecule, set(mask["retained_fragment_atom_indices"])))
        with (destination / f"{identifier}.sdf").open("xb") as handle:
            handle.write(payload)
        write_json(destination / f"{identifier}.fragment_mask.json", mask)
        measurements.append({"candidate_id": identifier, "linker_size": int(row["linker_size"]),
                             "fragmentation_id": row["fragmentation_id"], "context": context.to_dict(),
                             "fragment_verification": mask["fragment_verification"]})
    shared = selection["mode"] == "shared_fragment_pair"
    fig = plt.figure(figsize=(15, 8.2) if shared else (15, 9))
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.set_xlim(0, 1); canvas.set_ylim(0, 1); canvas.axis("off")
    if shared:
        axis = fig.add_axes([0.015, 0.29, 0.235, 0.45]); axis.imshow(inputs[0]); axis.axis("off")
        canvas.text(.13, .78, "Two retained fragments", ha="center", fontsize=13)
        canvas.text(.13, .26, "* = attachment site", ha="center", fontsize=11)
        canvas.add_patch(FancyBboxPatch((.29,.41), .14,.19, boxstyle="round,pad=.015", facecolor="#eaf0f7", edgecolor="#506784"))
        canvas.text(.36,.525,"DiffLinker",ha="center",fontsize=14,weight="bold")
        canvas.text(.36,.46,"5LQF pocket",ha="center",fontsize=11)
        for left,right in [(.25,.275),(.445,.485)]:
            canvas.add_patch(FancyArrowPatch((left,.50),(right,.50),arrowstyle="-|>",mutation_scale=22,color="#506784"))
        for index,(row,image) in enumerate(zip(rows,outputs)):
            x=.49+(index%3)*.167; y=.53-(index//3)*.43
            axis=fig.add_axes([x,y,.165,.36]);axis.imshow(image);axis.axis("off")
            canvas.text(x+.0825,y+.355,f"{row['linker_size']} linker atoms",ha="center",fontsize=11)
    else:
        for index,(row,input_image,output_image) in enumerate(zip(rows,inputs,outputs)):
            x=.02+(index%2)*.5; y=.72-(index//2)*.31
            for offset,image in [(0,input_image),(.255,output_image)]:
                axis=fig.add_axes([x+offset,y,.22,.235]);axis.imshow(image);axis.axis("off")
            canvas.text(x+.238,y+.253,f"{row['linker_size']} linker atoms",ha="center",fontsize=12)
            canvas.add_patch(FancyArrowPatch((x+.218,y+.11),(x+.252,y+.11),arrowstyle="-|>",mutation_scale=18,color="#506784"))
            canvas.text(x+.235,y+.175,"DiffLinker",ha="center",fontsize=9)
        canvas.text(.5,.035,"5LQF pocket; each panel shows its own conditioning fragment pair (* = attachment site)",ha="center",fontsize=11)
    canvas.text(.5,.985,"Fragment-conditioned linker generation",ha="center",va="top",fontsize=16,weight="bold")
    canvas.text(.50,.002,"Green: retained fragments     Magenta: generated linker",ha="center",va="bottom",fontsize=11)
    save_figure(fig,directory,"generation")
    write_csv(selected, raw / "generation_selection.csv")
    selection.update({"target":"5lqf", "examples":measurements,
                      "inputs":{"final_survivors_sha256":sha256(source),"candidates_sdf_sha256":sha256(sdf),
                                "test_table_sha256":sha256(context_path),"test_fragments_sha256":fragment_hash}})
    write_json(raw / "generation_selection.json",selection)
    return selection
