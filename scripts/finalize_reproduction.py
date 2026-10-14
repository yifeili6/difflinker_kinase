#!/usr/bin/env python3
"""Validate a reproduced study and write its immutable publication run manifest.

All input/output arguments other than --project-root are relative to the project
root. Create the manifest after structural/contact analysis and before running
build_publication_assets.py. Existing manifest files are never overwritten.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path

from publication_validation import descriptions, done, execution_seals, files, fingerprint, load, project_path, shards, verify_file


def finalize(args):
    root = args.project_root.resolve()
    path = lambda name: project_path(root, name)
    output = path(args.output)
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}")
    selection_name = args.selection or f"outputs/hpo/studies/{args.study_id}/selection.json"
    receptor_name = args.receptor_diagnostics or args.evaluation + "/receptor_diagnostics"
    learning_name = args.hpo_learning_curves or f"outputs/hpo/studies/{args.study_id}/learning_curves"
    generation = done(root, args.generation + "/done.json", {"run_id": args.study_id})
    evaluation = done(root, args.evaluation + "/done.json", {"run_id": args.study_id, "record_count": generation["candidate_count"]})
    gt = done(root, args.evaluation + "/gt_done.json", {"run_id": args.study_id})
    metrics = done(root, args.evaluation + "/metrics/done.json", {
        "candidate_count": generation["candidate_count"], "final_survivor_count": evaluation["counts"]["final_pass"],
        "gt_candidate_count": gt["record_count"], "gt_final_survivor_count": gt["counts"]["final_pass"]})
    generated_provenance = load(path(args.generation + "/provenance.json"))
    evaluated_provenance = load(path(args.evaluation + "/provenance.json"))
    gt_provenance = load(path(args.evaluation + "/gt_provenance.json"))
    for key, suffix in [('table', 'table.csv'), ('molecules', 'mol.sdf'), ('chemistry', 'chemistry.jsonl')]:
        source = gt_provenance['source']
        expected = args.dataset + '/KLIFS_test_' + suffix
        if source[key] != expected:
            raise ValueError(f'Ground-truth {key} does not belong to the declared test dataset')
        verify_file(path(expected), {'sha256': source[key + '_sha256']})
    recorded_gt = evaluated_provenance.get('contract', {}).get('ground_truth')
    if recorded_gt is not None and recorded_gt != gt_provenance['source']:
        raise ValueError('Generated and ground-truth evaluation contracts bind different test sources')
    for record, count in [(generated_provenance, generation["condition_shards"]),
                          (evaluated_provenance, evaluation["shard_count"]), (gt_provenance, gt["shard_count"])]:
        if record.get("run_id") != args.study_id:
            raise ValueError("Shard provenance has a different study ID")
        shards(root, record, count)
    if evaluated_provenance["generation"]["done_sha256"] != fingerprint(path(args.generation + "/done.json"))["sha256"]:
        raise ValueError("Evaluation binds a different generation completion record")
    if evaluated_provenance["generation"]["contract_sha256"] != generation["contract_sha256"]:
        raise ValueError("Generation/evaluation contract mismatch")
    metric_provenance = load(path(args.evaluation + "/metrics/provenance.json"))
    for name, info in metric_provenance["inputs"].items():
        verify_file(path(name), info)
    contract = generated_provenance["contract"]
    if contract["expected_candidates"] != generation["candidate_count"]:
        raise ValueError("Generation contract count differs from completed candidates")
    verify_file(path(contract["config"]), {"sha256": contract["config_sha256"]})
    for info in contract["data_inputs"].values():
        verify_file(path(info["path"]), info)
    verify_file(path(contract["targets_manifest"]["path"]), contract["targets_manifest"])
    selection = load(path(selection_name))
    if selection.get("status") != "complete" or selection.get("study_id") != args.study_id:
        raise ValueError("HPO selection is not complete for this study")
    trials = selection['trials']
    if len(trials) != selection['trial_count'] or {t['trial_index'] for t in trials} != set(range(selection['trial_count'])):
        raise ValueError('HPO selection does not contain every declared trial exactly once')
    for trial in trials:
        if trial['outcome'] not in ('completed', 'numerical_failure') or trial['eligible'] is not (trial['outcome'] == 'completed'):
            raise ValueError('HPO outcome and eligibility are inconsistent')
    hpo_binding = generated_provenance['checkpoint']['hpo_selection']
    if hpo_binding['path'] != selection_name:
        raise ValueError('Generation names a different HPO selection')
    verify_file(path(selection_name), hpo_binding)
    winner = selection["winner"]
    if not math.isfinite(float(winner["objective"])):
        raise ValueError("Selected validation objective is non-finite")
    verify_file(path(winner["selected_checkpoint"]), {"sha256": winner["selected_checkpoint_sha256"]})
    verify_file(path(winner["config"]), {"sha256": winner["config_sha256"]})
    model = load(path(args.model_validation))
    if (model.get("status") != "validated" or model.get("device") != "cpu" or
            model.get("strict_state_dict") is not True or model.get("finite_model_state") is not True):
        raise ValueError("Model needs validated strict finite-state loading on CPU")
    if model["selection_sha256"] != fingerprint(path(selection_name))["sha256"]:
        raise ValueError("Model validation binds a different HPO selection")
    if model["checkpoint"]["sha256"] != winner["selected_checkpoint_sha256"] or generated_provenance["checkpoint"]["sha256"] != winner["selected_checkpoint_sha256"]:
        raise ValueError("Model validation, selected checkpoint and generated outputs differ")
    population = load(path(args.population))
    dataset = args.dataset
    for split, prefix in [("train", "train"), ("validation", "val"), ("test", "test")]:
        verify_file(path(dataset + f"/KLIFS_{prefix}_table.csv"), {"sha256": population["splits"][split]["sha256"]})
    receptor = done(root, receptor_name + "/done.json", {"mode": "run", "generated_records": metrics["final_survivor_count"]})
    done(root, learning_name + "/done.json", {"trial_count": selection["trial_count"]})
    receptor_provenance = load(path(receptor_name + '/provenance.json'))
    descriptions(root, receptor_provenance['inputs'])
    descriptions(root, receptor_provenance['receptors'])
    descriptions(root, receptor_provenance['sources'])
    learning_provenance = load(path(learning_name + '/provenance.json'))
    if learning_provenance['sources']['selection']['path'] != selection_name:
        raise ValueError('Learning curves name a different HPO selection')
    descriptions(root, learning_provenance['sources'])
    report = load(path(args.report + '/report_manifest.json'))
    report_inputs = {'candidate_evaluation': args.evaluation + '/metrics/final_survivors.csv',
                     'gt_evaluation': args.evaluation + '/gt_evaluations.csv',
                     'funnel': args.evaluation + '/metrics/funnel.csv', 'candidate_sdf': args.generation + '/candidates.sdf'}
    for key, name in report_inputs.items():
        if report['inputs'][key]['path'] != name:
            raise ValueError(f'Report input differs from the completed run: {key}')
        verify_file(path(name), report['inputs'][key])
    selected_examples = load(path(args.report + '/ranking/selection.json'))
    for key, name in [('candidate_evaluation_csv', report_inputs['candidate_evaluation']), ('candidate_sdf', report_inputs['candidate_sdf'])]:
        if selected_examples['inputs'][key]['path'] != name:
            raise ValueError(f'Structural selection input differs from the report: {key}')
        verify_file(path(name), selected_examples['inputs'][key])
    structural = load(path(args.structural + "/manifest.json"))
    inputs = structural["inputs"]
    expected_selection = args.report + "/ranking/selection.json"
    if inputs["selection_json"]["path"] != expected_selection:
        raise ValueError("Structural bundle does not bind the declared report selection")
    from kinase_gap.structural_bundle import verify_bundle_inputs
    verify_bundle_inputs(bundle_dir=path(args.structural), selection_json=path(expected_selection),
        contexts_csv=path(inputs["contexts_csv"]["path"]),
        snapshot_dir=path(inputs["snapshot_files_sha256"]["path"]).parent,
        fragment_sdf=path(inputs["fragment_sdf"]["path"]))
    contact_manifest = path(args.contact_inputs + "/source_structural_manifest.json")
    verify_file(contact_manifest, fingerprint(path(args.structural + "/manifest.json")))
    import csv
    with path(args.contact_inputs + "/contact_inputs.csv").open() as handle:
        contacts = list(csv.DictReader(handle))
    with path(args.structural_contacts + "/summary.csv").open() as handle:
        distances = list(csv.DictReader(handle))
    contact_keys = {(r["target"], r["panel"], r["candidate_id"]) for r in contacts}
    distance_keys = {(r["target"], r["panel"], r["candidate_id"]) for r in distances}
    expected_keys = set()
    for target, record in structural['targets'].items():
        expected_keys.add((target, 'native', 'native'))
        expected_keys.update((target, f'candidate_{i:02d}', candidate['candidate_id']) for i, candidate in enumerate(record['candidates'], start=1))
    if len(contact_keys) != len(contacts) or contact_keys != distance_keys or len(distance_keys) != len(distances) or contact_keys != expected_keys:
        raise ValueError("Contact inputs and receptor-distance records do not cover the same unique cases")
    for row in contacts:
        case = path(args.contact_inputs + "/" + row["complex_pdb"]).parent
        verify_file(path(args.contact_inputs + '/' + row['complex_pdb']), {'sha256': row['complex_sha256']})
        verify_file(path(args.contact_inputs + '/' + row['ligand_sdf']), {'sha256': row['ligand_sha256']})
        diagnostic = next(r for r in distances if (r["target"], r["panel"], r["candidate_id"]) == (row["target"], row["panel"], row["candidate_id"]))
        verify_file(case / "ligand.sdf", {"sha256": diagnostic["ligand_sha256"]})
        verify_file(case / "receptor.pdb", {"sha256": diagnostic["receptor_sha256"]})
    artifact_directories = {"report": args.report, "structural": args.structural, "contact_inputs": args.contact_inputs,
                            "structural_contacts": args.structural_contacts, "receptor_diagnostics": receptor_name,
                            "hpo_learning_curves": learning_name}
    for name in ("ligplot", "contact_figures", "overlays"):
        if getattr(args, name):
            artifact_directories[name] = getattr(args, name)
    artifact_hashes = {name: files(root, directory) for name, directory in artifact_directories.items()}
    source_provenance = execution_seals(root)
    marker_names = {"generation": args.generation + "/done.json", "evaluation": args.evaluation + "/done.json",
                    "ground_truth": args.evaluation + "/gt_done.json", "metrics": args.evaluation + "/metrics/done.json",
                    "receptor_diagnostics": receptor_name + "/done.json", "hpo_learning_curves": learning_name + "/done.json"}
    manifest = {"schema_version": 1, "status": "complete", "run_id": args.study_id,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "paths": {"generation": args.generation, "evaluation": args.evaluation, "dataset": dataset,
                  "structural": args.structural, "generation_config": contract["config"], "training_config": winner["config"],
                  **artifact_directories},
        "model": {"checkpoint": winner["selected_checkpoint"], "sha256": winner["selected_checkpoint_sha256"],
                  "hpo_selection": selection_name, "hpo_selection_sha256": fingerprint(path(selection_name))["sha256"],
                  "validation": args.model_validation, "selection": winner},
        "generation": {"attempts": generation["candidate_count"], "contexts": contract["representative_context_count"],
                       "linker_sizes": contract["linker_sizes"], "samples_per_condition": contract["samples_per_condition"],
                       "shards": generation["condition_shards"], "contract_sha256": generation["contract_sha256"]},
        "evaluation": {"generated": evaluation["counts"], "ground_truth": gt["counts"]},
        "dataset": {"population_summary": args.population, "splits": population["splits"]},
        "receptor_diagnostics": {"generated_records": receptor["generated_records"], "native_contexts": receptor["native_contexts"]},
        "scientific_inputs": contract,
        "source_provenance": source_provenance,
        "scientific_completion_records": {label: {"path": name, **fingerprint(path(name))} for label, name in marker_names.items()},
        "software": {"generation": generated_provenance.get("software", {}), "publication": model["software"]},
        "structural_and_contact_artifacts": artifact_hashes,
        "finalization_source": {"path": "scripts/" + Path(__file__).name, **fingerprint(Path(__file__))}}
    if args.validate_only:
        print(json.dumps({"status": "validated", "run_id": args.study_id, "generation_attempts": generation["candidate_count"]}))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": "complete", "path": args.output, **fingerprint(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path('.'))
    for name in ("study-id", "generation", "evaluation", "report", "structural", "model-validation", "population", "contact-inputs", "structural-contacts"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset", default="data/processed/klifs_v1")
    parser.add_argument("--selection")
    parser.add_argument("--receptor-diagnostics")
    parser.add_argument("--hpo-learning-curves")
    parser.add_argument("--ligplot")
    parser.add_argument("--contact-figures")
    parser.add_argument("--overlays")
    parser.add_argument("--output", default="provenance/run_manifest.json")
    parser.add_argument("--validate-only", action="store_true")
    finalize(parser.parse_args())


if __name__ == "__main__":
    main()
