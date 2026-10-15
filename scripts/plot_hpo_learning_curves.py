#!/usr/bin/env python3
"""Export training and validation traces for every declared HPO trial."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def fingerprint(path):
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": sha256(path), "bytes": path.stat().st_size}


def verified(value, digest):
    path = resolve(value)
    if sha256(path) != digest:
        raise ValueError("Input hash mismatch: " + str(path))
    return path


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, fields, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def epoch_points(rows):
    """Pair epoch-reduced losses by optimizer step, preserving logged epoch fields."""
    validation_epochs = {}
    for row in rows:
        if row.get("loss/val", ""):
            step, epoch = int(row["step"]), int(row["epoch"])
            if step in validation_epochs:
                raise ValueError("Validation metrics contain a duplicate optimizer step")
            validation_epochs[step] = epoch + 1
    points, seen = [], set()
    for index, row in enumerate(rows):
        for metric in ("loss/train", "loss/val"):
            text = row.get(metric, "")
            if not text:
                continue
            value = float(text)
            step = int(row["step"])
            if (metric, step) in seen:
                raise ValueError("An epoch loss was logged more than once")
            seen.add((metric, step))
            points.append({"source_row_index": index, "metric": metric, "logged_epoch": row["epoch"], "step": step,
                           "completed_epoch": validation_epochs.get(step, ""), "value_text": text,
                           "finite": int(math.isfinite(value)),
                           "plot_status": "plotted" if math.isfinite(value) and step in validation_epochs else "recorded"})
    return points


def collect(selection_path):
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "complete":
        raise ValueError("HPO selection must be complete before plotting")
    trials = selection["trials"]
    if len(trials) != selection["trial_count"] or {trial["trial_index"] for trial in trials} != set(range(len(trials))):
        raise ValueError("Selection does not contain every declared trial exactly once")
    sources, raw, points, summaries = {"selection": fingerprint(selection_path)}, [], [], []
    winner = selection["winner"]["trial_index"]
    for trial in sorted(trials, key=lambda row: row["trial_index"]):
        index = trial["trial_index"]
        if trial["outcome"] not in ("completed", "numerical_failure") or trial["eligible"] is not (trial["outcome"] == "completed"):
            raise ValueError("Unknown terminal trial outcome or inconsistent eligibility")
        metadata_path = verified(trial["run_metadata"], trial["run_metadata_sha256"])
        config_path = verified(trial["config"], trial["config_sha256"])
        metadata = json.loads(metadata_path.read_text())
        if metadata["run_id"] != trial["run_id"]:
            raise ValueError("Trial log identity mismatch")
        log = metadata_path.parent / "metrics.csv"
        sources[str(index)] = {"run_metadata": fingerprint(metadata_path), "config": fingerprint(config_path)}
        if trial["outcome"] == "completed":
            result = verified(trial["training_result"], trial["training_result_sha256"])
            sources[str(index)]["training_result"] = fingerprint(result)
        else:
            terminal = verified(trial["terminal_outcome"]["path"], trial["terminal_outcome"]["sha256"])
            outcome = json.loads(terminal.read_text())
            sources[str(index)]["terminal_outcome"] = fingerprint(terminal)
            if outcome["metrics"] is not None:
                if verified(outcome["metrics"]["path"], outcome["metrics"]["sha256"]) != log:
                    raise ValueError("Partial trace source differs from the registered outcome")
            elif log.exists():
                raise ValueError("A trace appeared after numerical-outcome registration")
        rows = []
        if log.exists():
            with log.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            sources[str(index)]["metrics"] = fingerprint(log)
        elif trial["outcome"] == "completed":
            raise ValueError("Completed trial lacks its metric trace")
        meta = {"trial_index": index, "run_id": trial["run_id"], "job_id": trial["job_id"],
                "outcome": trial["outcome"], "eligible": int(trial["eligible"]), "selected": int(index == winner),
                "learning_rate": trial["parameters"]["lr"], "batch_size": trial["parameters"]["batch_size"]}
        for row_index, row in enumerate(rows):
            raw.append(dict(meta, source_row_index=row_index, source_log=sources[str(index)]["metrics"]["path"],
                            raw_metric_row_json=json.dumps(row, sort_keys=True, separators=(",", ":"))))
        trial_points = epoch_points(rows)
        points.extend(dict(meta, **point) for point in trial_points)
        validation = [point for point in trial_points if point["metric"] == "loss/val" and point["finite"]]
        if trial["eligible"]:
            best = [point for point in validation if point["completed_epoch"] == trial["best_epoch"] + 1]
            if len(best) != 1 or not math.isclose(float(best[0]["value_text"]), trial["objective"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Selected validation objective differs from the recorded epoch trace")
        summaries.append(dict(meta, recorded_validation_epochs=len(validation),
                              recorded_training_epochs=sum(point["metric"] == "loss/train" and point["finite"] for point in trial_points),
                              best_recorded_validation=min((float(point["value_text"]) for point in validation), default=""),
                              selected_epoch=trial["best_epoch"] + 1 if trial["eligible"] else "",
                              trace_label="Completed training" if trial["eligible"] else "Recorded epochs (partial)"))
    return selection, sources, raw, points, summaries


def run(args):
    selection_path, output = resolve(args.selection), resolve(args.output)
    selection, sources, raw, points, summaries = collect(selection_path)
    output.mkdir(parents=True, exist_ok=False)
    meta_fields = ["trial_index", "run_id", "job_id", "outcome", "eligible", "selected", "learning_rate", "batch_size"]
    write_csv(output / "metric_rows.csv", meta_fields + ["source_row_index", "source_log", "raw_metric_row_json"], raw)
    write_csv(output / "learning_curve_points.csv", meta_fields + ["source_row_index", "metric", "logged_epoch", "step", "completed_epoch", "value_text", "finite", "plot_status"], points)
    write_csv(output / "trial_summary.csv", meta_fields + ["recorded_validation_epochs", "recorded_training_epochs", "best_recorded_validation", "selected_epoch", "trace_label"], summaries)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = min(4, len(summaries))
    rows = math.ceil(len(summaries) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.7 * columns, 3 * rows), squeeze=False, sharex=True, sharey=True)
    for ax, summary in zip(axes.flat, summaries):
        index = summary["trial_index"]
        trial_points = [point for point in points if point["trial_index"] == index and point["plot_status"] == "plotted"]
        for metric, color, label in (("loss/train", "#2166ac", "Training"), ("loss/val", "#c45b20", "Validation")):
            values = sorted((point for point in trial_points if point["metric"] == metric), key=lambda point: point["completed_epoch"])
            ax.plot([point["completed_epoch"] for point in values], [float(point["value_text"]) for point in values],
                    color=color, label=label, linewidth=1.2, marker="o" if len(values) < 3 else None,
                    linestyle="-" if summary["eligible"] else "--")
        title = "Trial %02d · lr=%g · batch=%s" % (index, summary["learning_rate"], summary["batch_size"])
        if summary["selected"]:
            title += "\nSelected checkpoint"
            ax.axvline(summary["selected_epoch"], color="#4a7c59", linewidth=.8, alpha=.7)
        elif not summary["eligible"]:
            title += "\nRecorded epochs (partial)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Completed epoch")
        ax.set_ylabel("Epoch mean loss")
        ax.grid(alpha=.18)
    for ax in list(axes.flat)[len(summaries):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / ("hpo_learning_curves." + suffix), dpi=220)
    plt.close(fig)
    provenance = {"schema_version": 1, "study_id": selection["study_id"], "sources": sources,
                  "script": fingerprint(Path(__file__).resolve()), "matplotlib_version": matplotlib.__version__,
                  "trial_count": selection["trial_count"], "selected_trial_index": selection["winner"]["trial_index"],
                  "epoch_alignment": "Validation logged epoch plus one; training and validation epoch means paired by their common optimizer step",
                  "raw_rows": "Every recorded CSV row is preserved with its original field values",
                  "partial_traces": "Recorded epoch means for terminal partial trials"}
    write_json(output / "provenance.json", provenance)
    outputs = {path.name: fingerprint(path) for path in sorted(output.iterdir())}
    done = {"schema_version": 1, "status": "complete", "trial_count": len(summaries), "raw_metric_rows": len(raw), "outputs": outputs}
    write_json(output / "done.json", done)
    print(json.dumps({key: value for key, value in done.items() if key != "outputs"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
