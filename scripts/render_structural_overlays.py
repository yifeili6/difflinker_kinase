#!/usr/bin/env python3
"""Render saved structural coordinates with PyMOL in a new output directory."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-python", default=sys.executable,
                        help="Python executable with Pillow for PNG-to-PDF conversion")
    args = parser.parse_args()
    if "kinase-abr" in str(args.structural_dir.resolve()):
        raise ValueError("Excluded source path")
    manifest = json.loads((args.structural_dir / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    import pymol
    pymol.finish_launching(["pymol", "-cq"])
    cmd = pymol.cmd
    records = []
    for target in ("5lqf", "4ag8"):
        cmd.reinitialize()
        root = args.structural_dir / "targets" / target
        cmd.load(str(root / "reference/protein_reference.pdb"), "receptor")
        cmd.load(str(root / "reference/native_reference.sdf"), "native")
        cmd.hide("everything", "all")
        cmd.show("cartoon", "receptor")
        cmd.color("gray70", "receptor")
        cmd.show("sticks", "native")
        cmd.color("green", "native")
        inputs = [root / "reference/protein_reference.pdb", root / "reference/native_reference.sdf"]
        for index, candidate in enumerate(manifest["targets"][target]["candidates"]):
            path = args.structural_dir / candidate["directory"] / "candidate_aligned.sdf"
            name = f"generated_{index:02d}"
            cmd.load(str(path), name)
            cmd.show("sticks", name)
            cmd.color("magenta", name)
            inputs.append(path)
        cmd.select("native_neighbours", "byres (receptor within 4.0 of native)")
        cmd.show("sticks", "native_neighbours")
        cmd.color("slate", "native_neighbours")
        cmd.hide("everything", "elem H")
        cmd.set("cartoon_transparency", 0.65)
        cmd.set("stick_radius", 0.16)
        cmd.set("ray_opaque_background", 1)
        cmd.set("antialias", 2)
        cmd.set("orthoscopic", 1)
        cmd.bg_color("black")
        cmd.orient("native")
        cmd.zoom("native or generated_*", buffer=5)
        base = "drug_overlayed" + ("_type2" if target == "4ag8" else "")
        cmd.png(str(args.output / (base + ".png")), width=2400, height=1800, dpi=300, ray=1)
        cmd.save(str(args.output / (base + ".pse")))
        subprocess.run([args.image_python, "-B", "-c",
                        "import sys; from PIL import Image; "
                        "im=Image.open(sys.argv[1]).convert('RGB'); "
                        "handle=open(sys.argv[2],'xb'); im.save(handle,'PDF',resolution=300); handle.close()",
                        str(args.output / (base + ".png")), str(args.output / (base + ".pdf"))], check=True)
        records.append({"target": target, "candidate_ids": [c["candidate_id"] for c in manifest["targets"][target]["candidates"]],
                        "view": list(cmd.get_view()), "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs]})
    with (args.output / "overlay_manifest.json").open("x") as handle:
        json.dump({"software": "PyMOL", "version": cmd.get_version()[0], "targets": records}, handle, indent=2)
    cmd.quit()


if __name__ == "__main__":
    main()
