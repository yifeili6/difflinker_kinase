#!/usr/bin/env python3
"""Arrange completed LigPlot panels using their recorded candidate order."""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.contacts / "contact_summary.csv").open() as handle:
        records = list(csv.DictReader(handle))
    font = ImageFont.truetype("DejaVuSans.ttf", 44)
    legend_font = ImageFont.truetype("DejaVuSans.ttf", 30)
    manifest = {}
    for target, count, basename in [("5lqf", 3, "CONTACT_FIG8"), ("4ag8", 6, "SI_CONTACT_FIGS5")]:
        selected = [r for r in records if r["target"] == target and (r["panel"] == "native" or r["panel"] in [f"candidate_{i:02d}" for i in range(1, count + 1)])]
        selected.sort(key=lambda r: (r["panel"] != "native", r["panel"]))
        if not selected or selected[0]["panel"] != "native":
            raise ValueError(f"Expected a native reference panel for {target}")
        columns = 2 if target == "5lqf" else 3
        width, height = 1500, 1600
        rows = (len(selected) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * width, rows * height + 130), "white")
        painter = ImageDraw.Draw(canvas)
        for index, record in enumerate(selected):
            with Image.open(args.contacts / target / record["panel"] / "diagram.png") as opened:
                im = opened.convert("RGB")
                box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
                if box:
                    im = im.crop(box)
                im.thumbnail((width - 90, height - 130), Image.Resampling.LANCZOS)
                x, y = (index % columns) * width, (index // columns) * height
                canvas.paste(im, (x + (width - im.width) // 2, y + 100 + (height - 120 - im.height) // 2))
                title = "Native ligand" if record["panel"] == "native" else "Candidate " + str(int(record["panel"].split("_")[-1]))
                painter.text((x + 35, y + 30), f"({chr(97 + index)}) {title}", font=font, fill="black")
        painter.text((35, rows * height + 15), "Green dashed lines: hydrogen bonds with donor–acceptor distances in Å", font=legend_font, fill="#126926")
        painter.text((35, rows * height + 63), "Red arcs: hydrophobic contacts; purple bonds: ligand; brown bonds: protein", font=legend_font, fill="black")
        with (args.output / (basename + ".png")).open("xb") as handle:
            canvas.save(handle, "PNG")
        with (args.output / (basename + ".pdf")).open("xb") as handle:
            canvas.save(handle, "PDF", resolution=300)
        manifest[target] = {"maximum_generated_panels": count,
                            "generated_panels": len(selected) - 1, "panels": selected}
    with (args.output / "panel_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
