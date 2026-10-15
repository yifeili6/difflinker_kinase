# LigPlot panels for manuscript figures

## Locate the input records

Use the environment described in the [archive README](../README.md). From the extracted archive root, set the completed figure directory:

```bash
export KINASE_ARCHIVE="$PWD"
export KG_PYTHON="${KINASE_RUNTIME_ENV:?Set the recorded environment prefix}/bin/python"
RUN_ID=$("$KG_PYTHON" -c 'import json; r=json.load(open("provenance/run_manifest.json")); assert r["status"] == "complete"; print(r["run_id"])')
FIGURES="$KINASE_ARCHIVE/figures/$RUN_ID"
```

All paths in the table are relative to `$FIGURES`.

| Complex | Structural identity | Editable drawing | Molecular input directory |
|---|---|---|---|
| CDK1, PDB 5LQF | `human__CDC2__5lqf__altA__chainA` | `ligplot/5lqf/native/ligplot.drw` | `contact_inputs/5lqf/native/` |
| VEGFR2, PDB 4AG8 | `human__KDR__4ag8__alt-__chainA` | `ligplot/4ag8/native/ligplot.drw` | `contact_inputs/4ag8/native/` |

Each molecular input directory contains `complex.pdb`, `receptor.pdb`, `ligand.sdf`, `ligand.cif`, `atom_mapping.json` and atom-role definitions. Each drawing directory retains the corresponding molecular files, raw HBPLUS records, plotted contact records, `hydrogen_bonds.json` and `execution.json`. The ligand identifier is **LIG**, chain **Z**, residue **900**; receptor residues retain chain **A**.

Generated panels use the corresponding `candidate_01`, `candidate_02`, … directories. The figure's candidate identities and order are recorded in `ligplot/contact_summary.csv` and `contacts/panel_manifest.json`.

## Prepare the drawing software

Install LigPlot+ following its supplied installation instructions. Set `LIGPLUS_DIR` to the local installation directory containing `lib/exe_linux64`; its manual is `$LIGPLUS_DIR/lib/docs/manual.html`. Sections 4a and 6b describe moving atoms, residues and text, and adjusting drawing sizes.

Install Ghostscript with the workstation's package manager so `gs` and `ps2pdf` are on `PATH`. Check the conversion commands:

```bash
command -v gs
command -v ps2pdf
gs --version
```

[Figure reproduction](FIGURE_REPRODUCTION.md) gives the commands for rebuilding contact inputs and automated diagrams using the recorded chemistry and interaction settings.

## Edit the two-dimensional layout

1. Copy each complete drawing directory into a new editing directory outside the archive. Open the copied `ligplot.drw` through LigPlot+'s file-open dialog to load its drawing and recorded contacts.
2. In the 5LQF drawing, use **Move atom** to position the two sulfonyl oxygen atoms on separate sides of sulfur. Position the attached nitrogen so the S–N and S=O bonds are individually visible. Use `ligand.sdf`, `ligand.cif` and `atom_mapping.json` to retain each atom identity, bond order and interaction.
3. Use **Move residue** to place interacting residues around the ligand. Position residue and distance labels in clear space, keeping each green line and red contact arc associated with its recorded atom or residue.
4. Adjust label sizes through the **Sizes** dialog. Check the exported figure at its final physical width and use at least 8-point residue and distance labels.
5. Save a new `.drw` file and export a new PostScript file. Keep an edit record listing moved atoms and labels alongside the input drawings and contact tables.

## Arrange and export

1. Set the composition width before arranging panels. CDK1 `CONTACT_FIG8` uses 5.949974 inches (`results.tex`, `1\textwidth`); VEGFR2 `SI_CONTACT_FIGS5` uses 4.759979 inches (`SI.tex`, `0.8\textwidth`). Use the same scale for comparable panels, with continuation pages as needed to retain 8-point labels.
2. Keep the native ligand as panel (a), followed by generated panels in the recorded order. Add panel letters and target/PDB identities. Retain complete contact arcs, residue labels and distance labels inside each panel.
3. Include an 8-point or larger legend: green dashed lines indicate hydrogen bonds with donor–acceptor distances in Å; red arcs indicate hydrophobic contacts; ligand bonds are purple and protein bonds are brown.
4. Export a vector PDF at the chosen physical width and a PNG at 600 dpi. At these widths, the PNGs are approximately 3570 pixels for CDK1 and 2856 pixels for VEGFR2. Final placement determines printed label size.
5. Inspect the PDF at actual size and the PNG at the same width. Check the 5LQF sulfonamide bonds, the 4AG8 Lys920 label, panel letters and legend text. Keep the final PDF, PNG, `.drw`, PostScript and edit record together.

## Verify contact labels

The native 5LQF panel displays four hydrogen bonds: Glu81 (2.80 Å), Leu83 (3.07 and 2.82 Å), and Asp86 (3.15 Å). The native 4AG8 panel displays Glu885 (2.70 Å), Glu917 (2.88 Å), Cys919 (2.97 Å) and Asp1046 (2.97 Å). Match labels to records with `figure_pass=true` in each panel's `hydrogen_bonds.json`.

The recorded hydrogen-bond definition uses donor–acceptor distance ≤3.35 Å, hydrogen–acceptor distance ≤2.70 Å and defined donor/acceptor angles ≥90°. Preserve the raw and plotted contact records with the drawings. Record SHA-256 hashes of final exports and input drawings in the figure manifest.
