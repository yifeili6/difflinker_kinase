# Figure reproduction

Each figure is supplied with its plotting data, input identifiers and rendering script. The figure manifest records file checksums and cohort definitions. Molecular maps also include the selected SDFs, retained-fragment masks and atom weight arrays.

The manuscript figure set is in `figures/reviewed_20261002/publication/`. Follow [Property figures at manuscript size](PROPERTY_FIGURES.md) for the final-size property panels and [LigPlot panel preparation](CONTACT_PANEL_PREPARATION.md) for interaction-diagram layout and export.

[Checkpoint continuation](CHECKPOINT_CONTINUATION.md) gives the command for rebuilding HPO learning curves from the recorded parent and continuation metric tables, aligned by optimizer step.

## Property figures and workflow diagram

Create the evaluation environment using the [archive README](../README.md), then run these commands from the archive root. Set `KINASE_RUNTIME_ENV` to that environment's absolute prefix. Read the input paths from the run manifest and choose a new output directory:

```bash
export KINASE_RUNTIME_ENV="${KINASE_RUNTIME_ENV:?Set the recorded environment prefix}"
export KG_PYTHON="$KINASE_RUNTIME_ENV/bin/python"
export LD_LIBRARY_PATH="$KINASE_RUNTIME_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/vendor/DiffLinker"
EVALUATION_DIR=$("$KG_PYTHON" -c 'import json; print(json.load(open("provenance/run_manifest.json"))["paths"]["evaluation"])')
GENERATION_DIR=$("$KG_PYTHON" -c 'import json; print(json.load(open("provenance/run_manifest.json"))["paths"]["generation"])')
DATASET_DIR=$("$KG_PYTHON" -c 'import json; print(json.load(open("provenance/run_manifest.json"))["paths"]["dataset"])')
STRUCTURAL_DIR=$("$KG_PYTHON" -c 'import json; print(json.load(open("provenance/run_manifest.json"))["paths"]["structural"])')
"$KG_PYTHON" scripts/build_publication_assets.py \
  --evaluation-dir "$EVALUATION_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --generation-dir "$GENERATION_DIR" \
  --structural-dir "$STRUCTURAL_DIR" \
  --run-manifest provenance/run_manifest.json \
  --output figures_rebuilt
```

The script writes PNG and PDF files, the exact property populations, the survival table and property summaries. It reads all attempted samples for filtering denominators and uses final survivors for property plots. The fragmentation illustration uses a CDK1/5LQF test context and marks the original cut-bond endpoints with dummy atoms.

The generation illustration selects a CDK1 fragment pair with verified final survivors at all six linker lengths when such a pair is available. Within each length, examples are ranked by QED, SA score, distance of molecular weight from 400 Da and candidate identifier. Each example passes the recorded attachment, retained-stereochemistry and nonanchor-hydrogen checks. The selection table, input pair and molecule files accompany the figure. When the six examples use different pairs, each panel displays its own input.

## Protein and ligand overlays

Create the recorded PyMOL environment at a new absolute prefix, then run the overlay script. Conda must be available; set `KINASE_CONDA` to its executable path when needed. For an environment already created from this lock file, set `KINASE_PYMOL_ENV` to its prefix and run the rendering command.

```bash
export KINASE_PYMOL_ENV="$PWD/../kinase-pymol"
"${KINASE_CONDA:-conda}" create --yes --prefix "$KINASE_PYMOL_ENV" \
  --file environment/pymol-explicit-lock.txt
env LD_LIBRARY_PATH="$KINASE_PYMOL_ENV/lib" \
  "$KINASE_PYMOL_ENV/bin/python" scripts/render_structural_overlays.py \
  --structural-dir "$STRUCTURAL_DIR" \
  --image-python "$KG_PYTHON" \
  --output overlays_rebuilt
```

The overlay command uses Pillow in the evaluation environment to create the PDF copies.

The saved PyMOL sessions contain the native ligand in green, generated ligands in magenta and nearby native-contact residues in slate. The manifest records the camera and selected candidate identifiers. Open a `.pse` file in PyMOL to change the camera or labels while retaining the molecular coordinates.

## Prepare interaction diagram inputs

Build a new contact-input folder:

```bash
"$KG_PYTHON" scripts/package_structural_inputs.py \
  --structural-dir "$STRUCTURAL_DIR" \
  --output contact_inputs_rebuilt
```

For each target, the folder contains a native complex and one complex per selected generated candidate. Each entry supplies `complex.pdb`, `receptor.pdb`, `ligand.sdf` and `atom_mapping.json`. The mapping connects zero-based SDF atom indices to PDB atom names and lists atom charges, hydrogen counts and bond types. The complex PDB identifies the ligand as **LIG**, chain **Z**, residue **900**.

The package also supplies a ligand dictionary in CCD format and HBPLUS atom definitions based on the recorded RDKit donor and acceptor SMARTS. The atom-role table records each assignment. Install Ghostscript using the workstation's package manager so `gs` and `ps2pdf` are on `PATH`; these commands provide PDF and PNG conversion.

```bash
command -v gs
command -v ps2pdf
gs --version
```

For automated rendering, set `LIGPLUS_DIR` to the installed LigPlus directory containing `lib/exe_linux64` and run:

```bash
"$KG_PYTHON" scripts/run_ligplot_inputs.py \
  --inputs contact_inputs_rebuilt --ligplus "$LIGPLUS_DIR" \
  --output contacts_rebuilt
"$KG_PYTHON" scripts/assemble_contact_figures.py \
  --contacts contacts_rebuilt --output contact_figures_rebuilt
```

Hydrogen-bond diagrams use donor–acceptor distances of at most 3.35 Å, hydrogen–acceptor distances of at most 2.70 Å, and the recorded donor/acceptor angles of at least 90 degrees. All plotted hydrogen bonds have defined geometry values. The output includes the original HBPLUS records, plotted contact records, atom definitions, display settings, program logs and software hashes.

## Receptor distance measurements

Calculate the accompanying heavy-atom distance tables with:

```bash
"$KG_PYTHON" scripts/structural_distance_qc.py \
  --inputs contact_inputs_rebuilt --output distances_rebuilt
```

The tables record nearest receptor atoms, contacts within 6 Å and distances divided by the sum of RDKit van der Waals radii. The reported overlap threshold is 0.75 times that sum. These measurements are a separate structural analysis with their own recorded settings.

## Create LigPlot diagrams

1. Install [LigPlot+](https://www.ebi.ac.uk/thornton-srv/software/LigPlus/) using its supplied installation instructions and configure the bundled HBPLUS/HBADD executables and ligand dictionary.
2. Open the target's native `complex.pdb` with **File → Open → PDB file**. Select **LIG Z 900**, or enter residue range **900 900 Z**.
3. Check the ligand's displayed bonds and donor/acceptor assignments against `ligand.sdf` and `atom_mapping.json`. Use a ligand dictionary entry with the supplied atom names for each generated molecule. Keep each molecule's dictionary entry with its diagram outputs.
4. Use the same hydrogen-bond and contact settings for the native complex and generated candidates. Save the parameter settings and program versions with the contact tables.
5. Generate the diagram. Display hydrogen-bond distances in Å and residue names, numbers and chains. Adjust two-dimensional label positions to separate overlapping labels.
6. Repeat for the selected generated complexes. Arrange the native diagram as panel (a), followed by candidate panels in the order of `contact_inputs.csv`. Use the candidate identifiers, panel counts and order recorded in `panel_manifest.json` for the CDK1 main figure and VEGFR2 supplementary figure.
7. Save the editable LigPlot files, hydrogen-bond and non-bonded contact tables, and PostScript output. Convert the PostScript to PDF with `ps2pdf diagram.ps diagram.pdf`. Export PNG at 300 dpi for manuscript previews.

The official [LigPlot+ distribution](https://github.com/RomanLas/LigPlus) supplies its manual. Keep all contact-analysis settings and output tables with the figure so each plotted interaction can be checked against its atom names and measured distance.
