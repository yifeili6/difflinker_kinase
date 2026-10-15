# Figure materials

Create the plotting environment with `conda env create -f environment.yml`, then activate `kinase-paper-figures`. Start JupyterLab and open a figure notebook from its own directory. Run its cells in order. The last cell writes PNG and PDF files to that figure’s `output/` directory.

Figures 2 and S3 use final-survivor CSV tables. Figures 4, 5, S4 and S5 use SDF structures, atom weights and fragment masks. Figures S1 and S2 show recorded fragmentation and generation examples.

For Figures 3 and S6, run `vmd -e Figure_3.tcl` or `vmd -e Figure_S6.tcl` from the matching directory. The scripts load the protein, native ligand and six selected candidates with the saved camera. The receptor uses NewCartoon. Edit the selections in Graphics → Representations, then render from VMD. SDF files preserve the ligand chemistry; MOL2 files supply the same atom order and bonds to VMD.

`figure_1_TOC/` contains four VMD TCL setups, transparent 2D notebooks for panels 2 and 4, an editable PowerPoint component for panel 3, and SVG layout components. Assemble these components in a graphics editor. Run the panel notebooks from their own directories. Install librsvg for transparent PNG conversion.

Figures 6 and S7 contain only the PDB/SDF input structures and a panel-order table. Open their prepared `figure.pdb` files in a separately installed LigPlot+ application. See the README in each directory for ligand selection and contact criteria.
