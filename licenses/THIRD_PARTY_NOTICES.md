# Third party notices

## DiffLinker source code

DiffLinker source code is distributed under the MIT License. Copyright © 2022 Ilia Igashov, Hannes Stärk and Clément Vignac. The complete notice is reproduced in `DiffLinker-MIT.txt`.

Source project: [github.com/igashov/DiffLinker](https://github.com/igashov/DiffLinker).

Research article: Igashov and colleagues, *Equivariant 3D-conditional diffusion model for molecular linker design*, Nature Machine Intelligence (2024), [doi:10.1038/s42256-024-00815-9](https://doi.org/10.1038/s42256-024-00815-9).

## DiffLinker pretrained model

The pretrained full-pocket anchor-conditioned model was obtained from Ilia Igashov, *DiffLinker Models*, published 11 April 2024, [doi:10.5281/zenodo.10988017](https://doi.org/10.5281/zenodo.10988017). That record licenses the model under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The original upstream filename is `pockets_difflinker_full_fc_pdb_excluded.ckpt`. Its SHA-256 is `94dad8a40b547cd29ec4fa942c90ad3e60cc2d3c872afa29a109dbca5b3f135e`. The archived pretrained checkpoint is identified by this checksum. The kinase model is derived by fine-tuning this checkpoint on the KLIFS training split; its own checksum and training provenance are recorded in `provenance/run_manifest.json` and the associated HPO selection record.

## Structural data and analysis software

KLIFS supplies the kinase structure annotations and structural inputs; source identifiers and snapshot provenance accompany the data. The [KLIFS FAQ](https://klifs.net/faq.php) describes its data as freely available/open for academic and industrial use. KLIFS requests the 2021 database paper as its primary citation, with the 2014 or 2019 papers cited for the corresponding database features. These references are included in [references.bib](../references.bib).

Protein structure identifiers refer to their deposited PDB entries. The [RCSB PDB usage policy](https://www.rcsb.org/pages/usage-policy) states that data files in the PDB archive are available under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). This attribution applies to the deposited PDB archive files. KLIFS data and annotations retain the source attribution described above. Cite the original authors of each deposited structure when using its structural data.

RDKit supplies molecular graph processing and descriptors. PoseBusters supplies the configured molecular geometry checks. MOSES supplies molecular filtering and distribution metrics. PyMOL supplies three-dimensional molecular rendering. LigPlot+ and HBPLUS supply protein–ligand interaction diagrams when run with the recorded configurations. Their versions and citations are recorded with the corresponding analysis products.

[references.bib](../references.bib) includes the DiffLinker, KLIFS, PoseBusters, MOSES, RDKit, synthetic accessibility, QED, LigPlot and HBPLUS references. The [RDKit documentation](https://www.rdkit.org/docs/Overview.html#citing-the-rdkit) gives its software citation. LigPlot+ and HBPLUS are obtained through their [official distribution](https://www.ebi.ac.uk/thornton-srv/software/LigPlus/) and [HBPLUS page](https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/); the structural analysis files record the executable hashes used for the diagrams.
