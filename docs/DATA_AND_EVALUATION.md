# Data and evaluation

## Structural sources

The workflow uses KLIFS kinase–ligand structures. The data manifest records the downloaded snapshot, source structure identifiers, protein and ligand paths, acquisition metadata and file checksums. Each processed context links to the corresponding native ligand and receptor.

Ligands are represented by heavy-atom coordinates and chemical graphs. Protein pocket atoms are selected from residues within 10 Å of the native ligand. The dataset configuration defines water removal, element selection, molecule-size criteria and the structural quality thresholds for the fine-tuning corpus. A fixed target registry specifies the nine held-out target PDB complexes.

## Fragmentation and splits

Matched Molecular Pair Analysis makes two cut bonds, yielding two retained fragments and the connecting linker. The fragment table records the cut-bond pattern, fragment and linker SMILES, atom counts and the two attachment indices. These indices refer to the ordered retained-fragment atoms used by the model.

Train, validation and test assignments are grouped by PDB identifier, canonical ligand SMILES, InChIKey and Murcko scaffold. The split manifest records each context's group and assigned partition. The data-derived `provenance/population_summary.json` gives the number of emitted structures, PDB entries, canonical isomeric molecular identities, kinases, scaffolds and fragmentation rows in each partition. The nine target PDB complexes listed in the generation configuration are assigned to the test partition.

## Chemical serialization

Native parent molecules and generated molecules that pass sanitization are exported as kekulized SDF records with the chemistry properties described below. Chemistry sidecars and SDF properties define molecular identity; SDF and NPZ records provide poses. The supplied `kinase_gap.structural_bundle.parse_sdf_bytes` loader validates the SDF against its embedded RDKit molecule and restores exact atom properties, hydrogen representation and full-precision coordinates. Retained fragment atoms preserve the source stereochemistry, while newly generated atoms receive assignments from their sampled three-dimensional poses. Readback checks compare canonical isomeric SMILES, atom ordering and coordinates using the supplied loader. Fragment atom properties include formal charge, explicit hydrogens and aromaticity. Assembly retains the source atom properties and records the resulting linker attachments against the supplied anchor indices.

Generated-molecule evaluation uses `scripts/run_authoritative_evaluation.py` with `configs/evaluation_authoritative_v2.yml`. The reader verifies the embedded molecule against the SDF, the generation identity and the saved full-precision coordinate array. It supplies that molecular graph and pose to PoseBusters and the subsequent filters. The evaluation contract records the reader source and the verified input contract. Ground-truth molecules are loaded from their parent-molecule chemistry sidecars.

## Filtering

Molecule-level evaluation records three cumulative stages:

1. `after_posebusters`: passage through the recorded PoseBusters molecular geometry checks.
2. `after_lipinski`: passage through the first stage and the four recorded Lipinski criteria.
3. `after_2d` or `final_pass`: passage through the preceding stages and the MOSES molecular filters.

The PoseBusters configuration records the checks and thresholds actually used. The Lipinski implementation records hydrogen-bond donors, acceptors, molecular weight and logP. The molecular filters record their PAINS matches and pass status.

A cumulative survival rate divides the stage count by the original number of attempts in the cohort. A conditional rate divides it by the preceding stage's count. Ground-truth survival uses the number of evaluated GT fragmentation rows as its denominator. The raw funnel table records both rates and their denominators.

## Molecular metrics

Novelty compares generated canonical SMILES with the training reference. Uniqueness compares the number of distinct canonical SMILES with the stated generated population. Internal diversity uses pairwise fingerprint similarity. FCD, nearest-neighbour similarity and fragment similarity compare the generated cohort with the reference named in the metric table. Wasserstein distances compare the distributions of logP, SA score, QED and molecular weight.

Metric comparisons retain final-survivor rows with their original multiplicity; the full-molecule reference set contains unique normalized identities. The metric definitions file names each reference set and population. Read `generated_count`, `generated_unique_count`, `reference_count` and `train_count` together with a metric value. Conditional validity among filtered molecules and validity among all attempted samples are separate quantities.

## Figure populations

Generated property plots use final survivors and keep one normalized canonical isomeric identity within each linker-size cohort. Overall generated-versus-GT comparisons deduplicate the generated identities across all sizes. The ground-truth property plot uses final GT survivors and keeps one record per canonical SMILES. The exact rows are supplied with each figure. Discrete-property bars show the mean and sample standard deviation; continuous-property plots show the empirical distribution and median.

Structural examples use the deterministic selection recorded in the selection manifest. The manifest records chemical eligibility, ranking and fingerprint-diversity settings. A rigid protein alignment applies the same rotation and translation to the source receptor, native ligand and generated ligand.

Morgan similarity maps use the recorded fingerprint settings and show atom perturbations relative to the native ligand. QED maps show the original score minus the score after deleting one atom and its incident bonds, sanitizing the result and retaining its components. The saved weight arrays give both raw values and fragment-masked display values.

Molecular deduplication and novelty use isomeric SMILES parsed and canonicalized with the recorded RDKit version. The source SMILES strings and candidate identifiers are retained alongside the normalized identities and normalization status. This normalization preserves stereochemical distinctions.
