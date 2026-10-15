"""Preserve existing double-bond stereo through native SDF serialization.

The frozen exporter remains the first path. Its exact stereo-restoration failure
can be retried with the same molecule's E/Z assignments encoded as neighboring
single-bond directions. Molecular identity, graph, atom order and coordinates
must remain exact; all original writer/readback checks still execute.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from rdkit import Chem

FROZEN_CHEMISTRY_SHA256 = "b9aac220cdd62e1768f91a86ebbc6227e305e323fac357d7af07abc8836234b8"
RESTORATION_ERROR = "Native stereochemistry metadata did not restore source identity"
_ORIGINAL = None
_EVENTS = []


def _identity(molecule):
    from kinase_gap.fragmentation import canonical_isomeric_identity
    return canonical_isomeric_identity(Chem.Mol(molecule))


def _graph(molecule):
    atoms = tuple((a.GetIdx(), a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
                   a.GetNumRadicalElectrons(), a.GetNumExplicitHs(), a.GetNumImplicitHs(),
                   a.GetNoImplicit(), a.GetIsAromatic(), int(a.GetChiralTag()),
                   a.GetAtomMapNum(), tuple(n.GetIdx() for n in a.GetNeighbors()))
                  for a in molecule.GetAtoms())
    bonds = tuple((b.GetIdx(), b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()),
                   b.GetIsAromatic(), b.GetIsConjugated(), int(b.GetStereo()),
                   tuple(b.GetStereoAtoms())) for b in molecule.GetBonds())
    return atoms, bonds


def _positions(molecule):
    return tuple((c.GetId(), c.Is3D(), np.array(c.GetPositions(), copy=True))
                 for c in molecule.GetConformers())


def _assert_unchanged(molecule, graph, positions, identity, *, stage):
    if _graph(molecule) != graph:
        raise ValueError(f"Stereo representation changed molecular graph or atom order: {stage}")
    observed = _positions(molecule)
    if len(observed) != len(positions) or any(
            a[0:2] != b[0:2] or not np.array_equal(a[2], b[2])
            for a, b in zip(observed, positions)):
        raise ValueError(f"Stereo representation changed full-precision coordinates: {stage}")
    if _identity(molecule) != identity:
        raise ValueError(f"Stereo representation changed canonical isomeric identity: {stage}")


def checked_native_mol_block(molecule):
    if _ORIGINAL is None:
        raise RuntimeError("Serialization extension has not been installed")
    try:
        return _ORIGINAL(molecule)
    except ValueError as error:
        if str(error) != RESTORATION_ERROR or not molecule.HasProp("smiles"):
            raise
        # A generation row supplies its independently computed pre-export identity.
        csv_molecule = Chem.MolFromSmiles(molecule.GetProp("smiles"))
        if csv_molecule is None:
            raise ValueError("Stereo representation requires a valid generation-row SMILES") from error
        expected = _identity(molecule)
        if expected != _identity(csv_molecule):
            raise ValueError("Stereo representation input differs from generation-row identity") from error
        graph, positions = _graph(molecule), _positions(molecule)
        candidate = Chem.Mol(molecule)
        directions_before = [int(b.GetBondDir()) for b in candidate.GetBonds()]
        Chem.SetDoubleBondNeighborDirections(candidate)
        _assert_unchanged(candidate, graph, positions, expected, stage="direction encoding")
        directions_after = [int(b.GetBondDir()) for b in candidate.GetBonds()]
        if directions_before == directions_after:
            raise error
        # Reuse all frozen constitutional/isomeric/order/H/coordinate checks.
        block, result = _ORIGINAL(candidate)
        _assert_unchanged(result, graph, positions, expected, stage="frozen export")
        # Frozen evaluation and structural readers sanitize the embedded graph again.
        resanitized = Chem.Mol(result)
        Chem.SanitizeMol(resanitized)
        if _identity(resanitized) != expected:
            raise ValueError("Stereo representation identity is not stable under reader sanitization")
        _EVENTS.append({
            "candidate_id": molecule.GetProp("_Name") if molecule.HasProp("_Name") else "",
            "trigger": RESTORATION_ERROR,
            "canonical_isomeric_identity": expected,
            "changed_bond_directions": [
                {"bond_index": index, "before": before, "after": after}
                for index, (before, after) in enumerate(zip(directions_before, directions_after))
                if before != after],
        })
        return block, result


def install():
    """Install the writer-only extension in this process; frozen readers stay intact."""
    global _ORIGINAL
    import kinase_gap.chemistry_io as chemistry_io
    observed = hashlib.sha256(Path(chemistry_io.__file__).read_bytes()).hexdigest()
    if observed != FROZEN_CHEMISTRY_SHA256:
        raise ValueError("Frozen chemistry source checksum differs from the reviewed exporter")
    if chemistry_io.checked_native_mol_block is checked_native_mol_block:
        return
    if _ORIGINAL is not None:
        raise RuntimeError("Another exporter replaced the installed serialization extension")
    _ORIGINAL = chemistry_io.checked_native_mol_block
    chemistry_io.checked_native_mol_block = checked_native_mol_block


def fallback_events():
    """Return immutable-value event records for the calling job's provenance."""
    import copy
    return copy.deepcopy(_EVENTS)
