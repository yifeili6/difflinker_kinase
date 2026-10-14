"""Isomeric molecular identity for statistics; source SMILES remain available."""
from functools import lru_cache
from rdkit import Chem


IDENTITY_FIELDS = ("normalized_identity", "identity_normalization_status")


@lru_cache(maxsize=65536)
def normalize_smiles_identity(smiles):
    text = str(smiles).strip()
    if not text:
        raise ValueError("Molecular identity requires a nonempty SMILES")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"Cannot normalize invalid SMILES: {text!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def with_normalized_identity(row, *, require_valid=False):
    """Add identity/status fields without replacing the input SMILES or row ID."""
    result = dict(row)
    text = str(result.get("canonical_smiles", "")).strip()
    try:
        normalized = normalize_smiles_identity(text)
    except ValueError as exc:
        if require_valid:
            identifier = result.get("candidate_id", result.get("gt_id", "unknown"))
            raise ValueError(f"Final survivor {identifier!r}: {exc}") from exc
        result["normalized_identity"] = ""
        result["identity_normalization_status"] = "empty" if not text else "invalid"
    else:
        result["normalized_identity"] = normalized
        result["identity_normalization_status"] = "unchanged" if text == normalized else "normalized"
    return result
