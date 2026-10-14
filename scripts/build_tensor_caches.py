"""Build portable CPU tensor inputs for every prepared split before GPU jobs."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"vendor/DiffLinker")]
from src.datasets import MOADDataset

result = {}
for split in ("train", "val", "test"):
    dataset = MOADDataset(data_path=str(ROOT/"data/processed/klifs_v1"), prefix=f"KLIFS_{split}.full", device="cpu")
    assert all(int((row["anchors"] == 1).sum()) == 2 for row in dataset)
    assert all(bool((row["anchors"] <= row["fragment_only_mask"]).all()) for row in dataset)
    assert all(row["positions"].device.type == "cpu" for row in dataset)
    result[split] = len(dataset)
    del dataset
with (ROOT/"provenance/tensor_cache_counts.json").open("x") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result))
