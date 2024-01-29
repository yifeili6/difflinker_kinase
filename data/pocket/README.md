# How to prepare Pockets dataset 

Download Binding KLIF:
```
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
unzip every_part_a.zip
unzip every_part_b.zip
```

Clean and split raw PL-complexes:
```
RAW_KLIF=../../data_docking/complex/klif_pdbs_wl
PROCESSED_KLIF=../../data_docking/complex/processed_klif_wl/
python -W ignore clean_and_split.py --in-dir $RAW_KLIF --proteins-dir $PROCESSED_KLIF/proteins --ligands-dir $PROCESSED_KLIF/ligands
```

Create fragments and conformers:
```
python -W ignore generate_fragmentation_and_conformers.py --in-ligands $PROCESSED_KLIF/ligands --out-fragmentations $PROCESSED_KLIF/generated_splits.csv --out-conformers $PROCESSED_KLIF/generated_conformers.sdf
```

Prepare dataset:
```
python -W ignore prepare_dataset.py --table $PROCESSED_KLIF/generated_splits.csv --sdf $PROCESSED_KLIF/generated_conformers.sdf --proteins $PROCESSED_KLIF/proteins --out-mol-sdf $PROCESSED_KLIF/KLIF_mol.sdf --out-frag-sdf $PROCESSED_KLIF/KLIF_frag.sdf --out-link-sdf $PROCESSED_KLIF/KLIF_link.sdf --out-pockets-pkl $PROCESSED_KLIF/KLIF_pockets.pkl --out-table $PROCESSED_KLIF/KLIF_table.csv
```

Final filtering and train/val/test split:
```
python -W ignore filter_and_train_test_split.py --mol-sdf $PROCESSED_KLIF/KLIF_mol.sdf --frag-sdf $PROCESSED_KLIF/KLIF_frag.sdf --link-sdf $PROCESSED_KLIF/KLIF_link.sdf --pockets-pkl $PROCESSED_KLIF/KLIF_pockets.pkl --table $PROCESSED_KLIF/KLIF_table.csv
```
