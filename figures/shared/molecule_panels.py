from pathlib import Path
import io, json, math, hashlib, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from structure_io import parse_sdf_bytes
DPI = 300
GREEN = (0.55, 0.8, 0.5)
MAGENTA = (0.85, 0.48, 0.79)

def molecule_state(molecule):
    return {'canonical_isomeric_smiles': Chem.MolToSmiles(molecule, isomericSmiles=True), 'all_properties_binary_sha256': hashlib.sha256(molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)).hexdigest(), 'coordinates': [hashlib.sha256(np.asarray(c.GetPositions(), dtype=np.float64).tobytes()).hexdigest() for c in molecule.GetConformers()], 'atom_count': molecule.GetNumAtoms(), 'bond_count': molecule.GetNumBonds()}

def weight_color(weight, scale):
    ratio = max(-1.0, min(1.0, float(weight) / float(scale))) if scale > 0 else 0.0
    if ratio >= 0:
        return (1.0, 1.0 - 0.7 * ratio, 1.0 - 0.7 * ratio)
    return (1.0 + 0.7 * ratio, 1.0 + 0.7 * ratio, 1.0)

def molecule_image(molecule, width, height, colors=None, radii=None):
    before = molecule_state(molecule)
    drawing = Chem.Mol(molecule)
    drawing.RemoveAllConformers()
    rdDepictor.Compute2DCoords(drawing, canonOrient=True, clearConfs=True)
    if Chem.MolToSmiles(drawing, isomericSmiles=True) != before['canonical_isomeric_smiles']:
        raise ValueError('Depiction changed molecular identity')
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.padding = 0.08
    options.addAtomIndices = False
    options.fixedFontSize = round(8.0 * DPI / 72)
    options.bondLineWidth = 2
    colors = colors or {}
    if not set(colors).issubset(range(molecule.GetNumAtoms())):
        raise ValueError('Atom colors exceed the molecular atom indices')
    drawer.DrawMolecule(drawing, highlightAtoms=sorted(colors), highlightBonds=[], highlightAtomColors=colors, highlightAtomRadii=radii or {})
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')
    if molecule_state(molecule) != before:
        raise ValueError('Depiction changed the source molecular record')
    return (image, before)

def image_axis(fig, box, molecule, colors=None, radii=None):
    width = round(box[2] * fig.get_figwidth() * DPI)
    height = round(box[3] * fig.get_figheight() * DPI)
    (image, state) = molecule_image(molecule, width, height, colors, radii)
    axis = fig.add_axes(box)
    axis.imshow(image, interpolation='none')
    axis.axis('off')
    return state

def load_molecule(path, parse_sdf_bytes, identifier):
    mols = parse_sdf_bytes(path.read_bytes(), label=identifier)
    if len(mols) != 1 or mols[0].GetProp('_Name') != identifier:
        raise ValueError(f'Candidate identity differs: {identifier}')
    return mols[0]

def read_json(path):
    return json.loads(path.read_text())
LAYOUT = json.loads((Path(__file__).resolve().parent / 'molecule_layout.json').read_text())
TEXTWIDTH = LAYOUT['textwidth_tex_points'] / LAYOUT['tex_points_per_inch']

def prepare():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})

def render_atom_map(data, target, metric):
    prepare()
    raw = Path(data)
    selection = list(csv.DictReader((raw / 'structural_selection.csv').open()))
    target_rows = [r for r in selection if r['target'] == target]
    assert [int(r['panel']) for r in target_rows] == list(range(1, 7))
    title = 'Tanimoto' if metric == 'morgan_similarity' else 'QED'
    basename = ('atomic_molgrid' if metric == 'morgan_similarity' else 'qed_molgrid') + ('_type2' if target == '4ag8' else '')
    fig = plt.figure(figsize=(TEXTWIDTH * LAYOUT['grid_fraction'], 5.75))
    panel_records = []
    for (index, row) in enumerate(target_rows):
        candidate = raw / f"candidate_{int(row['panel']):02d}"
        mol = load_molecule(candidate / 'candidate_aligned.sdf', parse_sdf_bytes, row['candidate_id'])
        mask = read_json(candidate / 'fragment_mask.json')
        weights = read_json(candidate / f'{metric}_weights.json')
        if weights['candidate_id'] != row['candidate_id'] or mask['candidate_id'] != row['candidate_id']:
            raise ValueError('Candidate identity differs between depiction inputs')
        retained = mask['retained_fragment_atom_indices']
        if retained != weights['masked_retained_fragment_atom_indices']:
            raise ValueError('Retained atom mask differs')
        if len(weights['raw_atom_weights']) != mol.GetNumAtoms():
            raise ValueError('Atom weight length differs')
        scale = float(weights['target_display_scale'])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid recorded display scale')
        colors = {i: weight_color(w, scale) for (i, w) in enumerate(weights['raw_atom_weights']) if i not in retained}
        radii = {i: 0.22 + 0.18 * min(1.0, abs(weights['raw_atom_weights'][i]) / scale) for i in colors}
        (x, y) = (0.01 + index % 2 * 0.5, 0.7 - index // 2 * 0.305)
        state = image_axis(fig, [x, y, 0.48, 0.265], mol, colors, radii)
        short_id = '-'.join(row['candidate_id'].split('-')[-2:])
        label = f"({chr(97 + index)}) {short_id}  |  {title} {weights['baseline_score']:.3f}"
        fig.text(x + 0.24, y - 0.018, label, ha='center', va='top', fontsize=8.5)
        panel_records.append({'figure': basename, 'panel': chr(97 + index), 'short_candidate_id': short_id, 'candidate_id': row['candidate_id'], 'baseline_score': weights['baseline_score'], 'weights': f"data/candidate_{int(row['panel']):02d}/{metric}_weights.json", 'recorded_display_scale': scale, 'source_molecule': state, 'colored_atom_indices': sorted(colors), 'masked_retained_atom_indices': retained})
    fig.text(0.5, 0.012, 'Red: positive weight    Blue: negative weight    Retained fragments: uncolored', ha='center', va='bottom', fontsize=8)
    return (fig, panel_records)

def render_mmpa(data):
    prepare()
    raw = Path(data)
    mmpa = read_json(raw / 'mmpa_5lqf.json')
    ctx = mmpa['context']
    fig = plt.figure(figsize=(TEXTWIDTH * LAYOUT['mmpa_fraction'], 2.1))
    mmpa_records = []
    for (i, (key, label)) in enumerate([('canonical_smiles', 'Native CDK1 ligand'), ('mmpa_fragments_with_dummies', 'Two retained fragments'), ('mmpa_linker_with_dummies', 'Removed linker')]):
        mol = Chem.MolFromSmiles(ctx[key])
        if mol is None:
            raise ValueError('MMPA SMILES cannot be parsed')
        state = image_axis(fig, [0.003 + i / 3, 0.25, 0.326, 0.7], mol)
        fig.text((i + 0.5) / 3, 0.135, label, ha='center', va='center', fontsize=8)
        mmpa_records.append({'panel': i + 1, 'label': label, 'source_molecule': state})
    fig.text(0.5, 0.025, '* = cut-bond attachment site; native ligand from PDB 5LQF', ha='center', fontsize=8)
    return (fig, mmpa_records)

def render_generation(data):
    prepare()
    raw = Path(data)
    generation = read_json(raw / 'generation_selection.json')
    examples = generation['examples']
    if generation['mode'] != 'shared_fragment_pair' or [r['linker_size'] for r in examples] != list(range(8, 14)):
        raise ValueError('This layout requires the recorded common fragment pair and six ordered linker sizes')
    if len({r['fragmentation_id'] for r in examples}) != 1:
        raise ValueError('Generation examples do not share a fragment pair')
    fig = plt.figure(figsize=(TEXTWIDTH * LAYOUT['generation_fraction'], 5.8))
    input_mol = Chem.MolFromSmiles(examples[0]['context']['mmpa_fragments_with_dummies'])
    input_state = image_axis(fig, [0.02, 0.72, 0.38, 0.25], input_mol, {i: GREEN for i in range(input_mol.GetNumAtoms())})
    fig.text(0.21, 0.982, 'Two retained fragments', ha='center', va='top', fontsize=9)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis('off')
    canvas.add_patch(FancyBboxPatch((0.52, 0.8), 0.27, 0.115, boxstyle='round,pad=.01', facecolor='#eaf0f7', edgecolor='#506784'))
    canvas.text(0.655, 0.874, 'DiffLinker', ha='center', fontsize=10, weight='bold')
    canvas.text(0.655, 0.826, '5LQF pocket', ha='center', fontsize=8.5)
    canvas.add_patch(FancyArrowPatch((0.405, 0.858), (0.5, 0.858), arrowstyle='-|>', mutation_scale=14, color='#506784'))
    canvas.add_patch(FancyArrowPatch((0.655, 0.785), (0.655, 0.724), arrowstyle='-|>', mutation_scale=14, color='#506784'))
    generation_records = []
    for (i, row) in enumerate(examples):
        cid = row['candidate_id']
        mol = load_molecule(raw / 'generation_examples' / f'{cid}.sdf', parse_sdf_bytes, cid)
        mask = read_json(raw / 'generation_examples' / f'{cid}.fragment_mask.json')
        (retained, linker) = (set(mask['retained_fragment_atom_indices']), set(mask['generated_linker_atom_indices']))
        if mask['candidate_id'] != cid or retained & linker or retained | linker != set(range(mol.GetNumAtoms())) or (len(linker) != row['linker_size']):
            raise ValueError('Generation example atom masks differ')
        colors = {j: GREEN if j in retained else MAGENTA for j in range(mol.GetNumAtoms())}
        (x, y) = (0.01 + i % 3 / 3, 0.397 - i // 3 * 0.31)
        state = image_axis(fig, [x, y, 0.317, 0.275], mol, colors)
        fig.text(x + 0.1585, y + 0.277, f"{row['linker_size']} linker atoms", ha='center', va='bottom', fontsize=8.5)
        short_id = '-'.join(cid.split('-')[-2:])
        generation_records.append({'candidate_id': cid, 'short_candidate_id': short_id, 'linker_size': row['linker_size'], 'source_molecule': state, 'retained_atom_indices': sorted(retained), 'generated_linker_atom_indices': sorted(linker)})
    fig.text(0.5, 0.025, 'Green: retained fragments    Magenta: generated linker    *: attachment site', ha='center', fontsize=8)
    return (fig, generation_records)
