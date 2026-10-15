from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .identity import with_normalized_identity
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SELECTION_LABEL = 'prospective_chemistry_only'
FRAGMENT_ELIGIBILITY_FIELDS = ('anchor_attachment_pass', 'retained_stereo_pass', 'retained_nonanchor_explicit_h_match', 'retained_nonanchor_total_h_match')
DEFAULT_LINKER_SIZES = tuple(range(8, 14))
DEFAULT_TARGETS = ('5lqf', '4ag8')
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_USE_CHIRALITY = True

@dataclass(frozen=True)
class PropertySpec:
    name: str
    label: str
    aliases: Tuple[str, ...]
    discrete: bool
PROPERTY_SPECS: Tuple[PropertySpec, ...] = (PropertySpec('strict_rotatable_bonds', 'Strict rotatable bonds', ('strict_rotatable_bonds', 'num_rotatable_bonds_strict', 'num_rotatable_bonds', 'num_rot_bonds', 'rot_bonds', 'rotatable_bond_count'), True), PropertySpec('num_rings', 'SymmSSSR rings', ('num_rings', 'ring_count', 'rings'), True), PropertySpec('num_fused_rings', 'Fused ring systems', ('num_fused_rings', 'fused_ring_systems', 'fused_rings', 'fused_ring_count'), True), PropertySpec('num_hetero_rings', 'Heteroaromatic rings', ('num_hetero_rings', 'hetero_rings', 'heteroaromatic_ring_count'), True), PropertySpec('num_aromatic_rings', 'Aromatic rings', ('num_aromatic_rings', 'aromatic_ring_count', 'aromatic_rings'), True), PropertySpec('logp', 'MolLogP', ('logp', 'mol_logp', 'mollogp'), False), PropertySpec('qed', 'QED', ('qed',), False), PropertySpec('sa', 'Synthetic accessibility', ('sa', 'sa_score', 'sascore'), False), PropertySpec('molecular_weight', 'Molecular weight (Da)', ('molecular_weight', 'mol_weight', 'molwt', 'mw', 'weight'), False))
PROPERTY_NAMES = tuple((spec.name for spec in PROPERTY_SPECS))
PROPERTY_BY_NAME = {spec.name: spec for spec in PROPERTY_SPECS}
EVALUATION_ALIASES: Mapping[str, Tuple[str, ...]] = {'candidate_id': ('candidate_id', 'sample_id', 'molecule_id'), 'pdb': ('pdb', 'target_pdb', 'pdb_id', 'target'), 'linker_size': ('linker_size', 'sampled_linker_size', 'sampled_atoms'), 'final_pass': ('final_pass', 'pass_final', 'cumulative_final_pass', 'moses_pass_cumulative', 'after_2d'), 'connected': ('connected', 'is_connected', 'all_atoms_connected'), 'novel': ('novel', 'novel_train', 'is_novel', 'novelty_pass'), 'canonical_smiles': ('canonical_smiles', 'standardized_smiles', 'smiles'), **{field: (field,) for field in FRAGMENT_ELIGIBILITY_FIELDS}}
FUNNEL_ALIASES: Mapping[str, Tuple[str, ...]] = {'scope': ('scope',), 'stage': ('stage', 'filter_stage', 'gate'), 'stage_order': ('stage_order', 'stage_index', 'gate_order'), 'group': ('group', 'population', 'stratum', 'group_value'), 'source': ('source', 'dataset', 'kind'), 'pdb': ('pdb', 'target_pdb', 'pdb_id', 'target'), 'linker_size': ('linker_size', 'sampled_linker_size', 'sampled_atoms'), 'input_n': ('input_n', 'initial_n', 'denominator_n', 'requested_n', 'total', 'input_count'), 'retained_n': ('retained_n', 'n_retained', 'pass_n', 'count', 'passed'), 'conditional_pct': ('conditional_pct', 'conditional_percent', 'stage_pass_pct', 'conditional_rate'), 'cumulative_pct': ('cumulative_pct', 'cumulative_percent', 'survival_pct', 'pass_pct', 'rate', 'cumulative_rate')}

def _normalise_name(value: str) -> str:
    return re.sub('[^a-z0-9]+', '', str(value).lower())

def _find_alias(frame: pd.DataFrame, canonical: str, aliases: Sequence[str], *, required: bool) -> Optional[str]:
    normalised: Dict[str, List[str]] = {}
    for column in frame.columns:
        normalised.setdefault(_normalise_name(column), []).append(str(column))
    matches: List[str] = []
    for alias in (canonical, *aliases):
        matches.extend(normalised.get(_normalise_name(alias), []))
    matches = list(dict.fromkeys(matches))
    if canonical == 'final_pass' and len(matches) == 2 and ({_normalise_name(value) for value in matches} == {'finalpass', 'after2d'}):
        final_column = next((value for value in matches if _normalise_name(value) == 'finalpass'))
        after_column = next((value for value in matches if _normalise_name(value) == 'after2d'))
        final_values = _coerce_bool_series(frame[final_column], name=final_column)
        after_values = _coerce_bool_series(frame[after_column], name=after_column)
        if not final_values.equals(after_values):
            raise ValueError(f'Conflicting equivalent columns {final_column!r} and {after_column!r}')
        return final_column
    if canonical in frame.columns:
        duplicate_canonical_spellings = [value for value in matches if value != canonical and _normalise_name(value) == _normalise_name(canonical)]
        if duplicate_canonical_spellings:
            raise ValueError(f'Ambiguous columns for {canonical!r}: {[canonical, *duplicate_canonical_spellings]}. Provide one canonical field only.')
        return canonical
    if len(matches) > 1:
        raise ValueError(f'Ambiguous columns for {canonical!r}: {matches}. Provide one canonical field only.')
    if not matches:
        if required:
            raise ValueError(f'Missing required column {canonical!r}; accepted aliases are {sorted(set((canonical, *aliases)))}')
        return None
    return matches[0]

def _canonicalise_columns(frame: pd.DataFrame, aliases: Mapping[str, Sequence[str]], *, required: Iterable[str]=()) -> pd.DataFrame:
    required_set = set(required)
    renames: Dict[str, str] = {}
    for (canonical, choices) in aliases.items():
        match = _find_alias(frame, canonical, choices, required=canonical in required_set)
        if match is not None and match != canonical:
            renames[match] = canonical
    return frame.rename(columns=renames).copy()

def _canonicalise_properties(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {spec.name: spec.aliases for spec in PROPERTY_SPECS}
    result = _canonicalise_columns(frame, aliases, required=PROPERTY_NAMES)
    for name in PROPERTY_NAMES:
        original = result[name]
        converted = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = original.loc[invalid].astype(str).head(3).tolist()
            raise ValueError(f'Non-numeric values in {name}: {examples}')
        result[name] = converted
    return result

def _coerce_bool_series(series: pd.Series, *, name: str) -> pd.Series:
    true_values = {True, 1, '1', 'true', 't', 'yes', 'y'}
    false_values = {False, 0, '0', 'false', 'f', 'no', 'n'}

    def convert(value: Any) -> bool:
        if pd.isna(value):
            raise ValueError(f'Missing boolean value in {name}')
        candidate = value
        if isinstance(candidate, str):
            candidate = candidate.strip().lower()
        if candidate in true_values:
            return True
        if candidate in false_values:
            return False
        raise ValueError(f'Invalid boolean value in {name}: {value!r}')
    return series.map(convert).astype(bool)

def _numeric_column(frame: pd.DataFrame, name: str, *, integer: bool=False) -> pd.Series:
    converted = pd.to_numeric(frame[name], errors='coerce')
    invalid = frame[name].notna() & converted.isna()
    if invalid.any():
        examples = frame.loc[invalid, name].astype(str).head(3).tolist()
        raise ValueError(f'Non-numeric values in {name}: {examples}')
    if integer:
        finite = converted.dropna()
        if not np.all(np.equal(finite, np.floor(finite))):
            raise ValueError(f'{name} must contain integers')
        return converted.astype('Int64')
    return converted.astype(float)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f'CSV contains no rows: {path}')
    return frame

def _population_frame(frame: pd.DataFrame, *, source: str, population: str, deduplicate_smiles: bool) -> pd.DataFrame:
    required = ['linker_size'] if source == 'generated' else []
    if population == 'final_pass':
        required.append('final_pass')
    result = _canonicalise_columns(frame, EVALUATION_ALIASES, required=required)
    result = _canonicalise_properties(result)
    if population == 'final_pass':
        result['final_pass'] = _coerce_bool_series(result['final_pass'], name='final_pass')
        result = result.loc[result['final_pass']].copy()
    elif population != 'all':
        raise ValueError("population must be 'final_pass' or 'all'")
    if source == 'generated':
        result['linker_size'] = _numeric_column(result, 'linker_size', integer=True)
        if result['linker_size'].isna().any():
            raise ValueError('Generated rows must all have linker_size')
    if deduplicate_smiles:
        result = _canonicalise_columns(result, EVALUATION_ALIASES, required=('canonical_smiles',))
        smiles = result['canonical_smiles'].fillna('').astype(str).str.strip()
        if (smiles == '').any():
            raise ValueError('Cannot deduplicate blank canonical_smiles')
        normalized = [with_normalized_identity({'canonical_smiles': value}, require_valid=True) for value in smiles]
        result['normalized_identity'] = [row['normalized_identity'] for row in normalized]
        result['identity_normalization_status'] = [row['identity_normalization_status'] for row in normalized]
        keys = ['linker_size', 'normalized_identity'] if source == 'generated' else ['normalized_identity']
        result = result.drop_duplicates(keys, keep='first').copy()
    result.insert(0, 'report_source', source)
    return result.reset_index(drop=True)

def _summary_rows(frame: pd.DataFrame, *, source: str, group: str, linker_size: Optional[int], population: str, deduplicate_smiles: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in PROPERTY_SPECS:
        values = frame[spec.name].dropna().astype(float)
        n = int(len(values))
        if not n:
            statistics = {'mean': math.nan, 'sd': math.nan, 'median': math.nan, 'q1': math.nan, 'q3': math.nan, 'min': math.nan, 'max': math.nan}
        else:
            statistics = {'mean': float(values.mean()), 'sd': float(values.std(ddof=1)) if n > 1 else 0.0, 'median': float(values.median()), 'q1': float(values.quantile(0.25, interpolation='linear')), 'q3': float(values.quantile(0.75, interpolation='linear')), 'min': float(values.min()), 'max': float(values.max())}
        rows.append({'source': source, 'group': group, 'linker_size': linker_size, 'population': population, 'deduplicated_canonical_smiles': bool(deduplicate_smiles), 'population_n': int(len(frame)), 'property': spec.name, 'property_label': spec.label, 'n': n, **statistics})
    return rows

def summarize_properties(candidate_evaluation: pd.DataFrame, gt_evaluation: pd.DataFrame, *, population: str='final_pass', deduplicate_smiles: bool=False) -> pd.DataFrame:
    generated = _population_frame(candidate_evaluation, source='generated', population=population, deduplicate_smiles=deduplicate_smiles)
    ground_truth = _population_frame(gt_evaluation, source='ground_truth', population=population, deduplicate_smiles=deduplicate_smiles)
    generated_overall = generated.drop_duplicates('normalized_identity', keep='first') if deduplicate_smiles else generated
    rows = _summary_rows(generated_overall, source='generated', group='generated_overall', linker_size=None, population=population, deduplicate_smiles=deduplicate_smiles)
    for size in sorted((int(value) for value in generated['linker_size'].dropna().unique())):
        subset = generated.loc[generated['linker_size'] == size]
        rows.extend(_summary_rows(subset, source='generated', group=f'generated_linker_{size}', linker_size=size, population=population, deduplicate_smiles=deduplicate_smiles))
    rows.extend(_summary_rows(ground_truth, source='ground_truth', group='ground_truth', linker_size=None, population=population, deduplicate_smiles=deduplicate_smiles))
    return pd.DataFrame.from_records(rows)

def _lazy_pyplot():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    return plt

def _save_figure(fig: Any, output_base: Path) -> Dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    for (suffix, kwargs) in (('.png', {'dpi': 300}), ('.pdf', {})):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches='tight', **kwargs)
        outputs[path.name] = sha256_file(path)
    return outputs

def _plot_property_groups(groups: Sequence[Tuple[str, pd.DataFrame]], *, output_base: Path, title: str, population_label: str) -> Dict[str, str]:
    if not groups:
        raise ValueError('At least one property group is required')
    plt = _lazy_pyplot()
    (fig, axes) = plt.subplots(3, 3, figsize=(12.0, 11.5))
    colors = plt.get_cmap('tab10')(np.linspace(0.05, 0.95, max(2, len(groups))))
    labels = [label for (label, _) in groups]
    x_positions = np.arange(len(groups), dtype=float)
    for (axis, spec) in zip(axes.flat, PROPERTY_SPECS):
        arrays = [frame[spec.name].dropna().astype(float).to_numpy() for (_, frame) in groups]
        if spec.discrete:
            means = np.asarray([float(values.mean()) if len(values) else math.nan for values in arrays])
            deviations = np.asarray([float(values.std(ddof=1)) if len(values) > 1 else 0.0 for values in arrays])
            axis.bar(x_positions, means, yerr=deviations, capsize=3, color=colors[:len(groups)], edgecolor='black', linewidth=0.4)
            axis.set_ylabel('Mean count ± sample SD')
        else:
            present = [(position, values, color) for (position, values, color) in zip(x_positions, arrays, colors) if len(values)]
            if present:
                violin = axis.violinplot([values for (_, values, _) in present], positions=[position for (position, _, _) in present], widths=0.8, showmeans=False, showmedians=True, showextrema=True)
                for (body, (_, _, color)) in zip(violin['bodies'], present):
                    body.set_facecolor(color)
                    body.set_edgecolor('black')
                    body.set_alpha(0.75)
            axis.set_ylabel('Value')
        axis.set_title(spec.label, fontsize=10.5, fontweight='bold')
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        axis.grid(axis='y', alpha=0.2)
        for (position, values) in zip(x_positions, arrays):
            axis.text(position, 0.01, f'n={len(values)}', transform=axis.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.5, rotation=90, alpha=0.75)
    fig.suptitle(f'{title}\nPopulation: {population_label}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs

def plot_property_distributions(candidate_evaluation: pd.DataFrame, gt_evaluation: pd.DataFrame, *, output_dir: Path, population: str='final_pass', deduplicate_smiles: bool=False) -> Dict[str, str]:
    generated = _population_frame(candidate_evaluation, source='generated', population=population, deduplicate_smiles=deduplicate_smiles)
    ground_truth = _population_frame(gt_evaluation, source='ground_truth', population=population, deduplicate_smiles=deduplicate_smiles)
    dedupe_label = 'yes' if deduplicate_smiles else 'no'
    population_label = f'{population}; canonical-SMILES deduplication: {dedupe_label}'
    by_linker = [(str(size), generated.loc[generated['linker_size'] == size]) for size in sorted((int(value) for value in generated['linker_size'].unique()))]
    outputs = _plot_property_groups(by_linker, output_base=output_dir / 'property_distributions_by_linker', title='Generated kinase-linker candidates by sampled linker size', population_label=population_label)
    generated_overall = generated.drop_duplicates('normalized_identity', keep='first') if deduplicate_smiles else generated
    comparison = _plot_property_groups([('Generated', generated_overall), ('GT', ground_truth)], output_base=output_dir / 'property_distributions_generated_vs_gt', title='Generated candidates compared with KLIFS ground truth', population_label=population_label)
    outputs.update(comparison)
    return outputs

def _percentage_values(series: pd.Series, *, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    invalid = series.notna() & values.isna()
    if invalid.any():
        raise ValueError(f'Non-numeric percentage values in {name}')
    finite = values.dropna()
    if not finite.empty and finite.abs().max() <= 1.0 + 1e-12:
        values = values * 100.0
    if ((values.dropna() < -1e-09) | (values.dropna() > 100.0 + 1e-09)).any():
        raise ValueError(f'{name} must be a fraction or percentage in [0, 100]')
    return values.astype(float)

def canonicalize_funnel(frame: pd.DataFrame) -> pd.DataFrame:
    result = _canonicalise_columns(frame, FUNNEL_ALIASES, required=('stage', 'retained_n', 'cumulative_pct'))
    result['stage'] = result['stage'].astype(str).str.strip()
    if (result['stage'] == '').any():
        raise ValueError('Funnel stage may not be blank')
    result['retained_n'] = _numeric_column(result, 'retained_n')
    result['cumulative_pct'] = _percentage_values(result['cumulative_pct'], name='cumulative_pct')
    if 'conditional_pct' in result:
        result['conditional_pct'] = _percentage_values(result['conditional_pct'], name='conditional_pct')
    if 'input_n' in result:
        result['input_n'] = _numeric_column(result, 'input_n')
    if 'stage_order' in result:
        result['stage_order'] = _numeric_column(result, 'stage_order')
    if 'linker_size' in result:
        result['linker_size'] = _numeric_column(result, 'linker_size', integer=True)
    return result

def _funnel_group_labels(frame: pd.DataFrame) -> pd.Series:
    columns = {}
    for name in ('group', 'scope', 'source', 'pdb'):
        columns[name] = frame[name].fillna('').astype(str).str.strip() if name in frame else pd.Series('', index=frame.index)
    sizes = frame['linker_size'].astype('Int64').astype(str) if 'linker_size' in frame else pd.Series('', index=frame.index)
    labels = []
    for (group, scope, source, pdb, size) in zip(columns['group'], columns['scope'], columns['source'], columns['pdb'], sizes):
        generic_group = _normalise_name(group) in {'', 'overall', 'linkersize', 'targetlinkersize'}
        has_size = size not in {'', '<NA>'}
        scope_key = _normalise_name(scope)
        if scope_key == 'linkersize' and has_size:
            label = f'linker {size}'
        elif scope_key == 'overall':
            label = 'overall'
        elif group and (not generic_group):
            label = group
        elif pdb and has_size:
            label = f'{pdb.lower()}, linker {size}'
        elif has_size:
            label = f'linker {size}'
        elif group:
            label = group
        elif source:
            label = source
        elif scope:
            label = scope
        else:
            label = 'overall'
        labels.append(label)
    return pd.Series(labels, index=frame.index)

def _stage_key(value: Any) -> str:
    key = _normalise_name(str(value))
    if key in {'2d', 'twod', 'after2d', 'final', 'finalpass', 'moses', 'mosesfilter', 'mosesfilterpass'}:
        return 'finalpass'
    return key

def plot_funnel(funnel: pd.DataFrame, *, output_base: Path) -> Dict[str, str]:
    data = canonicalize_funnel(funnel)
    if 'scope' in data:
        scope = data['scope'].fillna('').map(_normalise_name)
        preferred = data.loc[scope.isin({'overall', 'linkersize', 'groundtruth', 'gt', 'reference'})]
        if not preferred.empty:
            data = preferred
    if 'pdb' in data:
        target = data['pdb'].fillna('').astype(str).str.strip()
        aggregate = data.loc[target == '']
        if not aggregate.empty:
            data = aggregate
    data = data.copy()
    data['_report_group'] = _funnel_group_labels(data)
    if 'stage_order' in data:
        stage_table = data.loc[:, ['stage', 'stage_order']].drop_duplicates().sort_values(['stage_order', 'stage'])
        if stage_table['stage'].duplicated().any():
            raise ValueError('One funnel stage has conflicting stage_order values')
        stages = stage_table['stage'].tolist()
    else:
        stages = list(dict.fromkeys(data['stage'].tolist()))
    plt = _lazy_pyplot()
    (fig, axis) = plt.subplots(figsize=(9.0, 5.5))
    for (label, group) in data.groupby('_report_group', sort=True):
        if group['stage'].duplicated().any():
            raise ValueError(f'Duplicate funnel stage rows for group {label!r}')
        indexed = group.set_index('stage')
        values = [indexed.at[stage, 'cumulative_pct'] if stage in indexed.index else np.nan for stage in stages]
        axis.plot(stages, values, marker='o', linewidth=1.5, label=label)
    axis.set_ylim(0, 103)
    axis.set_ylabel('Cumulative survival (%)')
    axis.set_xlabel('Sequential filter stage')
    axis.set_title('Sequential molecule-filter survival\nPercentages use the original population denominator')
    axis.grid(axis='y', alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs

def plot_target_size_heatmap(funnel: pd.DataFrame, *, output_base: Path, stage: str='final_pass') -> Dict[str, str]:
    data = canonicalize_funnel(funnel)
    for column in ('pdb', 'linker_size'):
        if column not in data:
            raise ValueError(f'Target heatmap requires funnel column {column!r}')
    requested_stage = _stage_key(stage)
    stage_rows = data.loc[data['stage'].map(_stage_key) == requested_stage].copy()
    if 'source' in stage_rows:
        source = stage_rows['source'].fillna('').astype(str).str.lower()
        generated = stage_rows.loc[source.str.contains('generated')]
        if not generated.empty:
            stage_rows = generated
    stage_rows['pdb'] = stage_rows['pdb'].fillna('').astype(str).str.lower().str.strip()
    stage_rows = stage_rows.loc[stage_rows['pdb'] != '']
    stage_rows = stage_rows.loc[stage_rows['linker_size'].notna()]
    if stage_rows.empty:
        raise ValueError(f'No target/linker rows found for funnel stage {stage!r}')
    keys = ['pdb', 'linker_size']
    if stage_rows.duplicated(keys).any():
        if 'input_n' not in stage_rows:
            raise ValueError('Duplicate target/linker funnel rows require input_n for weighted aggregation')
        grouped = stage_rows.groupby(keys, as_index=False).agg(retained_n=('retained_n', 'sum'), input_n=('input_n', 'sum'))
        if (grouped['input_n'] <= 0).any():
            raise ValueError('Heatmap input_n must be positive')
        grouped['cumulative_pct'] = 100.0 * grouped['retained_n'] / grouped['input_n']
        stage_rows = grouped
    table = stage_rows.pivot(index='pdb', columns='linker_size', values='cumulative_pct').sort_index()
    table = table.reindex(sorted(table.columns), axis=1)
    plt = _lazy_pyplot()
    width = max(7.0, 0.9 * len(table.columns) + 2.5)
    height = max(4.0, 0.55 * len(table.index) + 2.0)
    (fig, axis) = plt.subplots(figsize=(width, height))
    image = axis.imshow(table.to_numpy(dtype=float), aspect='auto', interpolation='nearest', cmap='viridis', vmin=0.0, vmax=100.0)
    axis.set_xticks(np.arange(len(table.columns)))
    axis.set_xticklabels([str(int(value)) for value in table.columns])
    axis.set_yticks(np.arange(len(table.index)))
    axis.set_yticklabels(table.index)
    axis.set_xlabel('Sampled linker atoms')
    axis.set_ylabel('PDB target')
    axis.set_title(f'Target-by-size cumulative survival at stage: {stage}')
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = table.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f'{value:.1f}', ha='center', va='center', fontsize=7, color='white' if value < 55 else 'black')
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label('Cumulative survival (%)')
    fig.tight_layout()
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs

def dataframe_to_latex(frame: pd.DataFrame, output_path: Path, *, columns: Optional[Sequence[str]]=None, float_precision: int=4) -> None:
    table = frame
    if columns is not None:
        missing = [column for column in columns if column not in table]
        if missing:
            raise ValueError(f'LaTeX table columns are missing: {missing}')
        table = table.loc[:, list(columns)]
    text = table.to_latex(index=False, escape=True, na_rep='--', float_format=lambda value: f'{value:.{float_precision}f}')
    atomic_write_text(output_path, text)

def csv_to_latex(input_path: Path, output_path: Path, *, columns: Optional[Sequence[str]]=None, float_precision: int=4) -> None:
    dataframe_to_latex(_read_csv(input_path), output_path, columns=columns, float_precision=float_precision)

def _ranking_frame(candidate_evaluation: pd.DataFrame, *, mw_target: float) -> pd.DataFrame:
    required = ('candidate_id', 'pdb', 'linker_size', 'final_pass', 'connected', 'novel', 'canonical_smiles', *FRAGMENT_ELIGIBILITY_FIELDS)
    result = _canonicalise_columns(candidate_evaluation, EVALUATION_ALIASES, required=required)
    result = _canonicalise_properties(result)
    if result['candidate_id'].isna().any():
        raise ValueError('candidate_id may not be missing')
    result['candidate_id'] = result['candidate_id'].astype(str).str.strip()
    if (result['candidate_id'] == '').any():
        raise ValueError('candidate_id may not be blank')
    if result['candidate_id'].duplicated().any():
        duplicates = result.loc[result['candidate_id'].duplicated(False), 'candidate_id'].head(5).tolist()
        raise ValueError(f'candidate_id values must be unique: {duplicates}')
    result['pdb'] = result['pdb'].astype(str).str.lower().str.strip()
    if (result['pdb'] == '').any():
        raise ValueError('pdb may not be blank')
    result['linker_size'] = _numeric_column(result, 'linker_size', integer=True)
    if result['linker_size'].isna().any():
        raise ValueError('linker_size may not be missing')
    for flag in ('final_pass', 'connected', 'novel'):
        result[flag] = _coerce_bool_series(result[flag], name=flag)
    for flag in FRAGMENT_ELIGIBILITY_FIELDS:
        result[flag] = _coerce_bool_series(result[flag].replace('', False).fillna(False), name=flag)
    result['canonical_smiles'] = result['canonical_smiles'].fillna('').astype(str).str.strip()
    normalized = [with_normalized_identity(row, require_valid=bool(row['final_pass'])) for row in result.to_dict('records')]
    result['normalized_identity'] = [row['normalized_identity'] for row in normalized]
    result['identity_normalization_status'] = [row['identity_normalization_status'] for row in normalized]
    result['prospective_eligible'] = result['final_pass'] & result['connected'] & result['novel'] & result.loc[:, list(FRAGMENT_ELIGIBILITY_FIELDS)].all(axis=1)
    bad_properties = result['prospective_eligible'] & result.loc[:, ['qed', 'sa', 'molecular_weight']].isna().any(axis=1)
    if bad_properties.any():
        ids = result.loc[bad_properties, 'candidate_id'].head(5).tolist()
        raise ValueError(f'Eligible candidates lack ranking properties: {ids}')
    blank_smiles = result['prospective_eligible'] & (result['canonical_smiles'] == '')
    if blank_smiles.any():
        ids = result.loc[blank_smiles, 'candidate_id'].head(5).tolist()
        raise ValueError(f'Eligible candidates lack canonical_smiles: {ids}')

    def rejection_reason(row: pd.Series) -> str:
        reasons = []
        if not row['final_pass']:
            reasons.append('not_final_pass')
        if not row['connected']:
            reasons.append('not_connected')
        if not row['novel']:
            reasons.append('not_novel')
        reasons.extend((field for field in FRAGMENT_ELIGIBILITY_FIELDS if not row[field]))
        return ';'.join(reasons)
    result['prospective_rejection_reason'] = result.apply(rejection_reason, axis=1)
    result['mw_target'] = float(mw_target)
    result['mw_distance'] = (result['molecular_weight'] - float(mw_target)).abs()
    result['prospective_chemistry_rank'] = pd.Series(pd.array([pd.NA] * len(result), dtype='Int64'), index=result.index)
    for (_, index) in result.loc[result['prospective_eligible']].groupby('pdb').groups.items():
        ordered = result.loc[index].sort_values(['qed', 'sa', 'mw_distance', 'candidate_id'], ascending=[False, True, True, True], kind='mergesort')
        result.loc[ordered.index, 'prospective_chemistry_rank'] = np.arange(1, len(ordered) + 1)
    result['prospective_selection_label'] = SELECTION_LABEL
    result['prospective_rank_definition'] = f'QED descending; SA ascending; abs(MW-{float(mw_target):g}) ascending; candidate_id ascending'
    return result.sort_values(['pdb', 'prospective_eligible', 'prospective_chemistry_rank', 'candidate_id'], ascending=[True, False, True, True], na_position='last', kind='mergesort').reset_index(drop=True)

def rank_candidates(candidate_evaluation: pd.DataFrame, *, mw_target: float=400.0) -> pd.DataFrame:
    if not math.isfinite(mw_target) or mw_target <= 0:
        raise ValueError('mw_target must be a positive finite number')
    return _ranking_frame(candidate_evaluation, mw_target=mw_target)

def _morgan_fingerprint(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Cannot parse eligible canonical SMILES: {smiles!r}')
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, MORGAN_RADIUS, nBits=MORGAN_BITS, useChirality=MORGAN_USE_CHIRALITY)
    return fingerprint

def _json_scalar(value: Any) -> Any:
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value

def select_diverse_candidates(ranked: pd.DataFrame, *, targets: Sequence[str]=DEFAULT_TARGETS, linker_sizes: Sequence[int]=DEFAULT_LINKER_SIZES, tanimoto_threshold: float=0.7) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from rdkit import DataStructs
    if not 0.0 <= tanimoto_threshold <= 1.0:
        raise ValueError('tanimoto_threshold must be in [0, 1]')
    required = {'candidate_id', 'pdb', 'linker_size', 'canonical_smiles', 'prospective_eligible', 'prospective_chemistry_rank', 'qed', 'sa', 'molecular_weight', 'mw_distance'}
    missing = sorted(required.difference(ranked.columns))
    if missing:
        raise ValueError(f'Ranked candidate table is missing columns: {missing}')
    target_order = tuple((str(target).lower() for target in targets))
    size_order = tuple((int(size) for size in linker_sizes))
    if len(set(target_order)) != len(target_order):
        raise ValueError('targets must be unique')
    if len(set(size_order)) != len(size_order):
        raise ValueError('linker_sizes must be unique')
    selected_rows: List[pd.Series] = []
    payload_targets: Dict[str, Any] = {}
    for target in target_order:
        selected_fingerprints = []
        selected_ids: List[str] = []
        selected_records: List[Dict[str, Any]] = []
        missing_sizes: List[Dict[str, Any]] = []
        target_rows = ranked.loc[(ranked['pdb'].str.lower() == target) & ranked['prospective_eligible']]
        target_rows = target_rows.sort_values(['prospective_chemistry_rank', 'candidate_id'], kind='mergesort')
        for size in size_order:
            candidates = target_rows.loc[target_rows['linker_size'] == size]
            chosen: Optional[pd.Series] = None
            chosen_fingerprint = None
            chosen_max_similarity = 0.0
            for (_, candidate) in candidates.iterrows():
                fingerprint = _morgan_fingerprint(candidate['normalized_identity'])
                similarities = [float(DataStructs.TanimotoSimilarity(fingerprint, previous)) for previous in selected_fingerprints]
                max_similarity = max(similarities, default=0.0)
                if max_similarity <= tanimoto_threshold + 1e-12:
                    chosen = candidate.copy()
                    chosen_fingerprint = fingerprint
                    chosen_max_similarity = max_similarity
                    break
            if chosen is None:
                reason = 'no_eligible_candidate' if candidates.empty else 'no_candidate_under_diversity_threshold'
                missing_sizes.append({'linker_size': size, 'reason': reason, 'eligible_candidates': int(len(candidates))})
                continue
            chosen['prospective_selection_order'] = len(selected_rows) + 1
            chosen['prospective_target_selection_order'] = len(selected_ids) + 1
            chosen['prospective_diversity_max_tanimoto'] = chosen_max_similarity
            chosen['prospective_diversity_threshold'] = tanimoto_threshold
            selected_rows.append(chosen)
            selected_fingerprints.append(chosen_fingerprint)
            selected_ids.append(str(chosen['candidate_id']))
            selected_records.append({key: _json_scalar(chosen[key]) for key in ('candidate_id', 'pdb', 'linker_size', 'canonical_smiles', 'normalized_identity', 'identity_normalization_status', 'qed', 'sa', 'molecular_weight', 'mw_distance', 'prospective_chemistry_rank', 'prospective_target_selection_order', 'prospective_diversity_max_tanimoto', *FRAGMENT_ELIGIBILITY_FIELDS, 'fragment_linker_bond_count', 'fragment_linker_edges_json', 'retained_stereo_errors_json', 'retained_nonanchor_h_changes_json')})
        payload_targets[target] = {'selected': selected_records, 'selected_candidate_ids': selected_ids, 'selected_count': len(selected_ids), 'eligible_candidates_by_linker_size': {str(size): int((target_rows['linker_size'] == size).sum()) for size in size_order}, 'missing_linker_sizes': missing_sizes}
    if selected_rows:
        selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    else:
        selected = ranked.iloc[0:0].copy()
        selected['prospective_selection_order'] = pd.Series(dtype='Int64')
        selected['prospective_target_selection_order'] = pd.Series(dtype='Int64')
        selected['prospective_diversity_max_tanimoto'] = pd.Series(dtype=float)
        selected['prospective_diversity_threshold'] = pd.Series(dtype=float)
    payload = {'schema_version': SCHEMA_VERSION, 'selection_label': SELECTION_LABEL, 'candidate_source': 'recorded_generation_run', 'selection_scope': 'Chemistry ranking with verified fragment attachments and retained chemistry', 'eligibility': ['final_pass', 'connected', 'novel', *FRAGMENT_ELIGIBILITY_FIELDS], 'fragment_verification': {'anchor_attachment_pass': 'Exactly two fragment-to-generated bonds, one at each declared anchor, with no other fragment-to-generated bonds', 'retained_stereo_pass': 'Source-defined retained stereochemistry restored with native neighbor-order parity', 'retained_nonanchor_explicit_h_match': 'Explicit hydrogen count retained at every nonanchor fragment atom', 'retained_nonanchor_total_h_match': 'Total hydrogen count retained at every nonanchor fragment atom', 'scope': 'Illustrated candidate eligibility; aggregate filter cohorts retain their declared definitions'}, 'ranking': {'keys': ['qed descending', 'sa ascending', 'mw_distance ascending', 'candidate_id ascending'], 'mw_target': float(ranked['mw_target'].iloc[0]) if 'mw_target' in ranked and (not ranked.empty) else None}, 'diversity': {'fingerprint': 'Morgan bit vector', 'radius': MORGAN_RADIUS, 'n_bits': MORGAN_BITS, 'use_chirality': MORGAN_USE_CHIRALITY, 'maximum_tanimoto': float(tanimoto_threshold), 'policy': 'For each target, consider linker sizes in declared order and select the highest chemistry-ranked candidate whose maximum similarity to previous selections is at most the threshold.'}, 'targets': payload_targets}
    return (selected, payload)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (descriptor, temporary_name) = tempfile.mkstemp(prefix=f'.{path.name}.tmp.', dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')

def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (descriptor, temporary_name) = tempfile.mkstemp(prefix=f'.{path.name}.tmp.', dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def iter_sdf_records(path: Path) -> Iterator[bytes]:
    record: List[bytes] = []
    with path.open('rb') as handle:
        for line in handle:
            record.append(line)
            if line.strip() == b'$$$$':
                yield b''.join(record)
                record = []
    if record and any((line.strip() for line in record)):
        raise ValueError(f'SDF has an unterminated final record: {path}')
_CANDIDATE_PROPERTY = re.compile(b'^>\\s*<candidate_id>[^\\r\\n]*\\r?\\n([^\\r\\n]+)', flags=re.MULTILINE)

def sdf_record_candidate_id(record: bytes) -> str:
    first_line = record.splitlines()[0].decode('utf-8', errors='strict').strip()
    match = _CANDIDATE_PROPERTY.search(record)
    property_id = match.group(1).decode('utf-8', errors='strict').strip() if match is not None else ''
    if property_id and first_line and (property_id != first_line):
        raise ValueError(f'SDF title/property candidate_id mismatch: {first_line!r} != {property_id!r}')
    candidate_id = property_id or first_line
    if not candidate_id:
        raise ValueError('SDF record has no candidate ID in title or property')
    return candidate_id

def extract_selected_sdf(source_sdf: Path, *, expected_candidate_ids: Sequence[str], selected_candidate_ids: Sequence[str], output_sdf: Path) -> Dict[str, Any]:
    expected = [str(value) for value in expected_candidate_ids]
    selected = [str(value) for value in selected_candidate_ids]
    if len(set(expected)) != len(expected):
        raise ValueError('Expected candidate IDs are not unique')
    if len(set(selected)) != len(selected):
        raise ValueError('Selected candidate IDs are not unique')
    unknown = sorted(set(selected).difference(expected))
    if unknown:
        raise ValueError(f'Selected candidate IDs are absent from evaluation CSV: {unknown}')
    expected_set = set(expected)
    wanted = set(selected)
    records: Dict[str, bytes] = {}
    source_ids: List[str] = []
    source_seen = set()
    record_count = 0
    for record in iter_sdf_records(source_sdf):
        candidate_id = sdf_record_candidate_id(record)
        if candidate_id in source_seen:
            raise ValueError(f'SDF candidate IDs are not unique: {candidate_id!r}')
        source_seen.add(candidate_id)
        source_ids.append(candidate_id)
        if candidate_id in wanted:
            records[candidate_id] = record
        record_count += 1
    missing_expected = [candidate_id for candidate_id in expected if candidate_id not in source_seen]
    if missing_expected:
        raise ValueError(f'Evaluation candidate IDs are absent from source SDF: {missing_expected[:5]}')
    aligned_expected = [candidate_id for candidate_id in source_ids if candidate_id in expected_set]
    if aligned_expected != expected:
        mismatch_index = next((index for (index, (csv_id, sdf_id)) in enumerate(zip(expected, aligned_expected)) if csv_id != sdf_id), min(len(expected), len(aligned_expected)))
        csv_id = expected[mismatch_index] if mismatch_index < len(expected) else None
        sdf_id = aligned_expected[mismatch_index] if mismatch_index < len(aligned_expected) else None
        raise ValueError(f'Candidate evaluation/SDF relative-order mismatch at survivor row {mismatch_index}: {csv_id!r} != {sdf_id!r}')
    missing = [candidate_id for candidate_id in selected if candidate_id not in records]
    if missing:
        raise ValueError(f'Selected SDF records were not recovered: {missing}')
    payload = b''.join((records[candidate_id] for candidate_id in selected))
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    (descriptor, temporary_name) = tempfile.mkstemp(prefix=f'.{output_sdf.name}.tmp.', dir=str(output_sdf.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_sdf)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {'records': len(selected), 'source_records': record_count, 'evaluation_records': len(expected), 'alignment': 'evaluation IDs are an order-preserving SDF subsequence', 'sha256': sha256_bytes(payload), 'bytes': len(payload), 'candidate_ids': selected}

def build_ranking_artifacts(candidate_evaluation: pd.DataFrame, *, candidate_evaluation_path: Path, candidate_sdf_path: Path, output_dir: Path, mw_target: float=400.0, tanimoto_threshold: float=0.7, targets: Sequence[str]=DEFAULT_TARGETS, linker_sizes: Sequence[int]=DEFAULT_LINKER_SIZES) -> Dict[str, Any]:
    original = _canonicalise_columns(candidate_evaluation, EVALUATION_ALIASES, required=('candidate_id',))
    expected_ids = original['candidate_id'].astype(str).str.strip().tolist()
    ranked = rank_candidates(candidate_evaluation, mw_target=mw_target)
    (selected, selection) = select_diverse_candidates(ranked, targets=targets, linker_sizes=linker_sizes, tanimoto_threshold=tanimoto_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / 'ranking_all.csv'
    atomic_write_csv(ranking_path, ranked)
    outputs: Dict[str, Any] = {ranking_path.name: {'records': int(len(ranked)), 'sha256': sha256_file(ranking_path)}}
    source_record_count: Optional[int] = None
    for target in (str(value).lower() for value in targets):
        target_selected = selected.loc[selected['pdb'].str.lower() == target].copy()
        target_selected = target_selected.sort_values(['linker_size', 'prospective_chemistry_rank', 'candidate_id'], kind='mergesort')
        csv_path = output_dir / f'selected_{target}.csv'
        sdf_path = output_dir / f'selected_{target}.sdf'
        atomic_write_csv(csv_path, target_selected)
        sdf_report = extract_selected_sdf(candidate_sdf_path, expected_candidate_ids=expected_ids, selected_candidate_ids=target_selected['candidate_id'].astype(str).tolist(), output_sdf=sdf_path)
        if source_record_count is None:
            source_record_count = int(sdf_report['source_records'])
        elif source_record_count != int(sdf_report['source_records']):
            raise RuntimeError('Source SDF record count changed during extraction')
        outputs[csv_path.name] = {'records': int(len(target_selected)), 'sha256': sha256_file(csv_path)}
        outputs[sdf_path.name] = sdf_report
    selection['inputs'] = {'candidate_evaluation_csv': {'path': manifest_path(candidate_evaluation_path), 'records': int(len(candidate_evaluation)), 'sha256': sha256_file(candidate_evaluation_path)}, 'candidate_sdf': {'path': manifest_path(candidate_sdf_path), 'records': source_record_count, 'aligned_evaluation_records': len(expected_ids), 'sha256': sha256_file(candidate_sdf_path)}}
    selection['outputs'] = outputs
    selection_path = output_dir / 'selection.json'
    atomic_write_json(selection_path, selection)
    outputs[selection_path.name] = {'sha256': sha256_file(selection_path)}
    return selection

def project_path(value: str, *, must_exist: bool=False) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f'CLI paths must be project-root-relative: {value}')
    root = PROJECT_ROOT.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'CLI path escapes project root: {value}') from exc
    if must_exist and (not resolved.exists()):
        raise FileNotFoundError(resolved)
    return resolved

def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    candidate_path = project_path(args.candidate_evaluation, must_exist=True)
    gt_path = project_path(args.gt_evaluation, must_exist=True)
    funnel_path = project_path(args.funnel, must_exist=True)
    candidate_sdf_path = project_path(args.candidate_sdf, must_exist=True)
    output_dir = project_path(args.output_dir)
    candidates = _read_csv(candidate_path)
    ground_truth = _read_csv(gt_path)
    funnel = _read_csv(funnel_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    ranking_dir = output_dir / 'ranking'
    summary = summarize_properties(candidates, ground_truth, population=args.population, deduplicate_smiles=args.deduplicate_canonical_smiles)
    summary_path = tables_dir / 'property_summary.csv'
    atomic_write_csv(summary_path, summary)
    dataframe_to_latex(summary, tables_dir / 'property_summary.tex', columns=('group', 'property', 'population_n', 'n', 'mean', 'sd', 'median', 'q1', 'q3', 'min', 'max'))
    dataframe_to_latex(canonicalize_funnel(funnel), tables_dir / 'funnel.tex')
    figure_hashes = plot_property_distributions(candidates, ground_truth, output_dir=figures_dir, population=args.population, deduplicate_smiles=args.deduplicate_canonical_smiles)
    figure_hashes.update(plot_funnel(funnel, output_base=figures_dir / 'survival_funnel'))
    figure_hashes.update(plot_target_size_heatmap(funnel, output_base=figures_dir / 'target_by_linker_size_heatmap', stage=args.heatmap_stage))
    selection = build_ranking_artifacts(candidates, candidate_evaluation_path=candidate_path, candidate_sdf_path=candidate_sdf_path, output_dir=ranking_dir, mw_target=args.mw_target, tanimoto_threshold=args.tanimoto_threshold, targets=args.targets, linker_sizes=args.linker_sizes)
    for target in args.targets:
        selected_path = ranking_dir / f'selected_{target.lower()}.csv'
        dataframe_to_latex(pd.read_csv(selected_path), tables_dir / f'selected_{target.lower()}.tex', columns=('candidate_id', 'pdb', 'linker_size', 'qed', 'sa', 'molecular_weight', 'prospective_chemistry_rank', 'prospective_diversity_max_tanimoto'))
    manifest = {'schema_version': SCHEMA_VERSION, 'report_scope': 'prospective paper-result reporting', 'population': args.population, 'deduplicate_canonical_smiles': bool(args.deduplicate_canonical_smiles), 'property_deduplication_scope': 'Within each sampled linker size for size panels; globally for the overall generated and GT comparison', 'inputs': {'candidate_evaluation': {'path': args.candidate_evaluation, 'sha256': sha256_file(candidate_path)}, 'gt_evaluation': {'path': args.gt_evaluation, 'sha256': sha256_file(gt_path)}, 'funnel': {'path': args.funnel, 'sha256': sha256_file(funnel_path)}, 'candidate_sdf': {'path': args.candidate_sdf, 'sha256': sha256_file(candidate_sdf_path)}}, 'property_definitions': {spec.name: {'label': spec.label, 'discrete': spec.discrete} for spec in PROPERTY_SPECS}, 'figures': figure_hashes, 'selection_label': selection['selection_label'], 'selection_json': 'ranking/selection.json'}
    atomic_write_json(output_dir / 'report_manifest.json', manifest)
    return manifest

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate paper-result reports and prospective candidate selections.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    build = subparsers.add_parser('build', help='Build the complete report bundle')
    build.add_argument('--candidate-evaluation', required=True)
    build.add_argument('--gt-evaluation', required=True)
    build.add_argument('--funnel', required=True)
    build.add_argument('--candidate-sdf', required=True)
    build.add_argument('--output-dir', required=True)
    build.add_argument('--population', choices=('final_pass', 'all'), default='final_pass')
    build.add_argument('--deduplicate-canonical-smiles', action='store_true', help='Deduplicate property summaries/plots after population filtering.')
    build.add_argument('--heatmap-stage', default='final_pass')
    build.add_argument('--mw-target', type=float, default=400.0)
    build.add_argument('--tanimoto-threshold', type=float, default=0.7)
    build.add_argument('--targets', nargs='+', default=list(DEFAULT_TARGETS))
    build.add_argument('--linker-sizes', nargs='+', type=int, default=list(DEFAULT_LINKER_SIZES))
    build.set_defaults(handler=build_report)
    latex = subparsers.add_parser('latex', help='Convert an arbitrary project-relative CSV to a LaTeX tabular fragment')
    latex.add_argument('--csv', required=True)
    latex.add_argument('--output', required=True)
    latex.add_argument('--columns', nargs='*')
    latex.add_argument('--float-precision', type=int, default=4)

    def latex_handler(args: argparse.Namespace) -> Dict[str, Any]:
        input_path = project_path(args.csv, must_exist=True)
        output_path = project_path(args.output)
        csv_to_latex(input_path, output_path, columns=args.columns, float_precision=args.float_precision)
        return {'input': args.csv, 'output': args.output, 'sha256': sha256_file(output_path)}
    latex.set_defaults(handler=latex_handler)
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
