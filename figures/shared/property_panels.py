from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PROPERTIES = (('rotatable_bond_count', 'Rotatable bonds', True), ('ring_count', 'Rings', True), ('fused_ring_count', 'Fused ring systems', True), ('heteroaromatic_ring_count', 'Heteroaromatic rings', True), ('aromatic_ring_count', 'Aromatic rings', True), ('logp', 'logP', False), ('qed', 'QED', False), ('sa_score', 'SA score', False), ('molecular_weight', 'Molecular weight (Da)', False))
COLORS = ('#7c98c7', '#80b6a2', '#b5ad70', '#cc9c71', '#b98caa', '#a99fca')

def summarize(table, cohort):
    groups = [(str(s), table.loc[table.linker_size.astype(int) == s]) for s in range(8, 14)] if cohort == 'generated' else [('GT', table)]
    summaries = []
    for (column, _, _) in PROPERTIES:
        for (label, group) in groups:
            values = pd.to_numeric(group[column], errors='raise').to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite values: {column}/{label}')
            record = {'cohort': label, 'property': column, 'n': len(values)}
            if len(values):
                record.update(mean=float(values.mean()), median=float(np.median(values)), sd=float(values.std(ddof=1)) if len(values) > 1 else 0.0, minimum=float(values.min()), maximum=float(values.max()))
            summaries.append(record)
    return (groups, summaries)

def render(groups, cohort, width_inches):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6})
    height = 6.5 if cohort == 'generated' else 6.3
    (fig, axes) = plt.subplots(3, 3, figsize=(width_inches, height))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.16 if cohort == 'generated' else 0.075, top=0.915, wspace=0.53, hspace=0.6)
    for (axis, (column, title, discrete)) in zip(axes.flat, PROPERTIES):
        axis.set_title(title, fontsize=8.25, pad=6)
        for (index, (_, group)) in enumerate(groups):
            values = pd.to_numeric(group[column], errors='raise').to_numpy(dtype=float)
            color = COLORS[index] if cohort == 'generated' else '#bf88a8'
            if len(values):
                if discrete:
                    axis.bar(index, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0, color=color, capsize=2, width=0.7, error_kw={'elinewidth': 0.8, 'capthick': 0.8})
                elif len(values) > 1 and np.ptp(values) > 0:
                    parts = axis.violinplot(values, positions=[index], widths=0.85, showmedians=True)
                    for body in parts['bodies']:
                        body.set_facecolor(color)
                        body.set_alpha(0.8)
                    for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        parts[key].set_color('#333333')
                        parts[key].set_linewidth(0.8)
                else:
                    axis.scatter([index], [values[0]], color=color, s=16)
        axis.set_xticks(range(len(groups)) if cohort == 'generated' else [])
        if cohort == 'generated':
            axis.set_xticklabels([label for (label, _) in groups])
        axis.set_xlim(-0.65, len(groups) - 0.35)
        axis.tick_params(axis='both', length=2.5, pad=2)
        axis.spines[['top', 'right']].set_visible(False)
        if discrete:
            axis.set_ylabel('Mean ± SD', labelpad=3)
    if cohort == 'gt':
        fig.suptitle(f'Ground-truth final survivors (n = {len(groups[0][1])})', fontsize=9, y=0.976)
        fig.text(0.5, 0.025, 'Bars: mean ± sample SD.  Violins: distributions and medians.', ha='center', fontsize=8)
    else:
        fig.suptitle('Generated final survivors by linker length', fontsize=9, y=0.976)
        fig.text(0.5, 0.102, 'Linker heavy atoms and cohort size (n)', ha='center', fontsize=8)
        for (index, (label, group)) in enumerate(groups):
            x = 0.13 + index * 0.15
            fig.text(x, 0.07, label, ha='center', fontsize=8)
            fig.text(x, 0.045, f'n = {len(group)}', ha='center', fontsize=8)
        fig.text(0.5, 0.014, 'Bars: mean ± sample SD.  Violins: distributions and medians.', ha='center', fontsize=8)
    return (fig, axes)
