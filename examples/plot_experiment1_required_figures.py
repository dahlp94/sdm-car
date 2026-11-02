#!/usr/bin/env python3
"""Create ONLY the two agreed two-panel figures for Experiment 1.

Source: completed Experiment 1 master tables. This script never imports the model,
regenerates data, refits a model, changes predictions, or alters existing plots.

Figure 1:
    A) True Leroux modal precision and both models' across-seed means,
       with *pointwise 95% Student-t confidence intervals for each mean*.
    B) Same means, with *pointwise 5th-to-95th empirical percentiles*
       of the seed-specific fitted posterior mean precision curves.
Figure 2:
    A) Across-seed mean SDM-CAR residual from its OWN best affine projection,
       with pointwise 5th-to-95th seed-percentile band.
    B) Relative distance to an affine precision curve for each of 50 seeds,
       plus median and interquartile range. This is not second-derivative curvature.

Both models' modal precision estimates are E_q[1/F(lambda)], NOT 1/E_q[F].
In repeated eigenvalues all 144 *modes* are equally weighted in projection.

Required libraries: numpy, pandas, scipy, matplotlib. Python >= 3.9.

Example (from the project directory):
    python -m examples.plot_experiment1_required_figures \
        --input "C:\\Users\\pd006\\Desktop\\internship_search\\sdm-car\\experiment1_N144_final" \
        --output-dir "C:\\Users\\pd006\\Desktop\\internship_search\\sdm-car\\experiment1_N144_final\\required_plots"

--input may be the experiment output directory, its master_tables directory,
OR the experiment1_N144_final_analysis.zip file. All variants require
all_runs__precision_recovery.csv and all_runs__affine_precision_metrics.csv.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

TRUE_COLOR = '#1c1c1c'
LEROUX_COLOR = '#1d5d8f'
SDM_COLOR = '#c36e22'
BAND_ALPHA_CI = 0.19
BAND_ALPHA_SPREAD = 0.16

SOURCE_PRECISION = 'all_runs__precision_recovery.csv'
SOURCE_AFFINE = 'all_runs__affine_precision_metrics.csv'
SDM_LABEL = 'Adaptive P-spline SDM-CAR'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True,
                        help='Experiment output directory, master_tables directory, or analysis ZIP.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='New destination for exactly two figures (PNG + PDF).')
    parser.add_argument('--dpi', type=int, default=300, help='Raster PNG resolution (default: 300).')
    args = parser.parse_args()
    if args.dpi < 100:
        parser.error('--dpi must be >= 100.')
    return args


def read_master_tables(root: Path):
    """Read existing CSVs, including Windows-path ZIP members, without changing them."""
    if root.is_file() and root.suffix.lower() == '.zip':
        with ZipFile(root) as archive:
            csv_frames = []
            for wanted in (SOURCE_PRECISION, SOURCE_AFFINE):
                candidates = [name for name in archive.namelist()
                              if name.replace('\\', '/').split('/')[-1] == wanted]
                if len(candidates) != 1:
                    raise ValueError(f'Expected exactly one {wanted} in {root}, found {len(candidates)}.')
                csv_frames.append(pd.read_csv(io.BytesIO(archive.read(candidates[0]))))
            return tuple(csv_frames)
    master = root / 'master_tables' if (root / 'master_tables').is_dir() else root
    missing = [f for f in (SOURCE_PRECISION, SOURCE_AFFINE) if not (master / f).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing required CSV(s) in {master}: {missing}')
    return pd.read_csv(master / SOURCE_PRECISION), pd.read_csv(master / SOURCE_AFFINE)


def validate_and_prepare(precision: pd.DataFrame, affine: pd.DataFrame):
    required = {'seed', 'mode', 'lambda', 'q_true', 'q_leroux_mean', 'q_sdm_mean',
                'q_sdm_best_affine', 'q_sdm_affine_residual'}
    if required - set(precision.columns):
        raise ValueError(f'Precision table missing fields: {sorted(required-set(precision.columns))}')
    req_affine = {'seed', 'model', 'affine_precision_relative_L2_mean_curve'}
    if req_affine - set(affine.columns):
        raise ValueError(f'Affine table missing fields: {sorted(req_affine-set(affine.columns))}')

    expected_seeds = list(range(1001, 1051))
    expected_modes = list(range(144))
    seeds = sorted(precision.seed.unique().tolist())
    if seeds != expected_seeds:
        raise ValueError('This is the final Experiment 1 plotter: expected exactly seeds 1001..1050, '
                         f'but found {seeds}.')
    if precision.duplicated(['seed', 'mode']).any():
        raise ValueError('Duplicate seed/mode records in precision table.')
    counts = precision.groupby('seed').size()
    if not counts.eq(144).all():
        raise ValueError('Every seed must have exactly 144 graph modes.')
    precision = precision.sort_values(['seed', 'mode']).reset_index(drop=True)
    for _, f in precision.groupby('seed', sort=True):
        if sorted(f['mode'].astype(int).tolist()) != expected_modes:
            raise ValueError('Some seed is missing a graph mode or has noncanonical mode IDs.')
    cols = ['lambda', 'q_true', 'q_leroux_mean', 'q_sdm_mean',
            'q_sdm_best_affine', 'q_sdm_affine_residual']
    if not np.isfinite(precision[cols].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite entries in precision table.')
    if (precision[['q_true', 'q_leroux_mean', 'q_sdm_mean']].to_numpy() <= 0).any():
        raise ValueError('Modal precisions must be strictly positive.')

    lam = np.stack([f['lambda'].to_numpy(dtype=float) for _, f in precision.groupby('seed', sort=True)])
    true = np.stack([f.q_true.to_numpy(dtype=float) for _, f in precision.groupby('seed', sort=True)])
    leroux = np.stack([f.q_leroux_mean.to_numpy(dtype=float) for _, f in precision.groupby('seed', sort=True)])
    sdm = np.stack([f.q_sdm_mean.to_numpy(dtype=float) for _, f in precision.groupby('seed', sort=True)])
    q_aff_saved = np.stack([f.q_sdm_best_affine.to_numpy(dtype=float) for _, f in precision.groupby('seed', sort=True)])
    resid_saved = np.stack([f.q_sdm_affine_residual.to_numpy(dtype=float) for _, f in precision.groupby('seed', sort=True)])

    if not np.allclose(lam, lam[0], atol=1e-11, rtol=0):
        raise ValueError('Graph eigenvalues differ across seeds; the common-basis comparison is invalid.')
    if not np.allclose(true, true[0], atol=1e-10, rtol=0):
        raise ValueError('True Leroux precision differs across seeds.')
    if np.count_nonzero(lam[0] < 1e-10) != 1:
        raise ValueError('Expected one (and only one) zero-frequency eigenmode.')
    if np.any(np.diff(lam[0]) < -1e-11):
        raise ValueError('Eigenvalues not sorted by graph mode; inspect the input.')

    # Reconstruct seed-specific least-squares affine projections using ALL 144 modes.
    X = np.column_stack([np.ones(144), lam[0]])
    a_hat = (np.linalg.lstsq(X, sdm.T, rcond=None)[0]).T
    q_aff = a_hat @ X.T
    residual = sdm - q_aff
    if not np.allclose(q_aff, q_aff_saved, rtol=1e-9, atol=1e-9):
        raise ValueError('Stored best-affine curves are inconsistent with a fresh LS projection.')
    if not np.allclose(residual, resid_saved, rtol=1e-9, atol=1e-9):
        raise ValueError('Stored residuals are inconsistent with fitted SDM - best-affine.')
    if not np.allclose(X @ np.linalg.lstsq(X, true[0], rcond=None)[0], true[0], atol=1e-10, rtol=0):
        raise ValueError('Leroux generating truth is not affine in the supplied eigenvalues.')
    B = np.linalg.norm(residual, axis=1) / np.linalg.norm(sdm, axis=1)

    sdma = affine[affine.model.eq(SDM_LABEL)].sort_values('seed')
    if sdma.seed.tolist() != expected_seeds:
        raise ValueError('Missing/duplicated SDM-CAR affine diagnostic records.')
    if not np.allclose(B, sdma.affine_precision_relative_L2_mean_curve, atol=2e-9, rtol=1e-8):
        raise ValueError('Recomputed bending does not match saved affine metrics.')

    return dict(seeds=np.asarray(seeds), lam=lam[0], truth=true[0],
                leroux=leroux, sdm=sdm, residual=residual, bending=B)


def draw_band(ax, x, center, lower, upper, color, label, alpha):
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0, label=label)
    ax.plot(x, center, color=color, linewidth=1.75)


def create_precision_figure(data, out_dir: Path, dpi: int):
    lam, truth, leroux, sdm = (data[k] for k in ('lam', 'truth', 'leroux', 'sdm'))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True, sharey=True)
    n = leroux.shape[0]
    tc = float(student_t.ppf(.975, n - 1))
    order = np.argsort(lam, kind='stable')
    x = lam[order]
    for ax, kind in ((ax_a, 'mean_CI'), (ax_b, 'replication_range')):
        for values, color, short in ((leroux, LEROUX_COLOR, 'Leroux-CAR'),
                                     (sdm, SDM_COLOR, 'SDM-CAR')):
            mean = values.mean(axis=0)
            if kind == 'mean_CI':
                margin = tc * values.std(axis=0, ddof=1) / math.sqrt(n)
                lo, hi = mean - margin, mean + margin
                spread_label = f'{short}: pointwise 95% CI for mean'
                alpha = BAND_ALPHA_CI
            else:
                lo, hi = np.quantile(values, (.05, .95), axis=0)
                spread_label = f'{short}: 5th–95th percentile band'
                alpha = BAND_ALPHA_SPREAD
            draw_band(ax, x, mean[order], lo[order], hi[order], color, spread_label, alpha)
        ax.plot(x, truth[order], color=TRUE_COLOR, linewidth=2.0,
                linestyle='--', label='True Leroux precision')
        ax.set_xlabel('Graph Laplacian eigenvalue $\\lambda$')
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
        ax.set_xlim(x.min(), x.max())
    ax_a.set_title('A. Average recovery (95% CI for mean)')
    ax_b.set_title('B. Between-dataset variation (central 90%)')
    ax_a.set_ylabel('Modal precision $q(\\lambda)$')
    # One common legend reduces chart crowding; labels disclose band type.
    # Custom compact legend for curve identities, explanation in panel titles and footnote.
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0],[0],color=TRUE_COLOR,lw=2,ls='--',label='Leroux truth'),
                      Line2D([0],[0],color=LEROUX_COLOR,lw=2,label='Fitted Leroux-CAR'),
                      Line2D([0],[0],color=SDM_COLOR,lw=2,label='Fitted SDM-CAR')]
    fig.legend(handles=legend_handles,loc='lower center',ncol=3,
               bbox_to_anchor=(.5,.075),frameon=False)
    fig.text(.5,.037,'Both panels show the across-seed fitted means. A: pointwise $t_{49}$ confidence intervals; '
             'B: empirical 5th–95th percentiles across 50 fitted curves.',ha='center',fontsize=8.3)
    fig.subplots_adjust(left=.075,right=.985,top=.90,bottom=.225,wspace=.17)
    save_both(fig,out_dir/'experiment1_precision_recovery_two_panel',dpi)


def create_nonaffinity_figure(data, out_dir: Path, dpi: int):
    lam, residual, seeds, B = (data[k] for k in ('lam', 'residual', 'seeds', 'bending'))
    xorder = np.argsort(lam, kind='stable')
    xx = lam[xorder]
    rr = residual.mean(axis=0)
    lo, hi = np.quantile(residual, (.05, .95), axis=0)
    fig,(ax_a,ax_b) = plt.subplots(1,2,figsize=(13.2,5.2))
    ax_a.fill_between(xx,lo[xorder],hi[xorder],color=SDM_COLOR,alpha=.21,
                      label='Across-seed 5th–95th percentiles')
    ax_a.plot(xx,rr[xorder],color=SDM_COLOR,lw=1.9,label='Across-seed mean residual')
    ax_a.axhline(0,color=TRUE_COLOR,lw=1.25,ls='--',label='Exact affine relationship')
    ax_a.set_xlim(xx.min(),xx.max())
    ax_a.set_xlabel('Graph Laplacian eigenvalue $\\lambda$')
    ax_a.set_ylabel(r'$\widehat q_{\mathrm{SDM},r}(\lambda)-\widehat q_{\mathrm{aff},r}(\lambda)$')
    ax_a.set_title('A. Residual from each fit’s best affine curve')
    ax_a.legend(loc='lower right',fontsize=8.3,frameon=False)
    ax_a.grid(alpha=.15);ax_a.set_axisbelow(True)

    b_pct=100*B
    q1, med, q3 = np.quantile(b_pct,(.25,.50,.75))
    ax_b.axhspan(q1,q3,color=SDM_COLOR,alpha=.15,label='Across-seed interquartile range')
    ax_b.axhline(med,color=SDM_COLOR,ls='--',lw=1.7,label=f'Median = {med:.2f}%')
    ax_b.scatter(seeds,b_pct,s=27,alpha=.85,color=SDM_COLOR,edgecolors='white',lw=.35,
                 label='Individual replication',zorder=3)
    ax_b.set_xlabel('Simulation seed')
    ax_b.set_ylabel('Relative non-affinity $B_r$ (%)')
    ax_b.set_title('B. Non-affinity magnitude across 50 datasets')
    ax_b.set_xlim(999.5,1051.5)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(alpha=.15);ax_b.set_axisbelow(True)
    ax_b.legend(loc='upper left',fontsize=8.3,frameon=False)
    fig.text(.5,.04, r'$B_r=\|\widehat q_{\mathrm{SDM},r}-\widehat q_{\mathrm{aff},r}\|_2/'
             r'\|\widehat q_{\mathrm{SDM},r}\|_2$; '
             'projections count all 144 graph modes equally.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.085,right=.985,top=.90,bottom=.215,wspace=.20)
    save_both(fig,out_dir/'experiment1_sdm_nonaffinity_two_panel',dpi)


def save_both(fig, stem: Path, dpi: int):
    fig.savefig(stem.with_suffix('.png'),dpi=dpi,facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'),facecolor='white')
    plt.close(fig)


def main():
    args=parse_args()
    root=args.input.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'Input does not exist: {root}')
    out = (args.output_dir.expanduser().resolve() if args.output_dir else
           ((root.parent if root.is_file() else root) / 'required_plots'))
    out.mkdir(parents=True,exist_ok=True)
    precision, affine = read_master_tables(root)
    data = validate_and_prepare(precision, affine)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    create_precision_figure(data, out, args.dpi)
    create_nonaffinity_figure(data, out, args.dpi)
    b = 100*data['bending']
    print('VALIDATED: 50 replications, 144 modes, same eigenvalues and true precision, '
          'matching saved affine residuals and bending metrics.')
    print(f'Non-affinity B: mean = {b.mean():.3f}%, median = {np.median(b):.3f}%, '
          f'IQR = [{np.quantile(b,.25):.3f}%, {np.quantile(b,.75):.3f}%].')
    print('Saved ONLY these figures:')
    for base in ('experiment1_precision_recovery_two_panel', 'experiment1_sdm_nonaffinity_two_panel'):
        for ext in ('.png','.pdf'):
            print('  ',out / (base+ext))

if __name__ == '__main__':
    main()
