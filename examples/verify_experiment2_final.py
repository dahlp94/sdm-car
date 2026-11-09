"""Read-only completion and protocol checks for Experiment 2 two-start output.

Usage: python -m examples.verify_experiment2_final --output-dir PATH
For development output add: --mode smoke --seeds 9991
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

METRICS = ('response_RMSE','response_MAE','latent_phi_RMSE',
           'predictive_95_coverage','predictive_95_avg_width',
           'joint_response_NLPD_per_site','energy_score',
           'oracle_conditional_KL_per_site','joint_predictive_cov_relative_frobenius')

def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--mode',choices=('final','smoke'),default='final')
    p.add_argument('--seeds',type=int,nargs='+',default=None)
    a=p.parse_args()
    base=a.output_dir.resolve()
    seeds=a.seeds if a.seeds is not None else (list(range(1001,1051)) if a.mode=='final' else [9991])
    check(len(seeds)==len(set(seeds)), 'Duplicate requested seeds')
    selections=[]; predict=[]; spectra=[]
    for seed in seeds:
        prefix=f'seed_{seed:05d}'
        d=base/'per_run'/prefix
        metadata=d/f'{prefix}__metadata.json'
        done=d/f'{prefix}__COMPLETE.txt'
        check(done.is_file() and metadata.is_file(),f'Seed {seed}: incomplete or missing metadata')
        m=json.loads(metadata.read_text(encoding='utf-8'))
        check(m['run_mode']==a.mode, f'Seed {seed}: wrong mode')
        check(m['beta_handling']=='known_fixed_zero_generation_fit_oracle',f'Seed {seed}: beta design')
        check(m['fixed_sigma2'] is True and m['sigma2_true']==0.15,f'Seed {seed}: sigma2 design')
        check(m['two_start_method']=='original_and_curved',f'Seed {seed}: not two-start')
        check(m['selection_elbo_repeats']==(128 if a.mode=='final' else 8),f'Seed {seed}: ELBO bank')
        if a.mode=='final':
            check(m['predictive_draws']==2048 and m['oracle_kl_mc_draws']==4096 and
                  m['spectrum_draws']==2048 and m['num_mc']==4,
                  f'Seed {seed}: draw counts differ')
        check((d/'frozen_inputs.npz').is_file(),f'Seed {seed}: missing exact data')
        with np.load(d/'frozen_inputs.npz',allow_pickle=False) as f:
            check(f['y'].shape==(144,) and f['U_true'].shape==(144,144),f'Seed {seed}: data shape')
            check(int(f['test_mask'].sum())==16,f'Seed {seed}: holdout')
        selection=json.loads((d/'sdm_two_start_selection.json').read_text(encoding='utf-8'))
        status=json.loads((d/'sdm_two_start_final_status.json').read_text(encoding='utf-8'))
        chosen=('curve' if selection['difference_per_site']>
                2*selection['mcse_per_site'] else 'original')
        check(selection['selected_start']==chosen==status['selected_start']==m['selected_start'],
              f'Seed {seed}: inconsistent training ELBO selection')
        if a.mode=='final':
            check(bool(status['selected_convergence_pass']) and bool(status['leroux_convergence_pass']),
                  f'Seed {seed}: selected fit or Leroux convergence failed')
            for fit in ('sdm_original','sdm_curve'):
                path=d/'fits'/fit/'checkpoints'/f'{prefix}__sdm_iter_014000.pt'
                check(path.is_file(),f'Seed {seed}: missing original 14k {fit} checkpoint')
        selection['seed']=seed
        selections.append(selection)
        for filename,dest in (('predictive_metrics',predict),('spectral_metrics',spectra)):
            f=d/f'{prefix}__{filename}.csv'
            check(f.is_file(),f'Seed {seed}: missing {filename}')
            t=pd.read_csv(f)
            check(len(t)==2 and t['model'].nunique()==2, f'Seed {seed}: wrong models in {filename}')
            check(t.select_dtypes('number').replace([np.inf,-np.inf],np.nan).notna().all().all(),
                  f'Seed {seed}: nonfinite numeric metrics')
            dest.append(t)
    ptab=pd.concat(predict,ignore_index=True)
    check(all(c in ptab.columns for c in METRICS),'Missing required predictive metrics')
    check(len(ptab)==2*len(seeds),'Wrong final model rows')
    sel=pd.DataFrame(selections)
    tab=ptab.pivot(index='seed',columns='model',values=list(METRICS))
    check(len(tab)==len(seeds),'Paired dataset count mismatch')
    print('VALIDATED:',len(seeds),'datasets; 2 fitted families per dataset; mode=',a.mode)
    print('Selected Original:',int((sel['selected_start']=='original').sum()))
    print('Selected Curved:',int((sel['selected_start']=='curve').sum()))
    print('All required output files and model rows are present.')
    print('Choice reproducible from training-ELBO selection rule; fixed beta and sigma2 verified.')
    if a.mode=='final' and len(seeds)==50:
        check((base/'master_tables'/'summary__paired_sdm_minus_leroux.csv').is_file(),
              'Missing paired summary')
        print('50-replication bank is complete. Statistical analysis may proceed.')

if __name__=='__main__':
    main()
