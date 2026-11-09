"""Read only checkpoint diagnostics for a locally generated, trusted Experiment 2 checkpoint.

Usage: python -m examples.inspect_experiment2_convergence --checkpoint PATH [--csv PATH]
Does not need the sdmcar package or alter the source checkpoint.
"""
import argparse
from pathlib import Path
import pandas as pd
import torch

FIELDS = [
    'iteration', 'evaluation_ELBO', 'ELBO_improvement_per_site',
    'ELBO_abs_upper_2MCSE_per_site', 'precision_relative_change',
    'zero_mode_precision_relative_change', 'zero_mode_variance_relative_change',
    'zero_mode_precision_relative_change_exact', 'zero_mode_variance_relative_change_exact',
    'zero_mode_guard_uses_exact',
    'pilot_ELBO_pass','pilot_precision_pass','pilot_zero_precision_pass','pilot_zero_variance_pass',
    'pilot_checkpoint_pass','pilot_consecutive_passes','pilot_persistent_pass',
    'w1000_precision_relative_change','w2000_precision_relative_change',
    'w2000_zero_variance_relative_change','w2000_zero_variance_relative_change_exact',
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--csv',type=Path,default=None,help='Optional CSV for full convergence trajectory')
    parser.add_argument('--last',type=int,default=12,help='Number of checkpoint rows to display')
    args=parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    # weights_only=False is needed for the original checkpoint's NumPy RNG and histories.
    # Use ONLY checkpoints produced by your own Experiment 2 run.
    ck = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    df=pd.DataFrame(ck['convergence_history'])
    if df.empty:
        raise RuntimeError('Empty checkpoint convergence history')
    cols=[c for c in FIELDS if c in df]
    last=df.iloc[-1]
    print('Checkpoint:', args.checkpoint)
    print('Seed:', ck.get('seed'),'Model:',ck.get('label'),'Final iteration:',ck.get('iteration'))
    print('Final persistent pass:',bool(last.get('pilot_persistent_pass',False)))
    if 'pilot_persistent_pass' in df:
        passes=df.loc[df['pilot_persistent_pass'].fillna(False),'iteration']
        print('Previously passing checkpoints:', len(passes),
              '| last:', int(passes.iloc[-1]) if len(passes) else 'none')
    failed=[name for name in ('pilot_ELBO_pass','pilot_precision_pass',
                'pilot_zero_precision_pass','pilot_zero_variance_pass')
            if name in last and not bool(last[name])]
    print('Failed individual checks at final checkpoint:', ', '.join(failed) or 'none')
    print('Final consecutive passes:',last.get('pilot_consecutive_passes','missing'))
    print('Pilot tolerances: ELBO + 2MCSE < 0.01/site; precision < 0.005; zero q and F changes < 0.02; 3 consecutive passes.')
    print('\nFinal checkpoints:')
    pd.set_option('display.max_columns',len(cols))
    pd.set_option('display.width',300)
    print(df[cols].tail(args.last).to_string(index=False,float_format=lambda x:f'{x:.6g}'))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv,index=False)
        print('\nWrote full convergence history:',args.csv)

if __name__ == '__main__':
    main()
