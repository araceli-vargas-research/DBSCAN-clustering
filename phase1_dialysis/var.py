"""Exploratory reduced-form VAR(1), with transparent short-sample override.
Default: refuse fewer than 30 consecutive annual observations.
python var.py --allow-short-sample estimates a fragile descriptive example if regular.
No structural identification or causal merger inference is supplied.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--allow-short-sample',action='store_true');args=parser.parse_args()
p=pd.read_csv(ROOT/'data/clean/analysis_panel.csv')
s=p.groupby('outcome_year').agg(hhi=('hhi','mean'),hospital_bad_share=('worse_hospitalization','mean'))
s.to_csv(ROOT/'results/var_annual_inputs.csv')
if not np.all(np.diff(s.index)==1):raise SystemExit('VAR not estimated: clinical years have gaps. Do not interpolate or treat irregular years as annual lags.')
if len(s)<30 and not args.allow_short_sample:raise SystemExit(f'VAR not estimated: only {len(s)} annual observations, below the 30-observation screening minimum. Main results use panel regressions. --allow-short-sample is for a fragile classroom demonstration only; 30 is not a universal sufficiency guarantee.')
z=s.copy();z.hhi/=1000;z.hospital_bad_share*=100
# Level VAR with intercept: y_t = c + A y_(t-1) + u_t. No claim of stationarity.
y=z.to_numpy();X=np.column_stack([np.ones(len(y)-1),y[:-1]]);Y=y[1:]
if len(Y)-X.shape[1]<Y.shape[1]:raise SystemExit('Insufficient residual degrees of freedom.')
B=np.linalg.lstsq(X,Y,rcond=None)[0];U=Y-X@B;Sigma=U.T@U/(len(Y)-X.shape[1]);A=B[1:].T
pd.DataFrame(B,index=['intercept','lag_hhi','lag_hospital_bad_share'],columns=z.columns).to_csv(ROOT/'results/var_coefficients.csv')
roots=np.linalg.eigvals(A);print('Transition eigenvalues:',roots,'Stable:',bool(np.all(np.abs(roots)<1)))
# Ordering is an assumption: HHI first. These are Cholesky responses, NOT merger effects.
P=np.linalg.cholesky(Sigma);horizons=np.arange(11);responses=np.array([np.linalg.matrix_power(A,int(h))@P for h in horizons])
rows=[]
fig,axs=plt.subplots(2,2,figsize=(9,6))
for i in range(2):
 for j in range(2):
  axs[i,j].plot(horizons,responses[:,i,j]);axs[i,j].axhline(0,color='gray',lw=.7);axs[i,j].set_title(f'{z.columns[i]} response to {z.columns[j]} innovation',fontsize=8);axs[i,j].set_xlabel('Years')
  for h in horizons:rows.append(dict(horizon=int(h),response=z.columns[i],innovation=z.columns[j],value=float(responses[h,i,j])))
fig.suptitle('Exploratory VAR: very short sample; no causal interpretation');fig.tight_layout();fig.savefig(ROOT/'results/figures/08_exploratory_var.png');fig.savefig(ROOT/'results/figures/08_exploratory_var.pdf')
pd.DataFrame(rows).to_csv(ROOT/'results/var_irf.csv',index=False)
(ROOT/'results/VAR_WARNING.txt').write_text(f'Only {len(s)} annual observations. No causal identification, stationarity evidence, or reliable uncertainty intervals. Do not use this demonstration as antitrust evidence. Aggregates use a changing facility sample. Eigenvalues: {roots}.')
