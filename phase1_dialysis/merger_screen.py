"""compute a PROPOSED merger's mechanical concentration change; no quality causality.
Example only, not an announced deal:
python merger_screen.py --buyer DAVITA --target 'FRESENIUS MEDICAL CARE'
Resolve ultimate ownership and actual transaction parties before empirical use.
"""
from pathlib import Path
import argparse
import pandas as pd
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--buyer',required=True);p.add_argument('--target',required=True);a=p.parse_args()
if a.buyer.upper()==a.target.upper():raise SystemExit('Buyer and target must differ.')
d=pd.read_csv(ROOT/'data/clean/facility_snapshots.csv',dtype={'ccn':str});d=d[d.snapshot.eq(d.snapshot.max())]
rows=[]
for market,g in d.groupby('market'):
 shares=g.groupby('owner').stations.sum()/g.stations.sum()
 sa=shares.get(a.buyer.upper(),0);sb=shares.get(a.target.upper(),0)
 if sa<=0 or sb<=0:continue
 before=10000*(shares**2).sum();delta=20000*sa*sb
 rows.append(dict(market=market,buyer_share=sa,target_share=sb,pre_hhi=before,delta_hhi=delta,post_hhi=before+delta,stations=g.stations.sum(),facilities=len(g)))
out=ROOT/'results/proposed_merger_screen.csv';pd.DataFrame(rows).to_csv(out,index=False);print('Pro forma overlap counties:',len(rows),'Saved:',out)
