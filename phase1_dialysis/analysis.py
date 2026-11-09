"""data, clustered regressions, and every figure from bundled CMS CSVs."""
from pathlib import Path
import re,json,zipfile
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'; CLEAN=ROOT/'data'/'clean'; OUT=ROOT/'results'; FIG=OUT/'figures'
for p in [CLEAN,FIG]:p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
def canonical(s):return re.sub('[^a-z0-9]','',s.lower())
@lru_cache(maxsize=500)
def dates(s):
 parts=re.findall(r'\d{2}/\d{2}/\d{4}|\d{2}[A-Za-z]{3}\d{4}',str(s))
 return tuple(pd.to_datetime(x,format='%m/%d/%Y' if '/' in x else '%d%b%Y') for x in parts) if len(parts)==2 else (pd.NaT,pd.NaT)
def build():
 frames=[];audit=[]
 for p in sorted(RAW.glob('dfc_*.zip')):
  date=p.stem[4:]
  with zipfile.ZipFile(p) as z:
   name=next(n for n in z.namelist() if n=='23ew-n7w9.csv' or n=='DFC_SOCRATA_FAC_DATA.csv' or ('DFC_FACILITY.csv' in n and 'ICH' not in n))
   target=RAW/('facility_'+date+'.csv')
   target.write_bytes(z.read(name))
 for p in sorted(RAW.glob('facility_*.csv')):
  date=p.stem[9:];d=pd.read_csv(p,dtype=str);cols={canonical(c):c for c in d}
  def col(*names):
   for n in names:
    if canonical(n) in cols:return d[cols[canonical(n)]].str.strip()
   return pd.Series('',index=d.index)
  f=pd.DataFrame({'ccn':col('CMS Certification Number (CCN)','Provider Number','Certification Number').str.zfill(6),'snapshot':date,'state':col('State'),'county':col('County/Parish','County').str.upper(),'chain_raw':col('Chain Organization').str.upper(),'chain_owned':col('Chain Owned').str.upper(),'stations':pd.to_numeric(col('# of Dialysis Stations'),errors='coerce'),'incenter':col('Offers in-center hemodialysis').str.upper(),'hospital_category':col('Patient hospitalization category text'),'infection_ratio':pd.to_numeric(col('Standard Infection Ratio'),errors='coerce'),'infection_window':col('SIR Date'),'hospital_window':col('SHR Date'),'readmission_window':col('SRR Date'),'readmission_rate':pd.to_numeric(col('Readmission Rate (Facility)'),errors='coerce')})
  for outcome in ['hospital','infection','readmission']:
   ds=f[outcome+'_window'].apply(dates);f[outcome+'_start']=[x[0] for x in ds];f[outcome+'_end']=[x[1] for x in ds]
  assert f.ccn.str.fullmatch(r'\d{6}').all() and f.ccn.ne('000000').all(),f'Invalid CCNs {date}'
  assert not f.ccn.duplicated().any(),f'Duplicate CCNs {date}'
  f['market']=f.state+'|'+f.county
  f=f[f.incenter.isin(['YES','1']) & f.stations.gt(0) & f.county.ne('') & f.state.isin(['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'])].copy()
  # Retain known reported chains; do not pretend ambiguous OTHER identifies a firm.
  f['owner']=f.chain_raw
  ambiguous=f.owner.isin(['','OTHER','INDEPENDENT']) | f.chain_owned.eq('NO') | f.owner.isna()
  f.loc[ambiguous,'owner']='FACILITY_'+f.loc[ambiguous,'ccn']
  f['ambiguous_owner']=f.chain_raw.isin(['','OTHER'])
  f['legacy_owner']=f.chain_raw.str.contains('GAMBRO|RENAL CARE GROUP|DIVERSIFIED|LIBERTY',regex=True,na=False)
  f['major_chain']=f.chain_raw.isin(['DAVITA','FRESENIUS MEDICAL CARE']).astype(int)
  frames.append(f);audit.append({'snapshot':date,'raw_rows':len(d),'eligible_rows':len(f),'ambiguous_owner':int(f.ambiguous_owner.sum()),'hospital_window':';'.join(f.hospital_window.unique())})
 allf=pd.concat(frames,ignore_index=True).sort_values(['snapshot','ccn']);allf['snapshot']=pd.to_datetime(allf.snapshot)
 markets=[]
 for (snap,market),g in allf.groupby(['snapshot','market']):
  total=g.stations.sum();shares=g.groupby('owner').stations.sum()/total
  markets.append(dict(snapshot=snap,market=market,hhi=float(10000*np.square(shares).sum()),n_facilities=len(g),n_firms=g.owner.nunique(),total_stations=total,major_share=g.loc[g.major_chain.eq(1),'stations'].sum()/total,ambiguous_owner=int(g.ambiguous_owner.sum()),legacy_owner=int(g.legacy_owner.sum()),hhi_facility=float(10000*np.square(g.stations/total).sum())))
 m=pd.DataFrame(markets);ex=allf[['ccn','snapshot','market','major_chain','stations','chain_raw']].merge(m,on=['snapshot','market'],validate='many_to_one')
 # Outcome is categorization based on risk-adjusted hospitalization, available across schemas.
 o=allf[allf.hospital_category.isin(['Worse than Expected','As Expected','Better than Expected']) & allf.hospital_start.notna()].copy()
 o=o.sort_values('snapshot').drop_duplicates(['ccn','hospital_start','hospital_end'],keep='first')
 o['worse_hospitalization']=o.hospital_category.eq('Worse than Expected').astype(int)
 o['outcome_year']=o.hospital_end.dt.year
 # Match latest ownership observation strictly before clinical window starts, max 450 days.
 o=o.sort_values('hospital_start');ex=ex.sort_values('snapshot')
 p=pd.merge_asof(o[['ccn','market','hospital_start','hospital_end','outcome_year','worse_hospitalization']],ex,on=None,left_on='hospital_start',right_on='snapshot',by='ccn',direction='backward',allow_exact_matches=False,tolerance=pd.Timedelta(days=450),suffixes=('_outcome',''))
 p['same_county']=p.market.eq(p.market_outcome)
 p=p[p.snapshot.notna() & p.same_county].copy()
 p['hhi_1000']=p.hhi/1000;p['log_stations']=np.log(p.stations);p['hhi_facility_1000']=p.hhi_facility/1000
 allf.to_csv(CLEAN/'facility_snapshots.csv',index=False);m.to_csv(CLEAN/'county_concentration.csv',index=False);p.to_csv(CLEAN/'analysis_panel.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'source_audit.csv',index=False)
 return allf,m,p

def regress(p,y,xs,name):
 d=p.dropna(subset=[y,*xs,'ccn','market','outcome_year']).copy()
 d=d[d.groupby('ccn').ccn.transform('size')>=2].reset_index(drop=True)
 # Alternate projections give exact two-way FE even for an unbalanced panel.
 a=d[[y,*xs]].astype(float).copy()
 for i in range(1000):
  old=a.to_numpy().copy()
  for group in ['ccn','outcome_year']:a-=a.groupby(d[group]).transform('mean')
  if np.max(np.abs(a.to_numpy()-old))<1e-10:break
 else:raise RuntimeError('FE projection did not converge')
 Y=a[y].to_numpy();X=a[xs].to_numpy();rank=np.linalg.matrix_rank(X)
 if rank<len(xs):raise ValueError('Collinear regressors')
 b=np.linalg.lstsq(X,Y,rcond=None)[0];e=Y-X@b;bread=np.linalg.inv(X.T@X)
 scores=pd.DataFrame(X*e[:,None]).groupby(d.market).sum().to_numpy();G=len(scores);N=len(d);K=len(xs)
 # Count FE degrees of freedom conservatively (disconnected panels may overcount).
 absorbed=d.ccn.nunique()+d.outcome_year.nunique()-1
 cov=bread@(scores.T@scores)@bread*(G/(G-1))*((N-1)/(N-K-absorbed))
 se=np.sqrt(np.diag(cov));crit=stats.t.ppf(.975,G-1)
 rows=[dict(model=name,outcome=y,term=x,coefficient=v,se=s,p_value=2*stats.t.sf(abs(v/s),G-1),ci_lower=v-crit*s,ci_upper=v+crit*s,n=N,facilities=d.ccn.nunique(),county_clusters=G,years=','.join(map(str,sorted(d.outcome_year.unique())))) for x,v,s in zip(xs,b,se)]
 return rows

def figures(f,m,p,r):
 def save(name):plt.tight_layout();plt.savefig(FIG/(name+'.png'));plt.savefig(FIG/(name+'.pdf'));plt.close()
 yearly=m.assign(year=m.snapshot.dt.year).groupby('year').agg(hhi_median=('hhi','median'),hhi_mean=('hhi','mean'),counties=('market','nunique'))
 yearly.to_csv(OUT/'annual_market_summary.csv')
 plt.figure(figsize=(7,4));plt.plot(yearly.index,yearly.hhi_mean,marker='o');plt.ylabel('Mean county reported-owner HHI (0–10,000)');plt.xlabel('Snapshot year');plt.title('Local dialysis concentration: exploratory proxy');save('01_concentration_trend')
 latest=m[m.snapshot.eq(m.snapshot.max())];plt.figure(figsize=(7,4));plt.hist(latest.hhi,bins=np.arange(0,10500,500),color='#256b81',edgecolor='white');plt.xlabel('Reported-owner HHI');plt.ylabel('Counties');plt.title('County concentration in latest ownership snapshot');save('02_concentration_distribution')
 shares=f.assign(year=f.snapshot.dt.year).groupby('year').apply(lambda g:g.loc[g.major_chain.eq(1),'stations'].sum()/g.stations.sum(),include_groups=False)
 plt.figure(figsize=(7,4));plt.plot(shares.index,100*shares.values,marker='o');plt.ylabel('Share of in-center stations (%)');plt.xlabel('Snapshot year');plt.title('DaVita + Fresenius share under reported labels');save('03_two_chain_share')
 q=p.groupby('outcome_year').worse_hospitalization.agg(['mean','count']);q.to_csv(OUT/'outcome_summary.csv');plt.figure(figsize=(7,4));plt.plot(q.index,100*q['mean'],marker='o');plt.ylabel('Facilities rated worse than expected (%)');plt.xlabel('Clinical outcome year');plt.title('Hospitalization quality in matched analysis sample');save('04_quality_trend')
 b=p.copy();b['bin']=pd.cut(b.hhi,bins=[0,2500,5000,7500,10001],include_lowest=True);g=b.groupby('bin',observed=True).worse_hospitalization.agg(['mean','count','std']);plt.figure(figsize=(7,4));plt.bar([str(x) for x in g.index],100*g['mean'],color='#256b81');plt.ylabel('Worse-than-expected hospitalization (%)');plt.xlabel('Reported-owner HHI bin');plt.title('Unadjusted association; no causal interpretation');save('05_quality_by_concentration')
 a=r[r.term.eq('hhi_1000')].reset_index(drop=True);plt.figure(figsize=(9,4.5));plt.errorbar(a.coefficient*100,np.arange(len(a)),xerr=np.vstack([(a.coefficient-a.ci_lower)*100,(a.ci_upper-a.coefficient)*100]),fmt='o',capsize=4);plt.yticks(range(len(a)),a.model);plt.axvline(0,color='gray',linestyle='--');plt.xlabel('Percentage-point change per 1,000 HHI points (95% CI)');plt.title('Facility + clinical-year FE; county-clustered uncertainty');save('06_regression_coefficients')
 counts=p.groupby('outcome_year').ccn.nunique();plt.figure(figsize=(7,4));plt.bar(counts.index.astype(str),counts.values,color='#256b81');plt.ylabel('Matched facilities');plt.xlabel('Clinical outcome year');plt.title('Data coverage after ownership timing checks');save('07_sample_coverage')
if __name__=='__main__':
 f,m,p=build();rows=[]
 rows+=regress(p,'worse_hospitalization',['hhi_1000','major_chain','log_stations'],'Main: all eligible counties')
 rows+=regress(p[p.n_facilities.ge(2)],'worse_hospitalization',['hhi_1000','major_chain','log_stations'],'Sensitivity: 2+ facilities')
 rows+=regress(p[p.ambiguous_owner.eq(0)],'worse_hospitalization',['hhi_1000','major_chain','log_stations'],'Sensitivity: no OTHER labels')
 rows+=regress(p[p.ambiguous_owner.eq(0) & p.legacy_owner.eq(0)],'worse_hospitalization',['hhi_1000','major_chain','log_stations'],'Sensitivity: no legacy/OTHER labels')
 rows+=regress(p[p.outcome_year.ne(2020)],'worse_hospitalization',['hhi_1000','major_chain','log_stations'],'Sensitivity: exclude 2020')
 rows+=regress(p,'worse_hospitalization',['hhi_facility_1000','major_chain','log_stations'],'Facility fragmentation benchmark')
 r=pd.DataFrame(rows);r.to_csv(OUT/'regression_results.csv',index=False);figures(f,m,p,r)
 print(r.to_string(index=False));print('Panel rows',len(p),'unique facilities',p.ccn.nunique(),'outcome years',sorted(p.outcome_year.unique()))
