#!/usr/bin/env python3
"""Step 14: Match CMS Hospital CHOW records to the existing rural hospital panel.

Place in phase2_hospitals/ alongside Step 13B. Put the original CMS ZIP
(or its extracted CSV) anywhere under data/raw/mergers/. Then run:
    python3 14_build_cms_chow_registry.py

This is a *reported transaction* registry, not a verified ultimate-parent
merger registry or a causal analysis. No network requests; no source changes.
"""
from pathlib import Path
from io import BytesIO
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW = DATA / 'raw' / 'mergers'
OUT = DATA / 'results' / 'merger_registry'
OUT.mkdir(parents=True, exist_ok=True)

def ccn(s):
    return (s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
            .str.replace(r'\s+', '', regex=True).str.zfill(6))

def load_chow():
    csvs = sorted(p for p in RAW.rglob('Hospital_CHOW_*.csv') if 'NPIs' not in p.name and '__MACOSX' not in str(p))
    if csvs:
        p = csvs[-1]
        return pd.read_csv(p, dtype='string', encoding='cp1252'), str(p)
    for zpath in sorted(RAW.rglob('*.zip'), reverse=True):
        with zipfile.ZipFile(zpath) as z:
            names = [n for n in z.namelist() if Path(n).name.startswith('Hospital_CHOW_') and n.endswith('.csv') and 'NPIs' not in n and '__MACOSX' not in n]
            if names:
                name = sorted(names)[-1]
                return pd.read_csv(BytesIO(z.read(name)), dtype='string', encoding='cp1252'), f'{zpath}::{name}'
    raise FileNotFoundError(f'Put Hospital Change of Ownership.zip or Hospital_CHOW_*.csv under {RAW}')

chow, source = load_chow()
required = ['CCN - BUYER','CCN - SELLER','EFFECTIVE DATE','CHOW TYPE TEXT',
            'ORGANIZATION NAME - BUYER','ORGANIZATION NAME - SELLER']
missing = [c for c in required if c not in chow]
if missing:
    raise ValueError(f'CMS file missing expected columns: {missing}')
chow = chow.copy()
chow['cms_record_id'] = ['CMS_CHOW_%05d' % (i+1) for i in range(len(chow))]
chow['buyer_ccn'] = ccn(chow['CCN - BUYER'])
chow['seller_ccn'] = ccn(chow['CCN - SELLER'])
chow['effective_date'] = pd.to_datetime(chow['EFFECTIVE DATE'], errors='coerce', format='%m/%d/%Y')
chow['event_year'] = chow['effective_date'].dt.year.astype('Int64')

panel = pd.read_csv(DATA/'processed'/'hospital_panel.csv', dtype={'facility_id':'string'}, low_memory=False)
geo = pd.read_csv(DATA/'spatial'/'hospital_cluster_rucc_panel.csv', dtype={'facility_id':'string'}, low_memory=False)
for d in (panel,geo):
    d['facility_id'] = ccn(d['facility_id'])
    d['year'] = pd.to_numeric(d['year'], errors='coerce').astype('Int64')
panel = panel.sort_values(['facility_id','year','snapshot']).drop_duplicates(['facility_id','year'], keep='last')
if 'eps_km' in geo:
    geo['eps_km'] = pd.to_numeric(geo['eps_km'], errors='coerce')
    geo['preferred_radius'] = (geo['eps_km']-10).abs()
    geo = geo.sort_values(['facility_id','year','preferred_radius'])
geo = geo.drop_duplicates(['facility_id','year'], keep='first')
cols = [c for c in ['facility_id','year','rural','rucc_2023','county_fips','latitude','longitude'] if c in geo]
panel = panel.merge(geo[cols], on=['facility_id','year'], how='left', validate='one_to_one', suffixes=('','_geo'))
if 'rural_geo' in panel:
    panel['rural'] = panel['rural_geo'].combine_first(panel['rural']) if 'rural' in panel else panel['rural_geo']
panel['is_rural'] = pd.to_numeric(panel['rural'], errors='coerce').eq(1)
panel['rated'] = pd.to_numeric(panel['hospital_overall_rating'], errors='coerce').notna()

# Match BOTH buyer and seller CCNs; they may differ in the CMS transaction.
facilities = set(panel['facility_id'].dropna())
chow['buyer_ccn_in_panel'] = chow['buyer_ccn'].isin(facilities)
chow['seller_ccn_in_panel'] = chow['seller_ccn'].isin(facilities)
chow['either_ccn_in_panel'] = chow['buyer_ccn_in_panel'] | chow['seller_ccn_in_panel']

# For each matched side, retain an auditable facility-event row.
rows=[]
for _, event in chow.iterrows():
    for side in ('buyer','seller'):
        fid = event[f'{side}_ccn']
        if pd.isna(fid) or fid not in facilities:
            continue
        obs = panel.loc[panel['facility_id'].eq(fid)].copy()
        year = event['event_year']
        # Prefer last observed year on/before event; otherwise first after.
        if pd.notna(year):
            before = obs[obs['year'].le(year)]
            selected = before.iloc[-1] if len(before) else obs.iloc[0]
            pre = obs['year'].lt(year)
            post = obs['year'].gt(year)
        else:
            selected = obs.iloc[-1]
            pre = pd.Series(False,index=obs.index)
            post = pd.Series(False,index=obs.index)
        r = {'cms_record_id':event['cms_record_id'], 'matched_side':side,
             'facility_id':fid, 'event_date':event['effective_date'], 'event_year':year,
             'cms_chow_type':event['CHOW TYPE TEXT'],
             'buyer_name':event['ORGANIZATION NAME - BUYER'],
             'seller_name':event['ORGANIZATION NAME - SELLER'],
             'buyer_ccn':event['buyer_ccn'], 'seller_ccn':event['seller_ccn'],
             'facility_name':selected.get('facility_name'), 'state':selected.get('state'),
             'rural_at_reference_year':selected['is_rural'],
             'rural_reference_year':selected['year'],
             'rural_ever_in_panel':bool(obs['is_rural'].any()),
             'first_panel_year':obs['year'].min(), 'last_panel_year':obs['year'].max(),
             'n_pre_years':int(pre.sum()),'n_post_years':int(post.sum()),
             'n_pre_rated_years':int(obs.loc[pre,'rated'].sum()),
             'n_post_rated_years':int(obs.loc[post,'rated'].sum()),
             'source_file':source,
             'verification_status':'CMS_REPORTED_CHOW_NOT_VERIFIED_ULTIMATE_PARENT_MERGER'}
        rows.append(r)
matched = pd.DataFrame(rows)
chow.to_csv(OUT/'cms_chow_all_records.csv',index=False)
if not matched.empty:
    matched = matched.sort_values(['event_date','cms_record_id','matched_side'])
    matched.to_csv(OUT/'cms_chow_matched_facility_events.csv',index=False)
    rural = matched[matched['rural_ever_in_panel']].copy()
    rural.to_csv(OUT/'cms_chow_rural_facility_events.csv',index=False)
    rural[(rural['n_pre_years']>=2)&(rural['n_post_years']>=2)].to_csv(OUT/'rural_events_with_2plus_pre_post_panel_years.csv',index=False)
else:
    rural = matched
    print('WARNING: No buyer or seller CCNs matched your hospital panel. Check ID conventions.')

report = [
 'STEP 14: CMS HOSPITAL CHANGE-OF-OWNERSHIP MATCHING',
 f'CMS source: {source}',
 f'CMS transaction records: {len(chow):,}',
 f'CMS type counts: {chow["CHOW TYPE TEXT"].value_counts(dropna=False).to_dict()}',
 f'CMS event date range: {chow["effective_date"].min()} to {chow["effective_date"].max()}',
 f'Records matching buyer or seller CCN in panel: {int(chow["either_ccn_in_panel"].sum()):,}',
 f'Matched facility-event sides (buyer/seller, not unique transactions): {len(matched):,}',
 f'Matched rural facility-event sides (rural at least once): {len(rural):,}',
 f'Unique matched CMS transactions involving a rural panel facility: {rural["cms_record_id"].nunique() if len(rural) else 0:,}',
 f'Rural facility-event sides with >=2 pre and >=2 post panel years: {int(((rural["n_pre_years"]>=2)&(rural["n_post_years"]>=2)).sum()) if len(rural) else 0:,}',
 'WARNING: Buyer and seller can each match, so facility-event rows are not distinct transactions.',
 'WARNING: CMS CHOW is not necessarily a horizontal merger or ultimate-parent change.',
 'WARNING: Rural-ever status is not necessarily rural status at transaction date.',
 'WARNING: 2026 data and annual ratings may not be complete or aligned to event timing.',
 'NEXT: review matched records, confirm parent-level deal identity, assess outcome windows.'
]
(OUT/'README_step14.txt').write_text('\n'.join(report)+'\n')
print('\n'.join(report))
print(f'\nOutputs: {OUT}')
