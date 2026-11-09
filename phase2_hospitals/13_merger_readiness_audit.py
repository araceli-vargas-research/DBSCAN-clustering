#!/usr/bin/env python3
"""Step 13B: audit whether the existing rural hospital panel supports merger research.

Place alongside 13_county_ses_controls.py in phase2_hospitals/ and run:
    python3 13_merger_readiness_audit.py

No downloads or fabricated merger dates. Ownership-category changes are flagged as
UNVERIFIED candidates, never treated as confirmed mergers.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'results' / 'merger_readiness'
OUT.mkdir(parents=True, exist_ok=True)

def read(rel, dtype=None):
    p = DATA / rel
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. Run earlier pipeline steps first.')
    return pd.read_csv(p, dtype=dtype, low_memory=False)

def key(s):
    x = s.astype("string").str.strip()
    x = x.str.replace(r"\.0$", "", regex=True)
    return x.str.zfill(6)

def pct(s):
    return round(float(s.notna().mean()*100), 1)

panel = read('processed/hospital_panel.csv', {'facility_id':'string'})
geo = read('spatial/hospital_cluster_rucc_panel.csv', {'facility_id':'string','county_fips':'string'})
sys = read('spatial/hospital_2023_system_linkage.csv', {'facility_id':'string'})
panel['facility_id'] = key(panel['facility_id'])
geo['facility_id'] = key(geo['facility_id'])
sys['facility_id'] = key(sys['facility_id'])
for d in (panel,geo,sys):
    d['year'] = pd.to_numeric(d['year'], errors='coerce').astype('Int64')

# A single canonical record per facility/year; report duplicate conflicts.
dup = panel[panel.duplicated(['facility_id','year'], keep=False)].copy()
dup.to_csv(OUT/'duplicate_facility_year_rows.csv',index=False)
panel = panel.sort_values(['facility_id','year','snapshot']).drop_duplicates(['facility_id','year'],keep='last')
geo = geo.sort_values(['facility_id','year','eps_km'] if 'eps_km' in geo else ['facility_id','year']).drop_duplicates(['facility_id','year'],keep='first')

rural_cols = [c for c in ['facility_id','year','rural','rucc_2023','county_fips','latitude','longitude','cluster_id','nearest_competing_cluster_km','eps_km'] if c in geo]
df = panel.merge(geo[rural_cols],on=['facility_id','year'],how='left',validate='one_to_one')
df['rural_num'] = pd.to_numeric(df['rural'],errors='coerce')
df['is_rural'] = df['rural_num'].eq(1)
df['overall_rating_numeric'] = pd.to_numeric(df.get('hospital_overall_rating'),errors='coerce')

# Ownership category is not a corporate parent and does not identify an acquisition.
df['ownership_category'] = df['hospital_ownership'].astype('string').str.strip().str.lower()
df['prior_ownership_category'] = df.groupby('facility_id')['ownership_category'].shift()
df['prior_observed_year'] = df.groupby('facility_id')['year'].shift()
df['category_changed'] = (df['prior_ownership_category'].notna() & df['ownership_category'].notna() & (df['prior_ownership_category'] != df['ownership_category']))
candidates = df.loc[df['category_changed'] & df['is_rural'],['facility_id','facility_name','year','prior_observed_year','state','county','county_fips','prior_ownership_category','ownership_category','rucc_2023']].copy()
candidates['verification_status'] = 'UNVERIFIED: category change is not evidence of merger'
candidates.to_csv(OUT/'rural_ownership_category_changes_UNVERIFIED.csv',index=False)

# Panel coverage and quality availability by year and rural status.
summary = df.groupby(['year','is_rural'],dropna=False).agg(facilities=('facility_id','nunique'),rows=('facility_id','size'),rating_observed=('overall_rating_numeric','count'),geocoded=('latitude','count'),county_fips_observed=('county_fips','count')).reset_index()
summary['rating_coverage_pct'] = (100*summary['rating_observed']/summary['rows']).round(1)
summary.to_csv(OUT/'year_rural_coverage.csv',index=False)

# Pre/post potential for ANY future verified event (not actual event count).
counts = df.groupby('facility_id').agg(first_year=('year','min'),last_year=('year','max'),observed_years=('year','nunique'),rural_any=('is_rural','max'),rated_years=('overall_rating_numeric','count')).reset_index()
counts['span_years'] = counts['last_year'] - counts['first_year'] + 1
counts['complete_span'] = counts['observed_years'] == counts['span_years']
counts.to_csv(OUT/'facility_panel_coverage.csv',index=False)

# Compare availability of 2023 parent-system linkage; only one-year snapshot.
sys23 = sys[sys['year'].eq(2023)].drop_duplicates('facility_id')
syscols = [c for c in ['facility_id','health_sys_id','health_sys_name','corp_parent_id','corp_parent_name','hos_beds','firm_id','parent_firm_id'] if c in sys23]
joined = df[df['year'].eq(2023)].merge(sys23[syscols],on='facility_id',how='left',validate='one_to_one')
joined.to_csv(OUT/'hospital_2023_system_linkage_audit.csv',index=False)

# Long-format variable inventory: present and observed coverage, not presumed longitudinal consistency.
inv=[]
for name in ['hospital_overall_rating','hospital_ownership','hospital_type','rural','rucc_2023','county_fips','latitude','longitude','nearest_competing_cluster_km']:
    if name in df:
        for year,sub in df.groupby('year'):
            inv.append({'variable':name,'year':int(year),'nonmissing':int(sub[name].notna().sum()),'total':len(sub),'coverage_pct':pct(sub[name])})
pd.DataFrame(inv).to_csv(OUT/'variable_inventory_by_year.csv',index=False)

rural = df[df['is_rural']]
lines=[
 'RURAL HEALTHCARE MERGER RESEARCH: STEP 13B READINESS AUDIT',
 'IMPORTANT: This dataset is a hospital snapshot panel, NOT a verified merger-event registry.',
 f'Years: {int(df.year.min())}–{int(df.year.max())}',
 f'Unique hospital IDs (all): {df.facility_id.nunique():,}',
 f'Unique rural hospital IDs (rural in at least one observed year): {rural.facility_id.nunique():,}',
 f'Facility-year rows: {len(df):,}; rural facility-year rows: {len(rural):,}',
 f'Rural ownership-category changes (UNVERIFIED): {len(candidates):,}',
 f'Rural hospital-year rating coverage: {pct(rural.overall_rating_numeric)}%',
 f'Rural facilities with >= 5 observed years: {int((counts.rural_any & (counts.observed_years>=5)).sum()):,}',
 f'Rural facilities with >= 7 observed years: {int((counts.rural_any & (counts.observed_years>=7)).sum()):,}',
 f'2023 facilities linked to parent firm ID: {int(joined.parent_firm_id.notna().sum()) if "parent_firm_id" in joined else "column unavailable"}',
 '',
 'MISSING FOR A CREDIBLE PRE/POST MERGER ANALYSIS:',
 '1. Verified event registry: acquirer, target, effective date, transaction type, source.',
 '2. Historical ultimate-parent ownership by year; 2023 system linkage alone is insufficient.',
 '3. Repeated sector-specific outcomes (HCAHPS here is 2023 only; not a longitudinal survey).',
 '4. Staff FTE/hours and patient-volume denominators (not in current panel).',
 '5. Rural dialysis/physician/SNF sector facility registries if studying beyond hospitals.',
 '6. Actual patient travel-time measures if interpreting geographic isolation as access.',
 '7. Stable outcome measurement windows and crosswalks for CCN changes/closures.',
 '',
 'CAUTION: annual CMS snapshots are not necessarily calendar-year outcome measurements.',
 'CAUTION: changes in hospital_ownership categories are NOT verified acquisitions.',
 'CAUTION: this script does not estimate treatment effects or infer causal merger impacts.',
]
(OUT/'README_audit.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
print(f'\nOutputs: {OUT}')
