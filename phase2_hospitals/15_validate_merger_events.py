#!/usr/bin/env python3
"""Step 15: validate CMS CHOW matches and assess event-study feasibility.

Run from phase2_hospitals: python3 15_validate_merger_events.py
Uses existing Step 14 CSVs and hospital_panel.csv. Does not download data,
modify earlier outputs, or label CMS CHOWs as verified corporate mergers.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SOURCE = DATA / 'results' / 'merger_registry'
OUT = DATA / 'results' / 'merger_validation'
OUT.mkdir(parents=True, exist_ok=True)

def load(name, **kwargs):
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run Step 14 first.')
    return pd.read_csv(path, low_memory=False, **kwargs)

def normalize_ccn(s):
    return s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(6)

def yn(s):
    return s.astype('string').str.strip().str.lower().isin(['true','1','yes'])

all_records = load('cms_chow_all_records.csv', dtype={'buyer_ccn':'string','seller_ccn':'string'})
events = load('cms_chow_matched_facility_events.csv', dtype={'facility_id':'string','buyer_ccn':'string','seller_ccn':'string'})
panel = pd.read_csv(DATA / 'processed' / 'hospital_panel.csv', dtype={'facility_id':'string'}, low_memory=False)
geo = pd.read_csv(DATA / 'spatial' / 'hospital_cluster_rucc_panel.csv', dtype={'facility_id':'string'}, low_memory=False)
for frame in [events,panel,geo]:
    frame['facility_id'] = normalize_ccn(frame['facility_id'])
for frame in [panel,geo]:
    frame['year'] = pd.to_numeric(frame['year'],errors='coerce')

required = ['cms_record_id','facility_id','matched_side','event_year','cms_chow_type','rural_ever_in_panel','n_pre_years','n_post_years','n_pre_rated_years','n_post_rated_years']
missing = [c for c in required if c not in events.columns]
if missing:
    raise ValueError(f'Step 14 events missing columns: {missing}')

panel = panel.sort_values(['facility_id','year','snapshot'] if 'snapshot' in panel else ['facility_id','year']).drop_duplicates(['facility_id','year'],keep='last')
if 'eps_km' in geo:
    geo['eps_km'] = pd.to_numeric(geo['eps_km'],errors='coerce')
    geo['_radius_deviation'] = (geo['eps_km'] - 10).abs()
    geo = geo.sort_values(['facility_id','year','_radius_deviation'])
geo = geo.drop_duplicates(['facility_id','year'],keep='first')
if 'rural' not in geo:
    raise ValueError('Geographic panel has no rural classification.')
geo = geo[['facility_id','year','rural']].rename(columns={'rural':'rural_geo'})
panel = panel.merge(geo,on=['facility_id','year'],how='left',validate='one_to_one')
panel['rural_flag'] = pd.to_numeric(panel['rural_geo'],errors='coerce')
if 'rural' in panel:
    panel['rural_flag'] = panel['rural_flag'].combine_first(pd.to_numeric(panel['rural'],errors='coerce'))
panel['rating'] = pd.to_numeric(panel['hospital_overall_rating'],errors='coerce')
panel['year'] = panel['year'].astype('Int64')

# One facility per CMS record, even if the buyer and seller CCNs are identical.
events['event_year'] = pd.to_numeric(events['event_year'],errors='coerce').astype('Int64')
events['rural_ever_in_panel'] = yn(events['rural_ever_in_panel'])
events['matched_sides'] = events.groupby(['cms_record_id','facility_id'])['matched_side'].transform(lambda x: '+'.join(sorted(set(x.astype(str)))))
events['same_ccn_buyer_seller'] = normalize_ccn(events['buyer_ccn']).eq(normalize_ccn(events['seller_ccn']))
events['both_sides_matched_same_facility'] = events['matched_sides'].eq('buyer+seller')
events = events.sort_values(['cms_record_id','facility_id','matched_side']).drop_duplicates(['cms_record_id','facility_id'],keep='first').copy()

# Classify using the CMS label, not an inferred horizontal-merger label.
events['cms_acquisition_merger_label'] = events['cms_chow_type'].astype('string').str.upper().eq('ACQUISITION/MERGER')

# Assess rural status at event year; if missing, closest observed year (tie favors pre-event).
# Nearest-year rural status is a proxy, not verified contemporaneous rural status.
panel_groups = {fid: group.sort_values('year') for fid,group in panel.groupby('facility_id')}
assessment=[]
for e in events.itertuples(index=False):
    fid=e.facility_id
    year=e.event_year
    obs=panel_groups.get(fid)
    r={'cms_record_id':e.cms_record_id,'facility_id':fid}
    if obs is None or len(obs)==0 or pd.isna(year):
        r.update(rural_at_event_year=pd.NA,rural_nearest_year=pd.NA,rural_reference_year=pd.NA,rural_year_gap=pd.NA,
                 pre_panel_years=0,post_panel_years=0,pre_rated_years=0,post_rated_years=0,
                 rating_pre_2yr=False,rating_post_2yr=False)
    else:
        year=int(year)
        obs=obs.copy()
        obs['_abs_gap']=(obs['year'].astype(float)-year).abs()
        obs['_future']=(obs['year']>year).astype(int)
        nearest=obs.sort_values(['_abs_gap','_future','year']).iloc[0]
        exact=obs.loc[obs['year'].eq(year)]
        exact_rural=exact['rural_flag'].iloc[0] if len(exact) else np.nan
        pre=obs.loc[obs['year'].lt(year)]
        post=obs.loc[obs['year'].gt(year)]
        pre_rated=pre.loc[pre['rating'].notna()]
        post_rated=post.loc[post['rating'].notna()]
        r.update(rural_at_event_year=(bool(exact_rural==1) if pd.notna(exact_rural) else pd.NA),
                 rural_nearest_year=(bool(nearest['rural_flag']==1) if pd.notna(nearest['rural_flag']) else pd.NA),
                 rural_reference_year=int(nearest['year']),rural_year_gap=int(nearest['_abs_gap']),
                 pre_panel_years=int(pre['year'].nunique()),post_panel_years=int(post['year'].nunique()),
                 pre_rated_years=int(pre_rated['year'].nunique()),post_rated_years=int(post_rated['year'].nunique()),
                 rating_pre_2yr=bool(pre_rated['year'].nunique()>=2),rating_post_2yr=bool(post_rated['year'].nunique()>=2))
    assessment.append(r)
assessed=pd.DataFrame(assessment)
events=events.merge(assessed,on=['cms_record_id','facility_id'],how='left',validate='one_to_one')
events['panel_2plus_both']=events['pre_panel_years'].ge(2)&events['post_panel_years'].ge(2)
events['rating_2plus_both']=events['pre_rated_years'].ge(2)&events['post_rated_years'].ge(2)
events['rural_at_event_confirmed_from_panel']=events['rural_at_event_year'].eq(True).fillna(False)
events['rural_nearest_proxy']=events['rural_nearest_year'].eq(True).fillna(False)
events['rural_event_or_proxy']=events['rural_at_event_confirmed_from_panel'] | (events['rural_at_event_year'].isna() & events['rural_nearest_proxy'])
events['rural_classification_basis']=np.select(
    [events['rural_at_event_year'].notna(),events['rural_nearest_year'].notna()],
    ['observed_event_year','nearest_observed_year_proxy'],default='missing')

# Multiple CMS events at the same facility complicate a simple first-treatment design.
event_counts=events.groupby('facility_id')['cms_record_id'].nunique()
events['n_distinct_cms_records_for_facility']=events['facility_id'].map(event_counts)
events['multiple_reported_events_for_facility']=events['n_distinct_cms_records_for_facility'].gt(1)
events['event_year_in_panel_range']=events['event_year'].between(2019,2026)
# This is a conservative screening subset, NOT an identified causal sample.
events['candidate_for_rating_event_study']=(events['rural_at_event_confirmed_from_panel'] &
    events['rating_2plus_both'] & events['event_year_in_panel_range'] &
    ~events['multiple_reported_events_for_facility'])

# One row per CMS record with rural-event and feasibility indicators.
summary=events.groupby('cms_record_id',dropna=False).agg(
    distinct_panel_facilities=('facility_id','nunique'),
    any_rural_ever=('rural_ever_in_panel','any'),
    any_rural_event_year=('rural_at_event_confirmed_from_panel','any'),
    any_rural_nearest_proxy=('rural_nearest_proxy','any'),
    any_rural_panel_2plus=('panel_2plus_both',lambda x: bool((x & events.loc[x.index,'rural_event_or_proxy']).any())),
    any_rural_rated_2plus=('rating_2plus_both',lambda x: bool((x & events.loc[x.index,'rural_event_or_proxy']).any())),
    any_conservative_rating_candidate=('candidate_for_rating_event_study','any'),
    cms_acquisition_merger_label=('cms_acquisition_merger_label','first'),
    event_year=('event_year','first'),cms_chow_type=('cms_chow_type','first'),
).reset_index()
summary['event_requires_parent_transaction_validation']=True

# Preserve source names and dates for independent verification.
all_info=all_records[['cms_record_id','ORGANIZATION NAME - BUYER','ORGANIZATION NAME - SELLER','EFFECTIVE DATE']].drop_duplicates('cms_record_id')
summary=summary.merge(all_info,on='cms_record_id',how='left',validate='one_to_one')

# Detailed output and human-review shortlist.
events.to_csv(OUT/'facility_event_validation.csv',index=False)
summary.to_csv(OUT/'cms_transaction_validation.csv',index=False)
review=events.loc[events['rural_event_or_proxy']].copy()
review=review.sort_values(['candidate_for_rating_event_study','cms_acquisition_merger_label','rating_2plus_both'],ascending=False)
review.to_csv(OUT/'rural_transactions_for_manual_review.csv',index=False)

lines=[
 'STEP 15: CMS HOSPITAL CHOW VALIDATION AND OUTCOME FEASIBILITY',
 'CMS events are reported ownership transactions, NOT independently verified horizontal mergers.',
 f'Original matched facility-event sides: {len(load("cms_chow_matched_facility_events.csv")):,}',
 f'Unique CMS-record × facility pairs (deduplicated buyer/seller sides): {len(events):,}',
 f'Unique CMS records matched to panel: {summary.cms_record_id.nunique():,}',
 f'CMS records involving a rural-ever hospital: {int(summary.any_rural_ever.sum()):,}',
 f'CMS records involving a rural hospital in observed event year: {int(summary.any_rural_event_year.sum()):,}',
 f'CMS records with rural nearest-year proxy (includes exact-year): {int(summary.any_rural_nearest_proxy.sum()):,}',
 f'CMS records involving rural event/proxy hospital with >=2 pre and >=2 post panel years: {int(summary.any_rural_panel_2plus.sum()):,}',
 f'CMS records involving rural event/proxy hospital with >=2 pre and >=2 post rated years: {int(summary.any_rural_rated_2plus.sum()):,}',
 f'CMS records with >=1 conservative rating candidate: {int(summary.any_conservative_rating_candidate.sum()):,}',
 f'Unique facilities with conservative rating candidate: {events.loc[events.candidate_for_rating_event_study,"facility_id"].nunique():,}',
 f'Unique CMS records labeled ACQUISITION/MERGER among rural-ever: {int((summary.cms_acquisition_merger_label & summary.any_rural_ever).sum()):,}',
 f'Facility-event pairs with multiple CMS records for same facility: {int(events.multiple_reported_events_for_facility.sum()):,}',
 '',
 'IMPORTANT LIMITATIONS:',
 '- Event-year rural flag is from the hospital panel; nearest-year rural flag is a proxy.',
 '- Same CCN on buyer and seller sides is counted as one facility-event, not two.',
 '- CMS acquisition/merger labels do not verify horizontal overlap or ultimate parent.',
 '- 2+ observed rated years are not proof of comparable measurement periods.',
 '- Excluding the transaction year avoids ambiguity but does not align outcome reporting windows.',
 '- Repeated CMS events and CCN changes require manual resolution before causal modeling.',
 '- Conservative rating candidate is a SCREEN, not an identified treatment or causal sample.',
 '',
 'NEXT: Validate transaction identity and historical parent ownership; then collect aligned longitudinal outcomes.'
]
(OUT/'README_step15.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
print(f'\nOutputs: {OUT}')
