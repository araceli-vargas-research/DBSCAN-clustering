#!/usr/bin/env python3
"""Step 16: create a reviewable CMS transaction verification queue.

Run from phase2_hospitals: python3 16_build_transaction_verification_queue.py
No network downloads. No claim that a CMS CHOW is a confirmed horizontal merger.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
VALID = ROOT / 'data/results/merger_validation'
AUDIT = ROOT / 'data/results/merger_readiness'
OUT = ROOT / 'data/results/transaction_verification'
OUT.mkdir(parents=True, exist_ok=True)

def load(folder, name, **kwargs):
    path = folder / name
    if not path.exists():
        raise SystemExit(f'Missing {path}. Run Steps 13B and 15 first.')
    return pd.read_csv(path, low_memory=False, **kwargs)

def truth(s):
    return s.astype('string').str.strip().str.lower().isin(['true','1','yes'])

def ccn(s):
    return s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(6)

fac = load(VALID, 'facility_event_validation.csv', dtype={'facility_id':'string'})
txn = load(VALID, 'cms_transaction_validation.csv')
link = load(AUDIT, 'hospital_2023_system_linkage_audit.csv', dtype={'facility_id':'string'})
fac['facility_id'] = ccn(fac['facility_id'])
link['facility_id'] = ccn(link['facility_id'])

# Parent identifiers in this dataset are 2023 cross-sectional context only.
parent_cols = ['facility_id', 'health_sys_id', 'health_sys_name', 'corp_parent_id',
               'corp_parent_name', 'firm_id', 'parent_firm_id', 'hos_beds']
parent_cols = [x for x in parent_cols if x in link.columns]
link = link[parent_cols].drop_duplicates('facility_id', keep='first')
fac = fac.merge(link, on='facility_id', how='left', validate='many_to_one')

# Keep one row per distinct CMS record and facility, already deduplicated in Step 15.
assert not fac.duplicated(['cms_record_id','facility_id']).any(), 'Step 15 output contains duplicate CMS-record/facility pairs'
for col in ['rural_at_event_confirmed_from_panel', 'rural_nearest_proxy',
            'panel_2plus_both', 'rating_2plus_both', 'candidate_for_rating_event_study',
            'cms_acquisition_merger_label', 'multiple_reported_events_for_facility']:
    fac[col + '_bool'] = truth(fac[col]) if col in fac else False

fac['review_priority'] = 'C - other matched transaction'
fac.loc[fac['rural_nearest_proxy_bool'] | fac['rural_at_event_confirmed_from_panel_bool'], 'review_priority'] = 'B - rural candidate'
fac.loc[fac['candidate_for_rating_event_study_bool'], 'review_priority'] = 'A - rating event-study candidate'
fac.loc[fac['cms_acquisition_merger_label_bool'] & (fac['rural_nearest_proxy_bool'] | fac['rural_at_event_confirmed_from_panel_bool']), 'review_priority'] = 'A - CMS acquisition/merger label'

fac['historical_parent_verified'] = False
fac['horizontal_merger_verified'] = False
fac['independent_competitor_lost_verified'] = False
fac['transaction_verification_status'] = 'PENDING_EXTERNAL_VERIFICATION'
fac['required_review'] = 'Verify deal identity and date; seller/buyer ultimate parents before and after; transaction type; affected CCNs; sources'
fac['research_note'] = '2023 parent information is contextual only; never infer historical parent from it'
fac['source_url_1'] = ''
fac['source_url_2'] = ''
fac['reviewer_decision'] = ''
fac['reviewer_notes'] = ''
fac['reviewed_on'] = ''

# Human-review CSV. Do not auto-mark any merger verified.
order = ['review_priority','cms_record_id','facility_id','facility_name','state',
         'event_date','cms_chow_type','buyer_name','seller_name','buyer_ccn','seller_ccn',
         'rural_at_event_year','rural_nearest_year','rural_year_gap',
         'pre_panel_years','post_panel_years','pre_rated_years','post_rated_years',
         'candidate_for_rating_event_study','multiple_reported_events_for_facility',
         'health_sys_name','corp_parent_name','parent_firm_id','historical_parent_verified',
         'horizontal_merger_verified','independent_competitor_lost_verified',
         'transaction_verification_status','required_review','research_note',
         'source_url_1','source_url_2','reviewer_decision','reviewer_notes','reviewed_on']
order = [c for c in order if c in fac]
queue = fac.sort_values(['review_priority','event_date','cms_record_id','facility_id'], na_position='last')
queue[order].to_csv(OUT/'facility_event_review_queue.csv', index=False)

# Distinct transaction summary. A transaction can affect more than one hospital.
agg = fac.groupby('cms_record_id', as_index=False).agg(
    n_matched_facilities=('facility_id','nunique'),
    n_rural_at_event=('rural_at_event_confirmed_from_panel_bool','sum'),
    n_rural_proxy=('rural_nearest_proxy_bool','sum'),
    n_rating_candidates=('candidate_for_rating_event_study_bool','sum'),
    any_acquisition_merger_label=('cms_acquisition_merger_label_bool','max'),
    any_repeat_facility_event=('multiple_reported_events_for_facility_bool','max'),
)
cols = [c for c in ['cms_record_id','event_year','cms_chow_type','ORGANIZATION NAME - BUYER',
                    'ORGANIZATION NAME - SELLER','EFFECTIVE DATE'] if c in txn]
agg = agg.merge(txn[cols].drop_duplicates('cms_record_id'), on='cms_record_id', how='left', validate='one_to_one')
agg['transaction_verification_status'] = 'PENDING_EXTERNAL_VERIFICATION'
agg['reviewer_notes'] = ''
agg.to_csv(OUT/'distinct_transaction_review_queue.csv', index=False)

rural = fac['rural_at_event_confirmed_from_panel_bool'] | fac['rural_nearest_proxy_bool']
summary = [
    'STEP 16: TRANSACTION VERIFICATION WORKLIST',
    f'Distinct CMS records matched: {fac.cms_record_id.nunique():,}',
    f'Distinct CMS-record/facility pairs: {len(fac):,}',
    f'Distinct CMS records with rural exact/proxy facility: {fac.loc[rural,"cms_record_id"].nunique():,}',
    f'Distinct CMS records with rating-screen candidate: {fac.loc[fac.candidate_for_rating_event_study_bool,"cms_record_id"].nunique():,}',
    f'CMS acquisition/merger labeled records with rural exact/proxy facility: {fac.loc[rural & fac.cms_acquisition_merger_label_bool,"cms_record_id"].nunique():,}',
    'Verified horizontal mergers: NOT DETERMINED',
    '2023 parent linkage: context only, not a pre/post ownership history.',
    'All records require independent verification before causal treatment assignment.',
    'Outputs: facility_event_review_queue.csv and distinct_transaction_review_queue.csv',
]
(OUT/'README_step16.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8')
print('\n'.join(summary))
print(f'\nSaved to {OUT}')
