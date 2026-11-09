# Dialysis competition and quality: a reproducible public-data antitrust project

**Research question:** Is higher local dialysis concentration associated with worsened hospitalization quality, after accounting for persistent facility differences and common year shocks?

This is an exploratory study, with the complete source facility CSVs and codes needed to reproduce it. It cannot provide evidennce for a causal estimate of merger effects. Ownership labels with county boundaries would most likely need further validation before the results could support a merger challenge. The project concerns in-center dialysis in the 50 US states and DC; it doesn't identify rural facilities separately.

## Run everything

Use Python 3.11+ in this directory:

```sh
python3 -m pip install -r requirements.txt
python3 analysis.py
```

The bundled data work offline. `analysis.py` is ruilt to rebuild three cleaned CSVs, every regression, and seven PNG/PDF figures. No credentials, patient claims, USRDS application, or proprietary data are needed.

To refresh the exact historical source snapshots:

```sh
python3 download_sources.py
python3 analysis.py
```

Downloads total roughly 75 MB of theme archives, processed one at a time. The package contains the original, unedited facility CSVs extracted from those archives, not unrelated QIP/survey CSVs in the same theme ZIP. `data/raw/sources.json` records archive URLs, member names, sizes, SHA-256 checksums, and retrieval timestamps. CMS can change remote files; compare checksums before replacing the bundled inputs.

## Data and matching

Nine public CMS Dialysis Facility Compare snapshots: October 2019, October 2020, April 2021, November 2022, April and October 2023, October 2024, November 2025, July 2026. The April 2021 facility file repeats the October 2020 file; repeated outcomes are not new observations.

The script harmonizes CCN field-name changes and preserves six-digit identifiers. It retains positive-capacity facilities offering in-center hemodialysis with a county and a valid US state. Counties are state-plus-uppercase-county-name identifiers, not FIPS codes. It asserts unique CCNs within each source.

**Do not equate snapshot years with clinical outcome years.** The script parses the hospitalization window and keeps the first published observation for each facility-window. It matches the latest available facility ownership snapshot strictly before the clinical window starts, within 450 days. It excludes observations whose county changes between exposure and outcome. It never treats a multi-year mortality window as a one-year observation. Outcome-window coverage, exclusions, and usable samples are written to the audit files.

## Concentration measures

`HHI = 10,000 * sum(owner station share squared)` within county and ownership snapshot. Station shares measure capacity, not patient or revenue market shares. A single-firm market has HHI 10,000.

Named chains are grouped under their **reported CMS label**. Each independent facility is a separate firm; each ambiguous OTHER/blank label is also separate rather than incorrectly grouping every OTHER facility into a national chain. These assumptions can miss common ownership. Old Gambro, Renal Care Group, DSI, and Liberty labels can remain in CMS files. The code flags those labels and reports a sensitivity regression excluding exposure markets containing them or ambiguous OTHER labels. It does not infer ultimate owners solely from historical corporate mergers, because divestitures and subsequent sales require facility-specific verification.

A separate facility-level capacity HHI ignores common ownership; this is a fragmentation benchmark, not a replacement antitrust HHI. National DaVita/Fresenius station shares are descriptive under reported labels. Apparent jumps can reflect recoding, entry, exit, or acquisitions.

County is a feasible public-data **market proxy**, not a demonstrated antitrust relevant geographic market. Patients cross county borders, especially near urban boundaries. A travel-time market with historical geocodes and patient-flow evidence would improve the design.

## Main regression

A linear probability model:

`WorseHospital_it = alpha_i + year_t + beta*(HHI_pre_it/1000) + gamma*MajorChain_pre_it + delta*log(Stations_pre_it) + error_it`

- Outcome: 1 if CMS reports hospitalization as **Worse than Expected**, 0 if As Expected or Better than Expected. Not Available and blank categories are excluded, not coded as healthy.
- Exposure: ownership and capacity observed before the outcome window; not the ownership label published alongside the outcome.
- Facility fixed effects absorb persistent facility attributes; clinical-year effects absorb common shocks.
- Standard errors cluster by exposure county. Coefficients, t-based p-values, 95% intervals, sample counts, and clinical years are saved in `results/regression_results.csv`.
- A coefficient of 0.003 means a 0.3 percentage-point change in the probability of a worse-than-expected rating per 1,000 HHI points.

The script uses alternating projections to remove both fixed effects in an unbalanced panel, followed by OLS and a county-clustered sandwich covariance. It checks convergence and rank. Finite-sample adjustment counts facility and year effects conservatively; disconnected samples can make this adjustment somewhat conservative. Each model excludes facilities with only one usable observation.

Sensitivity models restrict to markets with two or more facilities; exclude ambiguous labels; exclude ambiguous and legacy labels; exclude 2020; and use facility fragmentation instead of firm concentration. These are robustness checks, not additional causal identification strategies.

## What the estimates can establish

They estimate within-facility associations with changes in pre-window local concentration. They cannot establish that mergers increased or decreased quality. Acquisition targeting, changing case mix, endogenous entry/closure, ownership miscoding, suppression, and market definition can confound the estimates. Classification also depends on statistical precision and patient volume, so this binary endpoint is not a continuous clinical harm measure. The inclusion of station capacity is an exploratory control that can absorb part of an acquisition mechanism.

Hospitalization categories are more comparable across the available schemas than the numeric hospitalization fields, whose definitions/units change. Even category definitions and risk-adjustment models may evolve. Pandemic exclusions and reporting delays further complicate comparisons; the no-2020 model is included for that reason. The raw infection and readmission fields are retained for extension, but no unsupported cross-schema numeric outcome pooling is performed.

## VAR code

```sh
python3 var.py
```

This produces `results/var_annual_inputs.csv`, checks consecutive clinical years and sample length, and **normally stops** because the panel is too short. The screening minimum is 30 years, not a guarantee that 30 is sufficient.

`--allow-short-sample` bypasses only that length screen for a classroom demonstration. The code still requires adequate residual degrees of freedom for a two-variable innovation covariance. It estimates a level VAR(1) with an intercept and reports transition eigenvalues and Cholesky IRFs if those checks pass. HHI is ordered first; that is an unverified ordering assumption, not merger-shock identification. It supplies no reliable short-sample confidence bands or stationarity claim. Do not fabricate quarterly/monthly observations by repeating or interpolating annual quality data.

The panel regression is the runnable empirical design for these data. A credible antitrust VAR needs substantially longer, consistent quality histories and a defensible source of exogenous competition shocks. A reduced-form VAR would describe dynamics, not identify the effect of a merger automatically.

## Merger-screen code

```sh
python merger_screen.py --buyer DAVITA --target 'FRESENIUS MEDICAL CARE'
```

This is a **hypothetical numerical example**, not a claimed announced transaction. The script calculates overlapping counties, pre/post HHI, and the mechanical `delta HHI = 20,000*s_buyer*s_target` under fixed station shares. It neither predicts quality effects nor determines legality. Replace the parties with validated transaction owners before use.

## Files

- `analysis.py`: cleaning, exposure matching, regressions, all seven main graphs.
- `download_sources.py`: exact public source downloads.
- `var.py`: VAR estimation and IRF code with data-sufficiency checks.
- `merger_screen.py`: pro forma concentration screen for specified reported owners.
- `PROPOSAL.md`: research framing and a path to a causal merger extension.
- `data/raw/`: all nine original facility CSVs and provenance manifest.
- `data/clean/`: facility snapshots, concentration panel, regression panel.
- `results/`: coefficients, audits, sample summaries, run log, figures.

## Sources and prior work

CMS data catalog: https://data.cms.gov/provider-data/dataset/23ew-n7w9

CMS archive API: https://data.cms.gov/provider-data/api/1/archive/aggregate/theme/dialysis-facilities/relative

Prior consolidation research: https://pmc.ncbi.nlm.nih.gov/articles/PMC8796909/

FTC DaVita/Gambro proceeding: https://www.ftc.gov/legal-library/browse/cases-proceedings/0510051-davita-inc

FTC Fresenius/Renal Care Group proceeding: https://www.ftc.gov/legal-library/browse/cases-proceedings/051-0154-fresenius-ag-matter

FTC Liberty dialysis divestiture case: https://www.ftc.gov/news-events/news/press-releases/2012/02/ftc-requires-fresenius-medical-care-ag-sell-60-dialysis-clinics-around-country-condition-acquiring

Research on dialysis acquisitions and quality already exists. No claim that this subject has never been studied is made. The proposed contribution is a reproducible contemporary public-data competition audit, with explicit exposure timing and ownership-data limitations.
