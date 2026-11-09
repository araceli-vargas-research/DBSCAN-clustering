# Competition without price competition? Local dialysis concentration and hospitalization quality

## Motivation and antitrust question

Dialysis is an antitrust case in the USA. which nonprice competition is consequential. As Medicare payment rules limit the role of freely chosen market prices for covered treatment, patients still depend on facility quality, staffing, access, and continuity.

While chain acquisitions may create operational efficiencies or weaken local competitive pressure, these mechanisms have the potential to produce quite the opposite quality of effects. The empirical question is whether changes in ownership concentration precede changes in facility hospitalization quality.

The FTC requires dialysis divestitures in multiple transactions; not limited to DaVita/Gambro, Fresenius/Renal Care Group, and Fresenius/Liberty. That makes local common ownership and whatever other alternatives remain, as a direct antitrust concern. A national chain's size does not adequately measure the competition faced by a particular clinic.

Church and Ware's industrial-organization framework motivates separating market structure, firms' of conduct, and performance. Here, the ownership concentration is a structure measure, and quality is a performance measure; the correlation between the two does not produce results that can identify conduct or prove anticompetitive harm.

## Contribution

A reproducible, public-data audit of contemporary local dialysis concentration and quality that explicitly matches ownership observed before a clinical window to outcomes measured during that same exact window. The additional ownership audit tests if conclusions are dependant on ambiguous or outdated chain labels. This is a feasible methodological contribution and should not be classified as a verified first-in-literature discovery.

## Hypotheses

i. If reduced competitive pressure worsens quality, higher pre-window concentration predicts a greater probability of worse-than-expected hospitalization

ii. If chain efficiencies dominate, chain affiliation may improve quality even after local concentration is held constant.

iii. Relationships may weaken after facility and year effects, ownership audits, and pandemic exclusions.

## Data and unit of observation

The CMS facility snapshots contain identifiers, county, chain organization, dialysis stations, and quality windows and categories. The main unit is a facility-clinical-year, not a facility-download-year. All needed data is bundled with this proposal. Code exposes the exact eligible sample rather than a preselected number of years.

The current executed study covers US in-center dialysis. A rural would require an explicitly defined rural classification and dated geographic crosswalk; small counties and single-facility counties should not simply be labeled rural. No rural-specific estimate is claimed here.

## Empirical design

Estimate a facility and clinical-year fixed-effects linear probability regression of worse-than-expected hospitalization on lagged county reported-owner HHI, major-chain affiliation, and log-station capacity. Cluster uncertainty by county. The baseline distinguishes the common ownership from the physical facility fragmentation. Reported prespecified sensitivity runs different scenario tests for multi-facility counties, uncertain ownership labels, and the pandemic period.

The coefficient is the association and asks: how the same facility's reported quality change when its pre-window local ownership concentration changes, which nets common shocks and included controls. A null or imprecise coefficient cannot tell us that mergers are harmless. A positive coefficient does not tell us that a specific merger caused harm.

## Causal merger extension

To answer the stronger question, "Do mergers worsen quality?", assemble facility-specific transaction dates and ultimate owners from FTC case documents and verified corporate records. Retain acquisitions for which the same CCN can be tracked; do not treat every chain-label change as an acquisition. Use a staggered-adoption cohort estimator comparing acquired facilities with appropriate not-yet-acquired or never-acquired facilities, and inspect pre-acquisition trends. Require clean clinical windows before and after the acquisition interval. Separately examine outcomes at unaffected rival facilities in overlap markets, since changes at acquired facilities may reflect operational practices rather than market power.

The current package does not claim to have completed that transaction verification. County markets, divestitures, selection into acquisition, overlapping mergers, spillovers, and outcome suppression must be addressed before causal interpretation. A merger-remedy comparison could be informative, but divestiture selection is itself nonrandom.

## VAR component

The package includes explicit reduced-form VAR(1) code and Cholesky impulse-response graph code. These short public annual histories do not support a persuasive causal VAR. Its data checks intentionally refuse an undersized innovation covariance or irregular clinical years. A longer harmonized series could support descriptive dynamic analysis; identifying a competition shock would still require more than a Cholesky ordering.

## Deliverables and interpretation

The executed deliverables are raw source data, provenance checksums, a cleaned exposure-outcome panel, clustered regressions, seven reproducible figures, and the optional VAR and pro forma merger-screen scripts. Conclusions must distinguish descriptive market structure, adjusted quality associations, and causal merger effects. Results should be reported whether their sign supports harm, efficiency, or neither.

References and official source links appear in README.md.

# Phase 2 research part 

