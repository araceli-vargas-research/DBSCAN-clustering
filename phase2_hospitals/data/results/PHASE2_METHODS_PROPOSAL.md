# Phase 2: Effective Independence in Rural Hospital Substitute Networks

**Status:** Research design proposal, 8 November 2026. Not an implemented or validated estimator.  
**Audience:** Research collaborators reviewing Phase 2 methodology.

## Research question and potential contribution

Can rural hospitals be geographically accessible yet lack competitively independent substitutes because nearby facilities share legal ownership, delegated management authority, or documented decision rights through nonownership affiliations?

The potential methodological contribution is a **time-varying, service- and decision-specific hospital–organization control network** layered onto geographically accessible hospital substitutes. K-nearest-neighbour (KNN) searches and DBSCAN are supporting spatial tools, not the novelty. The empirical claim remains untested: the network must be shown to identify meaningful variation that legal ownership alone misses and to explain outcomes or exposure to organizational shocks.

## Distinct concepts

1. **Physical access:** Which facilities are reachable and how far away are they?
2. **Clinical substitutability:** Which facilities can provide the relevant service line, e.g. emergency care, obstetrics, or surgery?
3. **Competitive independence:** Which entities independently control the relevant decisions, e.g. insurer contracting, operations, staffing, or service-line continuation?

A clinical affiliation is not proof of common pricing control. Unknown relationships must not be coded as independent.

## Spatial architecture

- **KNN:** Identifies and ranks the closest potential alternatives to each focal hospital. KNN itself does not form clusters or prove economic substitutability.
- **DBSCAN:** Identifies spatially connected groups and isolated facilities under a chosen distance and minimum-density rule. DBSCAN clusters are not automatically economic markets.
- **Primary choice set:** Consider a service-appropriate distance or travel-time radius with accessibility decay, rather than forcing every rural hospital to have exactly k meaningful substitutes. Use k = 3, 5, 10 as sensitivity checks.
- **Robustness:** Vary DBSCAN epsilon (existing 5, 10, 15, 20, 25 km baselines) and min_samples separately; compare road travel time and patient-flow-calibrated distance decay where data permit. Pre-event flow calibration helps avoid post-treatment leakage but does not remove all endogeneity.

## Corrected independent-decision-maker measure

For focal hospital i, year t, service s, and decision right r, define a geographic candidate set N_i. Let g_t^{s,r}(j) identify the verified controlling entity of hospital j for that decision. Under a *single, nonoverlapping controlling-entity partition*, a candidate effective substitute score is:

KES_{i,t}^{s,r} = SUM_{g in G(N_i) excluding g(i)} MAX_{j in N_i: g(j)=g} [ w_s(d_ij) * q_{ij,t}^s * a_{j,t}^s ].

- w_s(d_ij): prespecified geographic accessibility/decay weight.
- q_{ij,t}^s: service-specific substitutability indicator or empirically supported score.
- a_{j,t}^s: service-specific availability/capacity factor, kept separate in robustness checks.
- MAX groups multiple nearby hospitals controlled by one entity into **one decision-making alternative**. It is not a measure of total physical service capacity.
- Excluding g(i) removes the focal hospital's own controlling entity from the count of *independent rivals*.

**Rival-side correction:** If j and l are both controlled by System S but neither is tied to focal hospital i, the old pairwise formula c_ij and c_il would count two rivals. Grouping j and l under g=S counts one independent rival decision-maker. This is the central mathematical correction.

**Limits:** MAX may understate the value of multiple sites within one system; report a separate physical-access/capacity metric. Overlapping or partial control cannot always be represented by one g(j). For those cases retain a hospital–organization bipartite graph and a pairwise, service- and decision-specific control matrix C_t^{s,r}=[c_jl], including *rival–rival* links. Do not impose transitive grouping on weak affiliations without documented common decision rights.

## Control relationship evidence and scenarios

Constructing a dated, sourced hospital–organization relationship registry with facility identifier, organization identifier, effective start/end dates, service/decision scope, relationship type, authority evidence, verification status, and provenance.

Proposed classification scenarios (not assumed numerical weights):

A. **Legal control:** Verified controlling ownership.
B. **Operational control:** A plus documented delegated operational decision rights.
C. **Broader documented coordination:** B plus specific, documented network/affiliation rights relevant to the outcome.

These are *sensitivity scenarios*, valid upper/lower causal bounds must be further identified through research. Missing and ambiguous relationships are kept explicit. Must not assign arbitrary 0.25/0.5 weights to affiliation labels.

Sources to audit: AHA Annual Survey (licensing/coverage constraints), CMS provider and CHOW data (ownership transactions, not all affiliations), state hospital licensing and transaction filings, management contracts, annual reports, network directories, and dated hospital announcements. Verify what each source actually establishes about decision rights.

## Empirical tests

1. **Feasibility:** Can nonownership control relationships be documented reliably and dated for a pilot sample?
2. **Measurement:** Compare geographic candidate facilities, distinct legal owners, and distinct verified decision-making entities; quantify disagreement under the same geographic choice sets. This is not an HHI-centred design.
3. **Outcomes:** Test whether service-specific independent substitutes provide incremental, out-of-sample information about patient experience or service availability beyond physical access and hospital characteristics. Distinguish prediction from causation.
4. **Identification (conditional on data):** Identify rural hospitals exposed to changes in *nearby rival control* caused by externally dated parent-system mergers or management-rights transfers, without focal-hospital acquisition. Test pre-trends, event timing, regional shocks, anticipation, and spillovers. Bystander designs are not inherently exogenous or unprecedented.
5. **Heterogeneity:** Test whether the association between physical accessibility and outcomes differs when accessible substitutes share effective control. Effects may differ between commercial decision rights and clinical collaboration.

When studying service-line outcomes, avoid simultaneously determined neighbour-service variables: freeze pre-event service definitions or use explicitly lagged predetermined exposure and document remaining risks.

## Current evidence versus pending work

**Already developed in the existing Phase 2 workflow:** 2019–2026 hospital panel; rural classification; projected Euclidean distance and nearest-neighbour analysis; DBSCAN geographic clustering; 2023 parent-system linkage for context; CMS CHOW matching and validation queues. Step 16 identified 475 matched distinct CMS records, 152 rural exact/proxy transactions, and 33 rating-screen candidates. Screening counts should **not be treated as verified horizontal mergers**.

**Not yet established:** longitudinal management/affiliation control registry, decision-specific common-control groups, historical nonownership network events, validated KES implementation, service-specific capacity/flow calibration, outcome horse race, or causal estimates.

## Immediate next step: Step 17 feasibility pilot

1. Select a reproducible stratified sample of rural hospitals, including geographically isolated and multi-hospital-neighbour cases.
2. Audit dated management, affiliation, governance, and legal ownership relationships, recording source URLs and exact decision rights.
3. Produce `hospital_organization_relationships.csv`, `relationship_evidence_log.csv`, `hospital_control_entities_by_service_year.csv`, and an explicit missingness report.
4. Build legal-control and documented-operational-control versions of KES for pilot markets; report facility count versus effective rival-controller count.
5. Stop before causal claims. Decide whether national scale-up is feasible based on verifiable coverage and meaningful disagreement.

## Relation to the antitrust manuscript

This research can illustrate the information required to identify actual competitive alternatives in rural hospital markets. It **cannot**, by itself, prove that enforcement agencies cannot acquire that information, that all affiliations suppress competition, or that merger intervention is necessarily harmful. Distinguish the manuscript's theoretical argument from the empirical tests.

## Literature positioning to verify before novelty claims

- **Dafny, Ho, and Lee (2019):** Cross-market hospital merger price effects and related bystander identification logic.
- **Lewis and Pflum (2017):** Out-of-market hospital acquisitions, insurer bargaining power, and price effects.
- **Dranove, Gaynor, and Geddes (2025), NBER Working Paper 34159:** Rural hospital acquisitions, obstetric service closures, and patient outcomes. [NBER paper](https://www.nber.org/papers/w34159)
- **Becker's Hospital Review (2026):** Rural hospital partnerships, management arrangements, and network affiliations without ownership transfers. Relevant to identifying organizational relationships that conventional ownership records may miss. [Becker's report](https://www.beckershospitalreview.com/hospital-transactions-and-valuation/how-rural-hospitals-are-teaming-up-with-big-systems-without-selling/)
- **Harold Demsetz (1973), *Industry Structure, Market Rivalry, and Public Policy*:** Price-theory critique of inferring market power directly from concentration. Relevant to distinguishing observed organizational structure from actual competitive conduct and efficiency. [Original paper](https://doi.org/10.1086/466752)
- **Geoffrey Manne and Kristian Stout (2020), *Comments on the Draft Vertical Merger Guidelines*:** Economic analysis of organizational integration, contractual alternatives, efficiencies, and dynamic competitive effects. Relevant to evaluating hospital affiliations without presuming that integration necessarily harms competition. [FTC submission](https://www.ftc.gov/system/files/attachments/798-draft-vertical-merger-guidelines/icle_-_vmg_draft_comments_0.pdf)
- **FTC/DOJ (1996), *Statements of Antitrust Enforcement Policy in Health Care*, Statement 8:** Historical treatment of independent physician networks, joint contracting, financial-risk sharing, and clinical integration. The agencies withdrew these statements in 2023. [FTC withdrawal notice](https://www.ftc.gov/news-events/news/press-releases/2023/07/federal-trade-commission-withdraws-health-care-enforcement-policy-statements)
- **FTC, *MedSouth Independent Practice Association* (2002):** Historical advisory-opinion example concerning clinically integrated physician networks and joint contracting. Relevant to examining how antitrust rules distinguish coordination from efficiency-enhancing integration. [FTC advisory opinion](https://www.ftc.gov/legal-library/browse/advisory-opinions/advisory-opinion-miles-02-19-02)

**Novelty to establish:** Phase 2 does not claim that K-nearest-neighbour analysis, DBSCAN, or distance-based hospital competition measures are independently novel. Its proposed contribution is a service-specific spatial network that distinguishes physical hospital availability from effective competitive independence, incorporating verified ownership, management authority, and nonownership organizational relationships.

**Empirical limitation:** Existing Phase 2 data support geographic proximity analysis, preliminary parent-system linkage, and administrative transaction screening. They do not yet establish the competitive effects of nonownership affiliations or whether antitrust enforcement induced clinical integration. Those claims require additional historical relationship data and empirical validation.

**Novelty is a hypothesis to establish by targeted review and empirical validation, not a result of proposing the equation.**
