# Executed results

Main facility/clinical-year fixed-effects regression: **31,605 facility-year observations**, **6,828 facilities**, **1,796 county clusters**; clinical years 2020,2021,2022,2023,2024.

Per 1,000 additional points of pre-window reported-owner HHI, the estimated change in the probability of a worse-than-expected hospitalization rating is **0.054 percentage points**, with a 95% confidence interval **[-0.440, 0.548]** and p = **0.831**.

This does not show a statistically clear association in this specification. It does not establish that mergers improve quality, worsen quality, or have no effect. County boundaries and reported chain labels are exploratory proxies.

Sensitivity estimates and sample counts appear in regression_results.csv. The 493 overlap counties in proposed_merger_screen.csv concern an explicitly hypothetical DaVita/Fresenius combination, not an announced transaction.

The VAR input has only five consecutive annual observations. The default estimation correctly refuses this undersized series.

Validation completed: original-file SHA-256 checksums; unique facility clinical windows; ownership strictly before outcomes and within the 450-day tolerance; unchanged county; bounded HHI and owner HHI no smaller than facility fragmentation; binary outcome coding.
