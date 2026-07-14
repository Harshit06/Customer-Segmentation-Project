# Executive Summary: Customer Segmentation Analysis

**Prepared for:** Marketing & Growth Leadership
**Prepared by:** Data Analytics Team
**Date:** July 2026

## Objective

Identify distinct customer groups within the retail customer base so marketing spend can be
targeted rather than blanket, improving retention and revenue efficiency.

## Approach

We analyzed 1,200 customers (built on a real Kaggle customer dataset, expanded with modeled
behavioral data — see README for full data lineage) across seven behavioral dimensions: age,
income, spending score, purchase frequency, recency, tenure, and monetary value. Three clustering
algorithms — K-Means, Hierarchical Clustering, and DBSCAN — were tested and compared. K-Means
(k=4) was selected as the production model based on cluster balance, interpretability, and
stability testing.

## The Four Segments

| Segment | Share of Base | Profile | Recommended Action |
|---|---|---|---|
| **Premium Spenders** | 21% | Highest income & spending, most frequent, most recent, longest tenure | VIP loyalty tier, priority support, referral program |
| **Steady Mature Regulars** | 24% | Older, mid income/spend, dependable | Cross-sell campaigns, mid-tier loyalty perks |
| **New Bargain Hunters** | 28% | Younger, price-sensitive, engaged, growing | Time-boxed discounts, bundle deals, habit-building rewards |
| **At-Risk Customers** | 27% | High income but lowest engagement, ~120 days since last order | Urgent win-back campaign, root-cause survey |

## Key Findings

1. **No single metric predicts value.** Income alone does not predict spending — multi-dimensional
   clustering was necessary to find real structure.
2. **At-Risk Customers are the highest-leverage opportunity.** They have the income to spend but
   have quietly disengaged — unlike low-income segments, the barrier isn't affordability, making
   them a strong win-back target.
3. **The segmentation is statistically stable.** Repeated testing on random subsamples showed
   consistent results, meaning these segments can be trusted for ongoing campaign planning, not
   just a one-time snapshot.
4. **DBSCAN was evaluated but rejected for production use** — while it scored well numerically,
   it produced one oversized cluster and many tiny, unusable fragments rather than clean,
   equal-sized business segments.

## Recommended Next Steps

1. Launch a win-back campaign for At-Risk Customers within the next marketing cycle.
2. Pilot a VIP loyalty tier for Premium Spenders to protect and grow this high-value segment.
3. Re-run this segmentation quarterly and track how customers migrate between segments over time
   to measure whether interventions are working.
4. Replace the engineered behavioral fields with real transaction data as it becomes available to
   sharpen segment accuracy further.

## Expected Business Impact

Targeted campaigns per segment (instead of one blanket strategy) are expected to improve
campaign response rates and reduce wasted spend on customers unlikely to respond to generic
offers, while directly addressing the revenue at risk in the At-Risk segment.

---
*Full technical methodology, code, and visualizations are available in the accompanying Jupyter
notebook and GitHub repository.*
