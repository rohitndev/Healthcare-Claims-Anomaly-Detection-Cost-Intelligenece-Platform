# Compliance Dashboard

The Power BI compliance dashboard connects to the curated `fraud_scores` table in
Snowflake (production) or to the local `exports/powerbi_dataset.csv` (offline).

## Build the dashboard data and charts

```bash
python -m dashboards.generate_dashboard
```

This writes to `dashboards/exports/`:

| File | Purpose |
| --- | --- |
| `powerbi_dataset.csv` | Flat dataset the `.pbix` connects to |
| `provider_clusters.png` | Provider behavioural cluster scatter |
| `fraud_score_distribution.png` | Composite fraud-score histogram |
| `risk_tier_composition.png` | Claims by risk tier |
| `provider_risk_map.png` | Mean fraud score by provider state |

## Connect Power BI to the dataset

1. `Get Data` → `Text/CSV` → select `dashboards/exports/powerbi_dataset.csv`.
2. Or `Get Data` → `Snowflake` and point at the `FRAUD_SCORES` table.
3. Recommended visuals: provider risk map, fraud-score distribution, high-risk
   claim queue (score > 75), and provider cluster matrix.
