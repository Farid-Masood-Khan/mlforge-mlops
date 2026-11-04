# Data Card — Telco Customer Churn

| Attribute | Value |
|---|---|
| `data_sha256` | `5de3ddc9f589b369e71eae2a4dbcb5a96615910b1093c326f0ea57e3f9870571` |
| `source_url` | `synthetic://telco-customer-churn-v1` |
| `downloaded_at` | `2026-11-02` |
| `rows` | `7043` |
| `cols` | `21` |
| `target_rate` | `0.3558` |

## Dataset Notes & Known Quirks
- **Primary Source:** IBM Telco Customer Churn dataset.
- **Identifier:** `customerID` (string, unique per customer) must be dropped before modeling.
- **Tenure Quirk:** Rows where `tenure == 0` have blank / whitespace `TotalCharges` entries in the raw CSV. These must be explicitly coerced to NaN during ingest and median-imputed by the feature pipeline.
- **Raw Blank Ceiling:** Blank `TotalCharges` rate must be <= 2% (default Telco dataset has 11 blanks / 7043 rows = ~0.15%).
