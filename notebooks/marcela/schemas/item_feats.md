# item_feats_<date>.parquet

Item-level features. Key: item_id, one row per item.
Dates: 20250120 ... 20250127 (20250125 and 20250126 serve the folds, 20250127 serves eval).

## Cutoff logic
Each file is built at <date> 00:00 UTC and uses only data strictly before that time:
- click columns use unique clicks (from user histories and train labels) with click time < cutoff
- impressions and CTR use train rows with timestamp < cutoff
A training row on day D joins the file of day D. A fold or eval row joins the file of its fold or eval cutoff.

## Missing values
- Items without clicks before the cutoff have 0 in the count columns and NaN in the click-based ratios and shares.
- Items never seen before the cutoff have no row (NaN after a left join).
- No other fill values are used.

## Columns
| column | type | meaning |
|---|---|---|
| item_id | VARCHAR | key |
| itm_clicks_total | BIGINT | unique clicks before the cutoff |
| itm_users_total | BIGINT | distinct users who clicked the item |
| itm_clicks_6h | BIGINT | clicks in the 6 hours before the cutoff |
| itm_clicks_24h | BIGINT | clicks in the 24 hours before the cutoff |
| itm_clicks_per_day | DOUBLE | clicks_total divided by the days of click history |
| itm_trend_24h | DOUBLE | clicks_24h divided by the item's average daily clicks |
| itm_item_age_days | DOUBLE | days since the item's first click |
| itm_hours_since_last_click | DOUBLE | hours since the item's latest click |
| itm_share_b0 | DOUBLE | share of the item's clicks in time bucket 0 (00-06 UTC) |
| itm_share_b1 | DOUBLE | share in bucket 1 (06-12 UTC) |
| itm_share_b2 | DOUBLE | share in bucket 2 (12-18 UTC) |
| itm_share_b3 | DOUBLE | share in bucket 3 (18-24 UTC) |
| itm_share_weekend | DOUBLE | share of the item's clicks on Saturday or Sunday |
| itm_impressions | BIGINT | times the item was shown in train before the cutoff |
| itm_ctr_smooth | DOUBLE | (clicks + 20 x overall rate) / (impressions + 20) |

## Code
scripts/item_features.py builds the tables, scripts/export_item_feats.py writes the files.
