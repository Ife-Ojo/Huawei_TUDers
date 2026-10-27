# Exploration summary

Data: each EVENT has exactly 10 candidate items (rows); at most one of them is clicked. History = the user's last <=20 clicked items before the event.

## Whole training set
- 6,868,240 rows = 686,824 events, 53,003 users, 2025-01-20 00:00:00 -> 2025-01-26 23:59:00 UTC
- **Share of events with no click: 37.1%**
- Click rate per candidate row: 6.3%
- **Click rate for candidates already in the user's history: 29.3%** (vs 5.7% for items not in history); 2.6% of candidates are in history; 12.0% of all clicks are on a history item
- **History length** (items): median 20, p90 20, p99 20, max 20; events with empty history: 0.3%

## Validation fold 2025-01-25 (cutoff 2025-01-25 00:00 UTC)
- 43,964 users (1 event each), 439,640 rows
- **Time range: 2025-01-25 00:00:00 -> 2025-01-26 23:59:00 UTC**; 80.1% of events fall in the first 24h after the cutoff
- **History length**: median 16, p90 20, p99 20, max 20
- **Share of new users**: 0.8% have an empty history; 3.3% have no event in the training data before the cutoff
- Share of candidates already in the user's history: 2.5%
- **Share of events with no click: 36.6%**
- **Click rate, candidate in history: 30.1%** | not in history: 5.7%

## Validation fold 2025-01-26 (cutoff 2025-01-26 00:00 UTC)
- 35,138 users (1 event each), 351,380 rows
- **Time range: 2025-01-26 00:00:00 -> 2025-01-26 23:59:00 UTC**; 100.0% of events fall in the first 24h after the cutoff
- **History length**: median 20, p90 20, p99 20, max 20
- **Share of new users**: 0.6% have an empty history; 1.7% have no event in the training data before the cutoff
- Share of candidates already in the user's history: 2.6%
- **Share of events with no click: 36.6%**
- **Click rate, candidate in history: 30.0%** | not in history: 5.7%

## EVALUATION set (cutoff 2025-01-27 00:00 UTC)
- 52,939 users (1 event each), 529,390 rows
- **Time range: 2025-01-27 00:00:00 -> 2025-02-02 23:56:00 UTC**; 66.6% of events fall in the first 24h after the cutoff
- **History length**: median 15, p90 20, p99 20, max 20
- **Share of new users**: 1.3% have an empty history; 2.7% have no event in the training data before the cutoff
- Share of candidates already in the user's history: 2.5%

## Evaluation events per day (first event of each user)
- 2025-01-27: 35,269
- 2025-01-28: 8,764
- 2025-01-29: 3,914
- 2025-01-30: 2,236
- 2025-01-31: 1,273
- 2025-02-01: 898
- 2025-02-02: 585
