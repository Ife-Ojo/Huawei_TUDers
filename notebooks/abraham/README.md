# Abraham's Co-visitation Features (`cv_`)

## Overview
This module builds item-item transition patterns ("Item B clicked after Item A") from user click histories and scores each candidate item against the user's recent clicks prior to the event cutoff.

In the validation benchmark, adding these features increased average AUC from **0.62259** to **0.64691** (**+0.02432 lift**), with `cv_rank_in_event` emerging as the #1 most important feature in the LightGBM model.

---

## Leakage Prevention (Winnie's Review)
1. **Pre-Cutoff Transitions Only**: Co-visitation counts and transition probabilities are computed exclusively from training rows where `timestamp < cut` (`common.fit_sql(cut)`). No events or clicks from at or after the cutoff enter the transition matrix.
2. **Filtered History Sequences**: When scoring target events, any click in `history_timestamps` with timestamp $\ge \text{cut}$ is discarded.
3. **No Target Label Leakage**: Target labels (`clicked`) are never referenced during feature generation.
4. **Verified via Unit Tests**: `tests/test_abraham.py` asserts that injecting future events or future click timestamps produces identical feature outputs (`test_zero_leakage_future_events` and `test_zero_leakage_history_timestamps`).

---

## Feature Definitions (All Prefixed with `cv_`)

| Feature Name | Description | Fallback / Missing |
| :--- | :--- | :--- |
| `cv_has_history` | Binary indicator (1 if user has $\ge 1$ pre-cutoff click, else 0) | `0` |
| `cv_hist_len_valid` | Number of valid pre-cutoff historical clicks available (0 to 5) | `0` |
| `cv_in_recent_5` | 1 if the candidate item was clicked in user's last 5 clicks, else 0 | `0` |
| `cv_last1_pair_cnt` | Historical transition count from user's last click to candidate | `0` |
| `cv_last1_prob` | Empirical transition probability $P(\text{cand} \mid \text{last\_1})$ | `0.0` |
| `cv_last1_cos` | Cosine similarity between user's last click and candidate | `0.0` |
| `cv_last3_sum_cnt` | Sum of transition counts from user's last 3 clicks | `0` |
| `cv_last3_sum_prob` | Sum of transition probabilities from user's last 3 clicks | `0.0` |
| `cv_last3_max_prob` | Max transition probability across user's last 3 clicks | `0.0` |
| `cv_last3_sum_cos` | Sum of cosine similarities across user's last 3 clicks | `0.0` |
| `cv_last3_max_cos` | Max cosine similarity across user's last 3 clicks | `0.0` |
| `cv_last3_tw_prob` | Time-weighted transition probability across last 3 clicks | `0.0` |
| `cv_last5_sum_cnt` | Sum of transition counts from user's last 5 clicks | `0` |
| `cv_last5_sum_prob` | Sum of transition probabilities from user's last 5 clicks | `0.0` |
| `cv_last5_max_prob` | Max transition probability across user's last 5 clicks | `0.0` |
| `cv_last5_sum_cos` | Sum of cosine similarities across user's last 5 clicks | `0.0` |
| `cv_last5_tw_prob` | Time-weighted transition probability across last 5 clicks | `0.0` |
| `cv_rank_in_event` | Candidate rank (1-10) within the event sorted by `cv_last3_sum_prob` | `1..10` |
| `cv_prob_minus_event_mean` | Candidate `cv_last3_sum_prob` minus the event mean | `0.0` |
| `cv_event_max_last3_prob` | Highest `cv_last3_sum_prob` among all 10 candidates in the event | `0.0` |

---

## Settings & Hyperparameters
- **Sliding sequence window**: Up to 5 consecutive clicks ($c_{j} - c_{i} \le 5$).
- **Maximum time interval**: 24 hours (86,400 seconds).
- **Time decay weighting**:
  $$w_k = \frac{1}{1 + \Delta t_k / 3600.0}$$
  where $\Delta t_k = t_{\text{event}} - t_{\text{click}_k}$ in seconds.
- **Normalization**:
  - Transition probability: $\frac{\text{pair\_cnt}(A, B)}{\text{click\_count}(A)}$
  - Cosine similarity: $\frac{\text{pair\_cnt}(A, B)}{\sqrt{\text{click\_count}(A) \times \text{click\_count}(B)}}$

---

## How to Run & Regenerate

### 1. Run Unit Tests
```bash
python -m unittest tests/test_abraham.py
```

### 2. Standalone Co-visitation Precomputation
```bash
# Precomputes pair counts and exports Parquet to Data/
python notebooks/abraham/build_covisitation.py sample
python notebooks/abraham/build_covisitation.py full
```

### 3. Test Feature Generation
```bash
python features/abraham.py          # On sample DB
python features/abraham.py full     # On full DB
```

### 4. Run Model Training & Ablation
```bash
python train.py --ablation
python train.py full --ablation
```

---

## Runtime Benchmarks
- **Co-visitation matrix generation**: ~0.7 seconds (sample DB).
- **Feature table build**: ~0.08 seconds (1,590 candidate rows).
- **Memory footprint**: DuckDB memory capped at 8GB; executed completely in-process.

---

## Validation Benchmark Results

Evaluated across both official team validation folds:
- **Test 1** (learn Jan 24, test Jan 25): AUC **0.62850** (vs 0.59845 baseline)
- **Test 2** (learn Jan 25, test Jan 26): AUC **0.66532** (vs 0.64673 baseline)
- **Average AUC**: **0.64691** (vs 0.62259 baseline)
- **Ablation Lift**: **+0.02432** (Verdict: **HELPS**)
