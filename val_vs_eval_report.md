# Validation folds vs evaluation set

## 1. Side by side

                                     fold 2025-01-25  fold 2025-01-26       EVAL
events                                     43964.000        35138.000  52939.000
share weekend (day_type=1)                     1.000            1.000      0.028
share in first 24h after cutoff                0.801            1.000      0.666
history length, median                        16.000           20.000     15.000
share empty history                            0.008            0.006      0.013
hours since last click, median                33.017           24.600     42.550
candidates in history, mean (of 10)            0.252            0.257      0.254
user events per day (scaled), mean             2.104            2.395      1.844
share time_bucket 0 (00-05h UTC)               0.216            0.230      0.197
share time_bucket 1 (06-11h UTC)               0.286            0.294      0.281
share time_bucket 2 (12-17h UTC)               0.268            0.267      0.279
share time_bucket 3 (18-23h UTC)               0.229            0.210      0.242

## 2-3. Fold 2025-01-25 vs EVAL

Single-feature difference AUC (0.50 = same, higher = more different):

day_type                  0.986
dow                       0.796
hours_since_cutoff        0.585
hours_since_last_click    0.563
user_events_per_day       0.547
hist_len                  0.527
hour                      0.516
time_bucket               0.515
n_cand_in_history         0.501

Adversarial AUC, all features:                1.000
Adversarial AUC, WITHOUT calendar features:   0.579

## 2-3. Fold 2025-01-26 vs EVAL

Single-feature difference AUC (0.50 = same, higher = more different):

dow                       0.994
day_type                  0.986
hours_since_cutoff        0.667
hours_since_last_click    0.619
user_events_per_day       0.595
hist_len                  0.581
hour                      0.531
time_bucket               0.529
n_cand_in_history         0.501

Adversarial AUC, all features:                1.000
Adversarial AUC, WITHOUT calendar features:   0.624
