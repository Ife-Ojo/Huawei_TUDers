# Huawei_TUDers

Repository for the TechArena Click-Through Rate (CTR) challenge.

## Overview

This repository contains the code and models developed for the Huawei TechArena CTR competition.

### Starter Solution

`starter_solution.py` is a self-contained baseline script using only Python standard library modules (`csv`, `json`, `math`, `collections`). It combines:
1. **Item popularity**: Smoothed empirical click-through rate per item.
2. **User propensity**: Smoothed empirical click-through rate per user.
3. **Item-item co-click similarity**: Collaborative-filtering signal based on historical co-clicks.

#### Usage

```bash
python3 starter_solution.py --train train.csv --evaluation evaluation.csv --out submission.csv
```
