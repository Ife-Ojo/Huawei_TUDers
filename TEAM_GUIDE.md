# Team guide - how the code fits together

## Folder layout
```
Data/            data + databases (stays on your laptop, never on GitHub)
setup/           build the databases once
common.py        shared settings: paths, dates, which rows we predict
features/        one clue file per person (copy _template.py to start)
train.py         combines all clue files, trains, scores, makes submissions (Eoin)
tuning.py        finds the best model settings (Eoin)
compare_setups.py  compares training setups: days to learn from, num_leaves, learning rate (Eoin)
experiments.csv  log of every train.py run
outputs/         submission files (never on GitHub)
notebooks/<name> personal scratch space
reference/       baseline.py (our first model), for reading
starter_solution.py  organisers' starter (already on main)
```

## First-time setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install duckdb pandas lightgbm scikit-learn
brew install libomp                     # Mac only, needed by LightGBM
# put train.csv, evaluation.csv, example_submission.csv in Data/
python setup/make_sample.py             # 5% copy, a few minutes
python setup/make_full.py               # full database, several minutes
```

## Adding your clues
1. Copy `features/_template.py` to `features/<yourname>.py`
2. Prefix every column: Ife `u_`, Marcela `i_`, Abraham `cv_`, Elvita `emb_` (Eoin `base_`)
3. Learn only from `fit` (data before the cutoff)
4. Test: `python features/<yourname>.py`
5. Score: `python train.py full --groups base,<yours> --rounds 40 --seeds 3 --ablation`
6. Paste that output into your pull request description (don't commit experiments.csv)

## Commands
```
python train.py                 quick check on the 5% sample
python train.py full            real score
python train.py full --ablation does each clue file help?
python train.py full --submit   also writes a submission to outputs/
python train.py full --rounds 40 --seeds 3   fair comparison: fixed length, averaged over 3 seeds
python train.py full --train-days 4 --leaves 15 --lr 0.02   learn from more days, simpler trees, slower learning
python compare_setups.py full     tries 12 training setups and prints which is best
python tuning.py full             tries ~30 model settings, fair-checks the best, saves best_params.json
python train.py full --params best_params.json --seeds 3   use the tuned settings
```
