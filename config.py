"""The only file you may need to edit."""
from pathlib import Path

SHARED = Path("shared")                    # folder holding train.parquet and eval.parquet
TRAIN_PQ = SHARED / "train.parquet"
EVAL_PQ = SHARED / "eval.parquet"
FOLDS = SHARED / "folds"                   # validation folds get saved here

N_TRAIN_ROWS = 6_868_240
N_EVAL_ROWS = 529_390

# Validation cutoffs, midnight UTC (the data's timestamps are UTC epoch seconds).
CUTOFFS = ["2025-01-25", "2025-01-26"]
DATA_END = "2025-01-27"                    # training data ends here; the real evaluation starts here

# Submission file layout: [id column, prediction column].
# !! CHECK THIS AGAINST THE COMPETITION RULES / SAMPLE SUBMISSION !!
SUBMISSION_COLUMNS = ["rowid", "prediction"]
