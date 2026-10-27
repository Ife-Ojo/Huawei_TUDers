"""Stage 3a: helpers that make COUNT features comparable between the validation folds and the
real evaluation.

Why: a count like "clicks on this item so far" is built from all training data BEFORE the cutoff.
  fold Jan 25 -> 5 days of data (Jan 20-24)     fold Jan 26 -> 6 days
  evaluation  -> 7 days (Jan 20-26)
A raw count is ~1.4x bigger at evaluation time than in the Jan 25 fold, so a model trained/validated
on small counts sees "unusually large" numbers at evaluation. Dividing by the days observed fixes it.

Usage in anyone's feature code:
    from scaling import observation_days
    days = observation_days(con, "2025-01-25")       # 5.0
    # in SQL:   count(*) / {days} AS item_clicks_per_day
"""
from db import epoch_of

TRAIN_START = "2025-01-20"     # first day with real events (impressions + click labels)

def observation_days(con, cutoff, start=TRAIN_START):
    """Days of data between `start` and `cutoff` (cutoff = the fold's cutoff, or '2025-01-27' for eval)."""
    return (epoch_of(con, cutoff) - epoch_of(con, start)) / 86400.0

def per_day(count, days):
    return count / days
