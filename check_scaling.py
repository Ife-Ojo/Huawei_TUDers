"""Stage 3a check: show that per-day scaling makes count features comparable.
Builds three example count features for every row of each fold / the evaluation set, using ONLY
training events before that set's cutoff (so no leakage), and prints raw vs per-day averages."""
import pandas as pd
from config import *
from db import connect, epoch_of, fold_name
from scaling import observation_days

def example_features(con, target, cutoff):
    lo, days = epoch_of(con, cutoff), observation_days(con, cutoff)
    r = con.execute(f"""
        WITH ue AS (SELECT user_id, count(DISTINCT event_id) AS n FROM train WHERE timestamp < {lo} GROUP BY 1),
             ic AS (SELECT item_id, sum(clicked) AS clicks, count(*) AS shown FROM train WHERE timestamp < {lo} GROUP BY 1)
        SELECT avg(coalesce(ue.n, 0)), avg(coalesce(ic.clicks, 0)), avg(coalesce(ic.shown, 0))
        FROM {target} t LEFT JOIN ue ON t.user_id = ue.user_id LEFT JOIN ic ON t.item_id = ic.item_id""").fetchone()
    raw = pd.Series(r, index=["user_events", "item_clicks", "item_shown"])
    return days, raw

if __name__ == "__main__":
    con = connect()
    targets = {f"fold {c}": (fold_name(c), c) for c in CUTOFFS}
    targets["EVAL"] = ("evalset", DATA_END)
    raw, scaled, days = {}, {}, {}
    for name, (tbl, cutoff) in targets.items():
        d, r = example_features(con, tbl, cutoff)
        days[name], raw[name], scaled[name] = d, r, r / d
    pd.set_option("display.width", 200)
    print("Days of data before each cutoff:", {k: round(v, 1) for k, v in days.items()})
    print("\nRAW averages (not comparable):");        print(pd.DataFrame(raw).round(2))
    print("\nPER-DAY averages (should be close):");   print(pd.DataFrame(scaled).round(2))
    ratio = (pd.DataFrame(raw)["EVAL"].values[:, None] / pd.DataFrame(raw).drop(columns="EVAL").values)
    ratio2 = (pd.DataFrame(scaled)["EVAL"].values[:, None] / pd.DataFrame(scaled).drop(columns="EVAL").values)
    cols = [c for c in raw if c != "EVAL"]
    print("\nEVAL / fold ratio, raw:     ", pd.DataFrame(ratio, index=raw[cols[0]].index, columns=cols).round(2).to_dict("index"))
    print("EVAL / fold ratio, per day: ", pd.DataFrame(ratio2, index=raw[cols[0]].index, columns=cols).round(2).to_dict("index"))
