"""Stage 2c: short exploration summary -> exploration_summary.md"""
from config import *
from db import connect, epoch_of, fold_name

def summarize(con, src, cutoff, has_label):
    lo = epoch_of(con, cutoff)
    lab = "clicked" if has_label else "NULL::DOUBLE"
    inh = "contains(history_item_ids, item_id)"           # is this candidate already in the user's history?
    r = con.execute(f"""
        SELECT count(*), count(DISTINCT event_id), count(DISTINCT user_id),
               min(timestamp), max(timestamp),
               avg({inh}::INT),
               avg({lab}), avg({lab}) FILTER (WHERE {inh}), avg({lab}) FILTER (WHERE NOT {inh}),
               sum({lab}) FILTER (WHERE {inh}) / nullif(sum({lab}), 0),
               avg((timestamp < {lo} + 86400)::INT)
        FROM {src}""").fetchone()
    e = con.execute(f"""
        WITH ev AS (SELECT event_id, any_value(history_timestamps) h, sum({lab}) c FROM {src} GROUP BY 1),
             hl AS (SELECT c, len(CAST(h AS BIGINT[])) n FROM ev)
        SELECT avg((c = 0)::INT), quantile_cont(n, [0.5, 0.9, 0.99]), max(n), avg((n = 0)::INT) FROM hl""").fetchone()
    # users with no earlier event in the training data (before the cutoff)
    unseen = con.execute(f"""
        SELECT avg((user_id NOT IN (SELECT DISTINCT user_id FROM train WHERE timestamp < {lo}))::INT)
        FROM (SELECT DISTINCT user_id FROM {src})""").fetchone()[0]
    return dict(rows=r[0], events=r[1], users=r[2], t0=r[3], t1=r[4], in_hist=r[5],
                click_row=r[6], click_in=r[7], click_out=r[8], clicks_in_hist=r[9], day0=r[10],
                no_click=e[0], h50=e[1][0], h90=e[1][1], h99=e[1][2], hmax=e[2], empty_hist=e[3], unseen=unseen)

def pct(x): return "n/a" if x is None else f"{x:.1%}"
def when(con, t): return con.execute(f"SELECT make_timestamp({int(t)}::BIGINT * 1000000)").fetchone()[0]

if __name__ == "__main__":
    con = connect()
    sets = {f"Validation fold {c} (cutoff {c} 00:00 UTC)": summarize(con, fold_name(c), c, True) for c in CUTOFFS}
    sets[f"EVALUATION set (cutoff {DATA_END} 00:00 UTC)"] = summarize(con, "evalset", DATA_END, False)
    train = summarize(con, "train", "2025-01-20", True)

    days = con.execute("""SELECT strftime(ts_utc, '%Y-%m-%d'), count(DISTINCT event_id) FROM evalset
                          GROUP BY 1 ORDER BY 1""").fetchall()
    out = ["# Exploration summary", "",
           "Data: each EVENT has exactly 10 candidate items (rows); at most one of them is clicked. "
           "History = the user's last <=20 clicked items before the event.", "",
           "## Whole training set",
           f"- {train['rows']:,} rows = {train['events']:,} events, {train['users']:,} users, "
           f"{when(con, train['t0'])} -> {when(con, train['t1'])} UTC",
           f"- **Share of events with no click: {pct(train['no_click'])}**",
           f"- Click rate per candidate row: {pct(train['click_row'])}",
           f"- **Click rate for candidates already in the user's history: {pct(train['click_in'])}** "
           f"(vs {pct(train['click_out'])} for items not in history); "
           f"{pct(train['in_hist'])} of candidates are in history; "
           f"{pct(train['clicks_in_hist'])} of all clicks are on a history item",
           f"- **History length** (items): median {train['h50']:.0f}, p90 {train['h90']:.0f}, p99 {train['h99']:.0f}, "
           f"max {train['hmax']}; events with empty history: {pct(train['empty_hist'])}", ""]
    for name, s in sets.items():
        out += [f"## {name}",
                f"- {s['users']:,} users (1 event each), {s['rows']:,} rows",
                f"- **Time range: {when(con, s['t0'])} -> {when(con, s['t1'])} UTC**; "
                f"{pct(s['day0'])} of events fall in the first 24h after the cutoff",
                f"- **History length**: median {s['h50']:.0f}, p90 {s['h90']:.0f}, p99 {s['h99']:.0f}, max {s['hmax']}",
                f"- **Share of new users**: {pct(s['empty_hist'])} have an empty history; "
                f"{pct(s['unseen'])} have no event in the training data before the cutoff",
                f"- Share of candidates already in the user's history: {pct(s['in_hist'])}"]
        if s["click_row"] is not None:
            out += [f"- **Share of events with no click: {pct(s['no_click'])}**",
                    f"- **Click rate, candidate in history: {pct(s['click_in'])}** | not in history: {pct(s['click_out'])}"]
        out.append("")
    out += ["## Evaluation events per day (first event of each user)"] + [f"- {d}: {n:,}" for d, n in days]
    text = "\n".join(out) + "\n"
    open("exploration_summary.md", "w", encoding="utf-8").write(text)
    print(text)
