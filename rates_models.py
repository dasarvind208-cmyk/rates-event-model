"""
Rates Models v2: event pricing (10Y focused), recession odds, curve backtest
===========================================================================

PART 1  Event model
        Before a release (CPI, jobs, PCE) you enter consensus and your odds of
        a beat or miss. The model prices the expected move in the 10Y (main
        trade), the 2Y and the 2s10s curve, and sizes a trade. After the
        release you fill in the actual number and it drafts your post-mortem,
        including how the S&P 500 reacted.

        Coefficients (how many bp each surprise moves yields) are estimated
        from the last LOOKBACK_YEARS of releases with RECENCY WEIGHTING: a
        print from HALF_LIFE_MONTHS ago counts half as much as this month's.

PART 2  Event backtest
        Walks through your history file in order. For each release it only
        uses data available before that date, so there's no lookahead. It
        tests (a) whether the coefficients predict the actual 10Y reaction and
        (b) a simple systematic trade based on surprise momentum.

PART 3  Recession model   (10Y minus 3M spread, logistic regression)
PART 4  Curve backtest    (monthly steepener / flattener signal)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------- CHART STYLE ----------
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.alpha": 1,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
    "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED]),
})


# =============================================================================
# SETTINGS
# =============================================================================

PRIMARY = "10Y"            # main tenor you trade and grade yourself on
LOOKBACK_YEARS = 3         # only use releases from the last N years (post-COVID regime)
HALF_LIFE_MONTHS = 12      # recency weighting: weight halves every N months
MIN_OBS = 6                # releases needed before trusting a fitted coefficient
NOISE_BP = 4               # day-to-day noise unrelated to the print
MIN_EDGE_BP = 1.0          # skip trades with a smaller expected move than this
MOMENTUM_TILT = 0.15       # backtest only: how much recent surprises shift beat/miss odds

# History of past releases. Columns: date,event,consensus,actual
# Can be a file uploaded to Colab or a published Google Sheet CSV link.
HISTORY_FILE = "event_history.csv"

# Fallback coefficients, used until your history has MIN_OBS releases of a type.
#   CPI / PCE: per 1pp of core m/m   (0.1pp surprise x 70 = 7bp)
#   NFP:       per 1k jobs           (100k surprise x 0.07 = 7bp)
#   UNEMP:     per 1pp               (higher unemployment = lower yields)
# Rough starting guesses only. The history file replaces them.
DEFAULTS = {
    "CPI":   {"beta_2y": 100,  "beta_10y": 70,   "typical_surprise": 0.1},
    "PCE":   {"beta_2y": 50,   "beta_10y": 35,   "typical_surprise": 0.05},
    "NFP":   {"beta_2y": 0.10, "beta_10y": 0.07, "typical_surprise": 75},
    "UNEMP": {"beta_2y": -80,  "beta_10y": -55,  "typical_surprise": 0.1},
}


# =============================================================================
# YOUR WEEKLY INPUTS
# =============================================================================
# Upcoming releases. prob_above / prob_below = your odds vs consensus.
# Fill in "actual" after the release and rerun for the post-mortem.
# EXAMPLES BELOW: replace with real dates and consensus numbers.
EVENT_LOG = [
    {"date": "2026-11-06", "event": "NFP", "consensus": 120, "prob_above": 0.35, "prob_below": 0.50, "actual": None},
    {"date": "2026-11-12", "event": "CPI", "consensus": 0.3, "prob_above": 0.45, "prob_below": 0.20, "actual": None},
]


# =============================================================================
# DATA
# =============================================================================

_cache = {}

def fred(series_id):
    if series_id not in _cache:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
        _cache[series_id] = df.iloc[:, 0].rename(series_id).dropna()
    return _cache[series_id]


def day_change(series, date, pct=False):
    d = pd.Timestamp(date)
    if d not in series.index:
        return np.nan
    prev = series.loc[:d - pd.Timedelta(days=1)]
    if prev.empty:
        return np.nan
    if pct:
        return (series.loc[d] / prev.iloc[-1] - 1) * 100
    return (series.loc[d] - prev.iloc[-1]) * 100     # yields: percent -> bp


def add_market_moves(df):
    y2, y10 = fred("DGS2"), fred("DGS10")
    try:
        spx = fred("SP500")
    except Exception:
        spx = None
    df["move_2y"] = [day_change(y2, d) for d in df["date"]]
    df["move_10y"] = [day_change(y10, d) for d in df["date"]]
    df["move_curve"] = df["move_10y"] - df["move_2y"]
    df["spx_pct"] = [day_change(spx, d, pct=True) if spx is not None else np.nan for d in df["date"]]
    return df


def load_history():
    """History file + any completed EVENT_LOG entries, last LOOKBACK_YEARS only."""
    frames = []
    if HISTORY_FILE and (HISTORY_FILE.startswith("http") or os.path.exists(HISTORY_FILE)):
        h = pd.read_csv(HISTORY_FILE)
        h.columns = [c.strip().lower() for c in h.columns]
        frames.append(h[["date", "event", "consensus", "actual"]])
    else:
        print(f"(No {HISTORY_FILE} found, using default coefficients. "
              f"Upload it to Colab to enable calibration and the backtest.)")
    done = [e for e in EVENT_LOG if e["actual"] is not None]
    if done:
        frames.append(pd.DataFrame(done)[["date", "event", "consensus", "actual"]])
    if not frames:
        return pd.DataFrame(columns=["date", "event", "consensus", "actual", "surprise"])

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["event"] = df["event"].str.upper().str.strip()
    df = df.drop_duplicates(["date", "event"], keep="last")
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=LOOKBACK_YEARS)
    df = df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)
    df["surprise"] = df["actual"] - df["consensus"]
    return add_market_moves(df)


# =============================================================================
# RECENCY-WEIGHTED COEFFICIENTS
# =============================================================================

def recency_weights(dates, asof):
    age_months = (asof - dates).dt.days / 30.44
    return 0.5 ** (age_months / HALF_LIFE_MONTHS)


def fit_coefficients(hist, event, asof=None):
    """Recency-weighted betas using only releases before `asof`."""
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today()
    d = DEFAULTS[event]
    out = {"beta_2y": d["beta_2y"], "beta_10y": d["beta_10y"],
           "typical_surprise": d["typical_surprise"], "source": "defaults", "n": 0}
    if hist.empty:
        return out
    g = hist[(hist["event"] == event) & (hist["date"] < asof)].dropna(subset=["move_10y", "move_2y"])
    nz = g[g["surprise"] != 0]
    if len(nz) < MIN_OBS:
        out["n"] = len(nz)
        return out
    w = recency_weights(nz["date"], asof)
    s = nz["surprise"]
    for col, key in (("move_2y", "beta_2y"), ("move_10y", "beta_10y")):
        out[key] = float((w * s * nz[col]).sum() / (w * s ** 2).sum())
    wg = recency_weights(g["date"], asof)
    out["typical_surprise"] = float(np.sqrt((wg * g["surprise"] ** 2).sum() / wg.sum()))
    out["source"], out["n"] = f"fitted on {len(nz)} prints", len(nz)
    return out


# =============================================================================
# PRICING
# =============================================================================

def price_event(pa, pb, coefs):
    probs = np.array([pa, 1 - pa - pb, pb])            # above, in line, below
    shock = np.array([1, 0, -1]) * coefs["typical_surprise"]
    legs = {"2Y": coefs["beta_2y"] * shock, "10Y": coefs["beta_10y"] * shock}
    legs["2s10s"] = legs["10Y"] - legs["2Y"]

    res = {"probs": probs, "legs": legs}
    for name, moves in legs.items():
        noise = NOISE_BP / 2 if name == "2s10s" else NOISE_BP
        exp = probs @ moves
        std = np.sqrt(probs @ (moves - exp) ** 2 + noise ** 2)
        edge = MIN_EDGE_BP / 2 if name == "2s10s" else MIN_EDGE_BP
        pos = 0.0 if abs(exp) < edge else float(np.clip(exp / std, -1, 1))
        res[name] = {"exp": exp, "std": std, "pos": pos}
    return res


def trade_text(name, pos):
    if pos == 0:
        return f"No {name} trade (edge too small)"
    if name == "2s10s":
        return f"2s10s {'STEEPENER' if pos > 0 else 'FLATTENER'}, size {abs(pos):.2f}"
    side = "SHORT" if pos > 0 else "LONG"
    view = "yields rise" if pos > 0 else "yields fall"
    return f"{side} {name} (bet {view}), size {abs(pos):.2f}"


LEG_ORDER = [PRIMARY] + [t for t in ("10Y", "2Y", "2s10s") if t != PRIMARY]


def print_ticket(e, coefs, p):
    print(f"\n--- {e['event']} on {e['date']} | consensus {e['consensus']} "
          f"| coefficients: {coefs['source']} ---")
    print(f"{'Scenario':<10}{'Prob':>6}" + "".join(f"{t:>10}" for t in LEG_ORDER))
    for i, name in enumerate(["Above", "In line", "Below"]):
        row = "".join(f"{p['legs'][t][i]:>+8.1f}bp" for t in LEG_ORDER)
        print(f"{name:<10}{p['probs'][i]:>6.0%}{row}")
    for t in LEG_ORDER:
        tag = "  <- main trade" if t == PRIMARY else ""
        print(f"Expected {t:<6}{p[t]['exp']:>+6.1f}bp  (+/- {p[t]['std']:.1f})  "
              f"{trade_text(t, p[t]['pos'])}{tag}")


# =============================================================================
# PART 1: WEEKLY EVENT MODEL
# =============================================================================

def post_mortem(e, hist):
    asof = pd.Timestamp(e["date"])
    coefs = fit_coefficients(hist, e["event"], asof)
    p = price_event(e["prob_above"], e["prob_below"], coefs)
    row = add_market_moves(pd.DataFrame([{"date": asof}])).iloc[0]
    if np.isnan(row["move_10y"]) or np.isnan(row["move_2y"]):
        print(f"\n{e['event']} {e['date']}: yield data not on FRED yet, rerun tomorrow.")
        return None

    surprise = e["actual"] - e["consensus"]
    realized = {"10Y": row["move_10y"], "2Y": row["move_2y"], "2s10s": row["move_curve"]}
    model = {"10Y": coefs["beta_10y"] * surprise, "2Y": coefs["beta_2y"] * surprise}
    model["2s10s"] = model["10Y"] - model["2Y"]

    view = np.sign(e["prob_above"] - e["prob_below"])
    if view == 0:
        call = "no directional view"
    elif np.sign(surprise) == view:
        call = "RIGHT"
    else:
        call = "WRONG"

    r, m = realized[PRIMARY], model[PRIMARY]
    gap = r - m
    if surprise == 0:
        reaction = ("Print was in line, so the move came from details (revisions, "
                    "core vs headline) or other news.")
    elif abs(gap) < 3:
        reaction = "Market reacted about how the model expected."
    elif np.sign(r) != np.sign(m):
        reaction = ("Yields moved OPPOSITE to the surprise. Check for other news, revisions, "
                    "or a move that was already priced in.")
    elif abs(r) > abs(m):
        reaction = ("Market cared MORE than the model assumes. This data may matter more to the "
                    "Fed right now, or positioning was offside.")
    else:
        reaction = ("Market cared LESS than the model assumes. The surprise was likely priced "
                    "in already or something else dominated.")

    pnl = {t: p[t]["pos"] * realized[t] for t in LEG_ORDER}
    lines = [
        f"\n========== POST-MORTEM: {e['event']} ({e['date']}) ==========",
        f"Result: {e['actual']} vs {e['consensus']} consensus (surprise {surprise:+g})",
        f"Thesis: {e['prob_above']:.0%} above / {e['prob_below']:.0%} below, expected "
        f"{PRIMARY} {p[PRIMARY]['exp']:+.1f}bp. Call was {call}.",
        f"Reaction ({PRIMARY}): model said {m:+.1f}bp for this surprise, actual {r:+.1f}bp "
        f"(gap {gap:+.1f}bp).",
        f"  -> {reaction}",
        f"Other legs: 2Y {realized['2Y']:+.1f}bp (model {model['2Y']:+.1f}), "
        f"2s10s {realized['2s10s']:+.1f}bp (model {model['2s10s']:+.1f})",
    ]
    if not np.isnan(row["spx_pct"]):
        lines.append(f"Equities: S&P 500 {row['spx_pct']:+.2f}% on the day "
                     f"(10Y as discount rate: {'consistent' if np.sign(row['spx_pct']) == -np.sign(r) else 'not the driver'})")
    for t in LEG_ORDER:
        lines.append(f"Trade: {trade_text(t, p[t]['pos'])} -> P&L {pnl[t]:+.1f}bp")
    if surprise != 0:
        lines.append(f"Pricing update: this print implied {r / surprise:.3g} bp per unit "
                     f"for the {PRIMARY} vs model {coefs['beta_' + PRIMARY.lower()]:.3g}.")
    lines.append("Lesson (write your own): ________________________________")
    print("\n".join(lines))
    return {"pnl": pnl[PRIMARY], "call": call}


def run_event_model():
    print("\n#################### PART 1: WEEKLY EVENT MODEL ####################")
    hist = load_history()

    print(f"\nCurrent coefficients (last {LOOKBACK_YEARS}y, {HALF_LIFE_MONTHS}-month half-life):")
    print(f"{'Event':<7}{'2Y beta':>10}{'10Y beta':>10}{'Typical surprise':>18}  Source")
    for ev in DEFAULTS:
        c = fit_coefficients(hist, ev)
        print(f"{ev:<7}{c['beta_2y']:>10.3g}{c['beta_10y']:>10.3g}{c['typical_surprise']:>18.3g}  {c['source']}")

    results = []
    for e in sorted(EVENT_LOG, key=lambda x: x["date"]):
        if e["actual"] is not None:
            r = post_mortem(e, hist)
            if r:
                results.append(r)

    upcoming = [e for e in EVENT_LOG if e["actual"] is None]
    if upcoming:
        print("\n========== UPCOMING TRADE TICKETS ==========")
        for e in sorted(upcoming, key=lambda x: x["date"]):
            c = fit_coefficients(hist, e["event"])
            print_ticket(e, c, price_event(e["prob_above"], e["prob_below"], c))

    if results:
        calls = [r for r in results if r["call"] in ("RIGHT", "WRONG")]
        hit = np.mean([r["call"] == "RIGHT" for r in calls]) if calls else float("nan")
        print(f"\nScorecard: {len(results)} logged | direction hit rate {hit:.0%} | "
              f"{PRIMARY} P&L {sum(r['pnl'] for r in results):+.1f}bp")
    return hist


# =============================================================================
# PART 2: EVENT BACKTEST (walk-forward, last LOOKBACK_YEARS)
# =============================================================================

def run_event_backtest(hist):
    print("\n#################### PART 2: EVENT BACKTEST ####################")
    hist = hist.dropna(subset=["move_10y", "move_2y"])
    if len(hist) < MIN_OBS * 2:
        print(f"Need more history to backtest ({len(hist)} usable releases). "
              f"Fill in {HISTORY_FILE} with the last {LOOKBACK_YEARS} years of releases.")
        return

    rows = []
    for i, ev in hist.iterrows():
        prior = hist[(hist["event"] == ev["event"]) & (hist["date"] < ev["date"])]
        if len(prior[prior["surprise"] != 0]) < MIN_OBS:
            continue
        c = fit_coefficients(hist, ev["event"], ev["date"])

        # (a) reaction test: given the ACTUAL surprise, did the model call the move?
        pred = c["beta_10y"] * ev["surprise"]

        # (b) systematic trade: odds tilted by the last 3 surprises (stand-in for your judgment)
        mom = np.sign(prior["surprise"].tail(3)).mean()
        pa, pb = 0.4 + MOMENTUM_TILT * mom, 0.4 - MOMENTUM_TILT * mom
        p = price_event(pa, pb, c)
        rows.append({"date": ev["date"], "event": ev["event"], "surprise": ev["surprise"],
                     "pred_10y": pred, "move_10y": ev["move_10y"],
                     "pos": p[PRIMARY]["pos"],
                     "pnl": p[PRIMARY]["pos"] * ev["move_" + PRIMARY.lower()]})
    bt = pd.DataFrame(rows)
    if bt.empty:
        print(f"Not enough releases per type yet (need {MIN_OBS} before the first test).")
        return

    nz = bt[bt["surprise"] != 0]
    hit_dir = (np.sign(nz["pred_10y"]) == np.sign(nz["move_10y"])).mean()
    corr = nz["pred_10y"].corr(nz["move_10y"])
    print(f"Tested {len(bt)} releases from {bt['date'].min():%b %Y} to {bt['date'].max():%b %Y}")
    print(f"\n(a) Reaction model, {PRIMARY}: direction right {hit_dir:.0%} of the time when there "
          f"was a surprise | correlation of predicted vs actual move {corr:.2f}")

    traded = bt[bt["pos"] != 0]
    if len(traded):
        win = (traded["pnl"] > 0).mean()
        print(f"(b) Momentum trade, {PRIMARY}: {len(traded)} trades | win rate {win:.0%} | "
              f"total P&L {traded['pnl'].sum():+.1f}bp | avg {traded['pnl'].mean():+.2f}bp/trade")
        print("\nBy release:")
        summary = traded.groupby("event")["pnl"].agg(trades="count", total="sum", avg="mean")
        print(summary.round(2).to_string())
    else:
        print("(b) Momentum trade: no trades cleared the edge threshold.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for ev, g in nz.groupby("event"):
        ax.scatter(g["pred_10y"], g["move_10y"], label=ev, s=40, edgecolor=SURFACE, linewidth=1.5)
    lim = max(nz["pred_10y"].abs().max(), nz["move_10y"].abs().max(), 1) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=INK2, linestyle=":")
    ax.set_xlabel("Model predicted 10Y move (bp)")
    ax.set_ylabel("Actual 10Y move (bp)")
    ax.set_title("Reaction model: predicted vs actual")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(bt["date"], bt["pnl"].cumsum(), color=BLUE)
    ax.axhline(0, color=RED, linewidth=0.8)
    ax.set_title(f"Momentum trade cumulative P&L ({PRIMARY}, bp)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("6_event_backtest.png", dpi=150)
    plt.show()


# =============================================================================
# PART 3: RECESSION MODEL (uses long history on purpose: recessions are rare)
# =============================================================================

def run_recession_model(ax=None):
    print("\n#################### PART 3: RECESSION MODEL ####################")
    spread = fred("T10Y3M").resample("MS").mean()
    rec = fred("USREC")
    df = pd.concat([spread.rename("spread"), rec.rename("rec")], axis=1).dropna(subset=["spread"])
    df["target"] = df["rec"].shift(-12)
    train = df.dropna()
    model = sm.Logit(train["target"], sm.add_constant(train["spread"])).fit(disp=0)
    df["prob"] = model.predict(sm.add_constant(df["spread"]))

    now = df.iloc[-1]
    print(f"10Y-3M spread ({df.index[-1]:%b %Y}): {now['spread']:+.2f}pp")
    print(f"Model odds of recession 12 months out: {now['prob']:.0%}")
    print("(Compare with the NY Fed's number: search 'NY Fed yield curve recession probability')")
    u = fred("UNRATE")
    ma3 = u.rolling(3).mean()
    sahm_now = (ma3 - ma3.shift(1).rolling(12).min()).iloc[-1]
    status = "TRIGGERED (recession signal)" if sahm_now >= 0.5 else "not triggered"
    print(f"Sahm rule ({u.index[-1]:%b %Y}): {sahm_now:.2f}pp vs 0.50 threshold -> {status}")
    print("(Curve model = leading, ~12 months ahead. Sahm rule = real time, flags a recession once it has started.)")

    solo = ax is None
    if solo:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["prob"] * 100, color=BLUE, label="Model probability")
    ax.fill_between(df.index, 0, 100, where=df["rec"].fillna(0) > 0, color=INK2,
                    alpha=0.15, label="Actual recessions")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Probability of Recession in 12 Months (10Y-3M model)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    if solo:
        plt.tight_layout()
        plt.savefig("4_recession_probability.png", dpi=150)
        plt.show()


# =============================================================================
# PART 4: CURVE BACKTEST
# =============================================================================

def run_curve_backtest(ax=None, min_train_months=120):
    print("\n#################### PART 4: CURVE TRADE BACKTEST ####################")
    y2 = fred("DGS2").resample("MS").last()
    y10 = fred("DGS10").resample("MS").last()
    ff = fred("FEDFUNDS").resample("MS").last()
    cpi = fred("CPIAUCSL").resample("MS").last()

    df = pd.DataFrame({"spread": (y10 - y2) * 100})
    df["level"] = df["spread"]
    df["ff_chg6"] = ff.diff(6)
    df["cpi_yoy"] = (cpi.pct_change(12) * 100).shift(1)
    df["mom3"] = df["spread"].diff(3)
    df["target"] = df["spread"].shift(-1) - df["spread"]
    feats = ["level", "ff_chg6", "cpi_yoy", "mom3"]
    df = df.dropna(subset=feats)

    rows = []
    for i in range(min_train_months, len(df) - 1):
        train = df.iloc[:i].dropna()
        fit = sm.OLS(train["target"], sm.add_constant(train[feats])).fit()
        pred = float(fit.predict(sm.add_constant(df[feats].iloc[[i]], has_constant="add")).iloc[0])
        pos = np.sign(pred)
        rows.append({"date": df.index[i], "pred": pred, "actual": df["target"].iloc[i],
                     "pnl": pos * df["target"].iloc[i]})
    bt = pd.DataFrame(rows).set_index("date")

    def stats(b, label):
        p = b["pnl"]
        cum = p.cumsum()
        print(f"{label}: hit rate {(np.sign(b['pred']) == np.sign(b['actual'])).mean():.0%} | "
              f"avg {p.mean():+.2f}bp/month | Sharpe {p.mean() / p.std() * np.sqrt(12):.2f} | "
              f"max drawdown {(cum - cum.cummax()).min():.0f}bp")

    stats(bt, f"Full test ({bt.index[0]:%Y}-{bt.index[-1]:%Y})")
    recent = bt[bt.index >= bt.index[-1] - pd.DateOffset(years=LOOKBACK_YEARS)]
    stats(recent, f"Last {LOOKBACK_YEARS} years only")

    full = df.dropna()
    fit = sm.OLS(full["target"], sm.add_constant(full[feats])).fit()
    pred_now = float(fit.predict(sm.add_constant(df[feats].iloc[[-1]], has_constant="add")).iloc[0])
    print(f"Signal for next month: {'STEEPENER' if pred_now > 0 else 'FLATTENER'} "
          f"(expects {pred_now:+.1f}bp change in 2s10s)")

    solo = ax is None
    if solo:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt.index, bt["pnl"].cumsum(), label="Model strategy", color=BLUE)
    ax.plot(bt.index, bt["actual"].cumsum(), label="Always steepener", color=ORANGE)
    ax.axhline(0, color=RED, linewidth=0.8)
    ax.set_title("Curve Trade Backtest: cumulative P&L (bp of 2s10s)")
    ax.set_ylabel("bp")
    ax.legend()
    ax.grid(alpha=0.3)
    if solo:
        plt.tight_layout()
        plt.savefig("5_curve_backtest.png", dpi=150)
        plt.show()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    hist = pd.DataFrame()
    try:
        hist = run_event_model()
    except Exception as err:
        print(f"\nEvent model failed: {err}")
    try:
        run_event_backtest(hist)          # already two charts side by side
    except Exception as err:
        print(f"\nEvent backtest failed: {err}")

    # recession model and curve backtest side by side in one image
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for part, ax in ((run_recession_model, axes[0]), (run_curve_backtest, axes[1])):
        try:
            part(ax=ax)
        except Exception as err:
            print(f"\n{part.__name__} failed: {err}")
            ax.set_visible(False)
    plt.tight_layout()
    plt.savefig("models_summary.png", dpi=150)
    plt.show()
