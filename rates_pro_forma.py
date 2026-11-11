"""
Rates Pro Forma: 3-year projection of the yield curve
=====================================================
Paste this in its own Colab cell (cell 3) and run it.

How the projection works (a building-block model):
  - You set a Fed funds path for each scenario (cuts or hikes over the
    next 1, 2 and 3 years), a long-run neutral rate, and a term premium view.
  - 3M yield  = Fed funds at that date (+ today's small spread)
  - 2Y yield  = average expected Fed funds over the NEXT 2 years + premium
  - 10Y yield = average expected short rate over the NEXT 10 years
                + term premium (+ your term premium change)
  - Premiums (including the 10Y term premium) are calibrated ONCE off the Base
    path and shared by every scenario, so scenarios differ only by your inputs.
  - All scenarios start at today's yields. The market "learns" each scenario's
    Fed path over LEARN_MONTHS, so they fan out instead of jumping on day 1.
  - The shaded band around the Base case is a 68% range built from how much
    yields have actually moved over the last 3 years.
  - Purple hollow diamonds are MARKET-IMPLIED FORWARDS: the yield today's curve
    already prices in for that future date (math on today's curve, not a Fed
    projection, so not the same thing as the Fed's dot plot). Compare your scenarios to them: if your Base is below
    the dots, you're more dovish than the market and that's a trade idea.
  - Muni module: converts each scenario's 10Y into a muni yield with the
    muni/Treasury ratio, then shows tax-equivalent yield by bracket vs
    Treasuries and corporates (the wealth management / UHNW angle).
  - Recession signals: the Sahm rule (charted, from your unemployment path) and
    the 10Y minus 3M curve model (in the tables).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =============================================================================
# CHART STYLE
# =============================================================================
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
    "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED]),
})

# =============================================================================
# YOUR ASSUMPTIONS  (edit these)
# =============================================================================
HISTORY_YEARS = 3       # history shown on the charts
DRIFT_YEARS = 3         # after year 3, short rates drift to the neutral rate over this many years
BAND_SIGMA = 1.0        # width of the uncertainty band (1.0 = ~68% range)
LEARN_MONTHS = 6        # months for the market to fully price a scenario's Fed path

# Fed funds change from TODAY in bp at +1, +2 and +3 years.
# Neutral = long-run Fed funds (%). Term premium change = bp vs today at +1/+2/+3 years.
# Unemployment change = pp vs today at +1/+2/+3 years (drives the Sahm rule).
# THESE ARE PLACEHOLDER VIEWS. Replace them with your own thesis.
SCENARIOS = {
    "Base":    {"color": BLUE,   "fed_change_bp": [-50, -75, -75],   "neutral": 3.00,
                "tp_change_bp": [0, 10, 15], "unemp_change": [0.1, 0.2, 0.2],
                "story": "Gradual cuts, inflation drifts toward target"},
    "Hawkish": {"color": ORANGE, "fed_change_bp": [25, 50, 25],      "neutral": 3.50,
                "tp_change_bp": [20, 35, 40], "unemp_change": [-0.1, 0.0, 0.1],
                "story": "Sticky inflation, Fed hikes, deficits push term premium up"},
    "Dovish":  {"color": AQUA,   "fed_change_bp": [-150, -225, -225], "neutral": 2.50,
                "tp_change_bp": [-20, -15, -10], "unemp_change": [0.3, 0.4, 0.3],
                "story": "Growth scare, inflation cools fast, aggressive cuts (no recession)"},
}

# ---------- MUNI / WEALTH MANAGEMENT SETTINGS ----------
# Muni/Treasury ratio = 10Y AAA muni yield / 10Y Treasury yield.
# High ratio = munis cheap vs Treasuries. PLACEHOLDER: update with the current
# 10Y ratio from a news source or MMD/Bloomberg if you have access.
MUNI_RATIO_TODAY = 0.70
# Ratio each scenario drifts to by +3 years (heavy issuance / fund outflows cheapen munis)
MUNI_RATIO_3Y = {"Base": 0.70, "Hawkish": 0.76, "Dovish": 0.66}
# Federal brackets to show. 0.408 = 37% top rate + 3.8% net investment income tax
FED_BRACKETS = [0.24, 0.32, 0.35, 0.37, 0.408]
STATE_TAX = 0.109          # client's state rate for in-state munis (0.133 CA, 0.109 NY, 0 for FL/TX)
CORP_SPREAD_BP = 80        # A-rated corporate spread over the 10Y Treasury (placeholder)


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


SPOT_TENORS = {"DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2, "DGS3": 3,
               "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30}


def today_curve():
    pts = {t: fred(s).iloc[-1] for s, t in SPOT_TENORS.items()}
    return pd.Series(pts).sort_index()


def spot(curve, t):
    return float(np.interp(t, curve.index, curve.values))


def forward(curve, start, length):
    """Market-implied yield on a `length`-year bond, `start` years from now."""
    a, b = spot(curve, start) / 100, spot(curve, start + length) / 100
    return (((1 + b) ** (start + length) / (1 + a) ** start) ** (1 / length) - 1) * 100


# =============================================================================
# MODEL
# =============================================================================
H = 36                                    # projection horizon in months
LONG = H + 12 * DRIFT_YEARS + 121          # enough months to average 10 years ahead


def fed_path(ff0, sc):
    knots_m = [0, 12, 24, 36, 36 + 12 * DRIFT_YEARS, LONG]
    knots_v = [ff0] + [ff0 + c / 100 for c in sc["fed_change_bp"]] + [sc["neutral"], sc["neutral"]]
    return np.maximum(np.interp(np.arange(LONG + 1), knots_m, knots_v), 0)


def tp_path(sc):
    return np.interp(np.arange(H + 1), [0, 12, 24, 36], [0] + sc["tp_change_bp"]) / 100


def calibrate(ff0, now):
    """Premiums implied by today's yields if the market expects the Base path.
    Computed once and shared by all scenarios."""
    base = fed_path(ff0, SCENARIOS["Base"])
    return {"3M": now["DGS3MO"] - base[0],
            "2Y": now["DGS2"] - base[:24].mean(),
            "10Y": now["DGS10"] - base[:120].mean()}


def project(sc, ff0, prem):
    base = fed_path(ff0, SCENARIOS["Base"])
    path = fed_path(ff0, sc)
    m2, m10 = [], []
    for h in range(H + 1):
        w = min(h / LEARN_MONTHS, 1.0)            # how much the market has priced this scenario
        exp_path = (1 - w) * base + w * path
        m2.append(exp_path[h:h + 24].mean())
        m10.append(exp_path[h:h + 120].mean())
    fed = path[:H + 1]
    return pd.DataFrame({"fed": fed,
                         "3M": fed + prem["3M"],
                         "2Y": np.array(m2) + prem["2Y"],
                         "10Y": np.array(m10) + prem["10Y"] + tp_path(sc)})


def sahm(u):
    """Sahm rule: 3-month avg unemployment minus its low over the prior 12 months."""
    ma3 = u.rolling(3).mean()
    return ma3 - ma3.shift(1).rolling(12).min()


def smooth_path(x, knots_x, knots_y):
    """Like np.interp but eases in and out of each knot, so paths curve instead of kinking."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        j = min(np.searchsorted(knots_x, xi, side="right") - 1, len(knots_x) - 2)
        t = np.clip((xi - knots_x[j]) / (knots_x[j + 1] - knots_x[j]), 0, 1)
        t = t * t * (3 - 2 * t)                       # smoothstep easing
        out[i] = knots_y[j] + t * (knots_y[j + 1] - knots_y[j])
    return out


def sahm_projection(sc):
    u = fred("UNRATE")
    u = u[u.index >= u.index[-1] - pd.DateOffset(years=HISTORY_YEARS + 2)]
    months = pd.DatetimeIndex([u.index[-1] + pd.DateOffset(months=h) for h in range(1, H + 1)])
    path = u.iloc[-1] + smooth_path(np.arange(1, H + 1), [0, 12, 24, 36], [0] + sc["unemp_change"])
    full = pd.concat([u, pd.Series(path, index=months)])
    return sahm(full).loc[u.index[-1]:]


def recession_model():
    spread = fred("T10Y3M").resample("MS").mean()
    df = pd.concat([spread.rename("spread"), fred("USREC").rename("rec")], axis=1)
    df["target"] = df["rec"].shift(-12)
    train = df.dropna()
    return sm.Logit(train["target"], sm.add_constant(train["spread"])).fit(disp=0)


def monthly_vol(series_id, years=3):
    m = fred(series_id).resample("MS").last()
    m = m[m.index >= m.index[-1] - pd.DateOffset(years=years)]
    return m.diff().std()


def spread_end_labels(ax, min_gap_pt=12):
    """Nudge line-end labels apart so they never overlap."""
    from matplotlib.text import Annotation
    anns = [t for t in ax.texts if isinstance(t, Annotation)]
    if len(anns) < 2:
        return
    to_pt = 72 / ax.figure.dpi
    ys = [ax.transData.transform((0, float(a.xy[1])))[1] * to_pt for a in anns]
    order = np.argsort(ys)
    placed = []
    for i in order:
        y = ys[i] if not placed else max(ys[i], placed[-1] + min_gap_pt)
        placed.append(y)
        anns[i].xyann = (6, y - ys[i])
    # if the stack drifted up, shift it back down to center on the originals
    shift = (np.mean(placed) - np.mean([ys[i] for i in order])) / 2
    for i in order:
        x, dy = anns[i].xyann
        anns[i].xyann = (x, dy - shift)


# =============================================================================
# RUN
# =============================================================================

def run_pro_forma():
    now = {s: fred(s).iloc[-1] for s in ["DGS3MO", "DGS2", "DGS10"]}
    ff0 = fred("DFF").iloc[-1]
    asof = fred("DGS10").index[-1]
    dates = pd.DatetimeIndex([asof + pd.DateOffset(months=h) for h in range(H + 1)])
    rec_model = recession_model()
    prem = calibrate(ff0, now)

    proj, sahm_proj = {}, {}
    for name, sc in SCENARIOS.items():
        sahm_proj[name] = sahm_projection(sc)
        df = project(sc, ff0, prem)
        df.index = dates
        df["2s10s"] = (df["10Y"] - df["2Y"]) * 100
        df["rec_prob"] = rec_model.predict(sm.add_constant(df["10Y"] - df["3M"], has_constant="add")) * 100
        df["sahm"] = np.interp(np.arange(H + 1), np.arange(len(sahm_proj[name])), sahm_proj[name].values)
        proj[name] = df

    curve = today_curve()
    fwd = pd.DataFrame({
        "2Y": [now["DGS2"]] + [forward(curve, t, 2) for t in (1, 2, 3)],
        "10Y": [now["DGS10"]] + [forward(curve, t, 10) for t in (1, 2, 3)],
    }, index=[dates[0]] + [dates[12 * t] for t in (1, 2, 3)])

    # ---------- tables ----------
    print(f"\n========== RATES PRO FORMA (as of {asof:%b %d, %Y}) ==========")
    print(f"Today: Fed funds {ff0:.2f}% | 3M {now['DGS3MO']:.2f}% | 2Y {now['DGS2']:.2f}% | "
          f"10Y {now['DGS10']:.2f}%")
    print(f"Implied 10Y term premium (vs Base path): {prem['10Y'] * 100:+.0f}bp")
    cols = ["fed", "3M", "2Y", "10Y", "2s10s", "sahm", "rec_prob"]
    labels = ["Fed funds", "3M", "2Y", "10Y", "2s10s (bp)", "Sahm", "Curve recession odds (%)"]
    for name, df in proj.items():
        print(f"\n{name}: {SCENARIOS[name]['story']}")
        snap = df.iloc[[0, 12, 24, 36]][cols].copy()
        snap.columns = labels
        snap.index = ["Today", "+1Y", "+2Y", "+3Y"]
        print(snap.round(2).to_string())

    print("\nMarket-implied forwards (what today's curve already prices in):")
    f = fwd.copy()
    f.index = ["Today", "+1Y", "+2Y", "+3Y"]
    print(f.round(2).to_string())

    v10, v2 = monthly_vol("DGS10"), monthly_vol("DGS2")
    b = proj["Base"]["10Y"].iloc[-1]
    w = BAND_SIGMA * v10 * np.sqrt(36)
    print(f"\nBase case 10Y in 3 years: {b:.2f}% (68% range {b - w:.2f}% to {b + w:.2f}%)")
    diff = (proj["Base"]["10Y"].iloc[12] - fwd["10Y"].iloc[1]) * 100
    lean = "ABOVE" if diff > 0 else "BELOW"
    print(f"Your Base 10Y in 1 year is {abs(diff):.0f}bp {lean} the market forward "
          f"-> {'bearish duration (short 10Y)' if diff > 0 else 'bullish duration (long 10Y)'} "
          f"if you trust your Base case.")

    # ---------- charts ----------
    start = asof - pd.DateOffset(years=HISTORY_YEARS)
    hist = pd.DataFrame({t: fred(s).loc[start:] for s, t in
                         [("DGS3MO", "3M"), ("DGS2", "2Y"), ("DGS10", "10Y")]}).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    h_months = np.arange(H + 1)

    def yield_panel(ax, tenor, vol, title):
        ax.plot(hist.index, hist[tenor], color=INK, linewidth=1.4, label="History")
        base = proj["Base"][tenor]
        band = BAND_SIGMA * vol * np.sqrt(h_months)
        ax.fill_between(dates, base - band, base + band, color=BLUE, alpha=0.12,
                        linewidth=0, label="Base 68% range")
        for name, df in proj.items():
            ax.plot(dates, df[tenor], color=SCENARIOS[name]["color"], label=name)
            ax.annotate(f"{name} {df[tenor].iloc[-1]:.2f}%", (dates[-1], df[tenor].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK2)
        ax.scatter(fwd.index[1:], fwd[tenor].iloc[1:], s=80, marker="D", facecolor=SURFACE,
                   edgecolor=VIOLET, linewidth=2, zorder=5, label="Market-implied forward (priced in today)")
        ax.axvline(asof, color=INK2, linestyle=":", linewidth=1)
        ax.text(asof, ax.get_ylim()[1], "  today", va="top", fontsize=9, color=INK2)
        ax.set_title(title)
        ax.set_ylabel("Yield (%)")
        ax.set_xlim(start, dates[-1] + pd.DateOffset(months=9))
        ax.legend(loc="upper left", ncol=3)

    yield_panel(axes[0, 0], "10Y", v10, "10Y Treasury: history and 3-year pro forma")
    yield_panel(axes[0, 1], "2Y", v2, "2Y Treasury: history and 3-year pro forma")

    ax = axes[1, 0]
    ax.plot(hist.index, (hist["10Y"] - hist["2Y"]) * 100, color=INK, linewidth=1.4, label="History")
    for name, df in proj.items():
        ax.plot(dates, df["2s10s"], color=SCENARIOS[name]["color"], label=name)
        ax.annotate(f"{name} {df['2s10s'].iloc[-1]:+.0f}bp", (dates[-1], df["2s10s"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.axhline(0, color=RED, linewidth=1)
    ax.axvline(asof, color=INK2, linestyle=":", linewidth=1)
    ax.set_title("2s10s curve: steeper or flatter from here?")
    ax.set_ylabel("bp (below 0 = inverted)")
    ax.set_xlim(start, dates[-1] + pd.DateOffset(months=9))
    ax.legend(loc="upper left", ncol=4)

    ax = axes[1, 1]
    u_hist = sahm(fred("UNRATE"))
    u_hist = u_hist[u_hist.index >= start]
    ax.plot(u_hist.index, u_hist, color=INK, linewidth=1.4, label="History")
    for name, sp in sahm_proj.items():
        ax.plot(sp.index, sp.values, color=SCENARIOS[name]["color"], label=name)
        ax.annotate(f"{name} {sp.iloc[-1]:.2f}", (sp.index[-1], sp.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.axhline(0.5, color=RED, linewidth=1, linestyle="--")
    ax.text(start, 0.5, " recession signal (0.50)", va="bottom", fontsize=9, color=INK2)
    ax.axvline(asof, color=INK2, linestyle=":", linewidth=1)
    ax.set_title("Sahm rule: does unemployment signal a recession?")
    ax.set_ylabel("pp above 12-month low")
    ax.set_xlim(start, dates[-1] + pd.DateOffset(months=9))
    ax.legend(loc="upper left", ncol=4)

    fig.suptitle(f"Rates Pro Forma  |  as of {asof:%b %d, %Y}", x=0.01, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    plt.tight_layout()
    for ax in axes.flat:
        spread_end_labels(ax)
    plt.savefig("pro_forma.png", dpi=150)
    plt.show()
    return proj


# =============================================================================
# MUNI / TAX-EQUIVALENT YIELD MODULE
# =============================================================================

def muni_projection(proj):
    out = {}
    for name, df in proj.items():
        ratio = smooth_path(np.arange(H + 1), [0, 36], [MUNI_RATIO_TODAY, MUNI_RATIO_3Y[name]])
        m = pd.DataFrame(index=df.index)
        m["treasury_10y"] = df["10Y"]
        m["ratio"] = ratio
        m["muni_10y"] = df["10Y"] * ratio
        top = FED_BRACKETS[-1]
        # vs Treasury: Treasuries are state-tax exempt, so only federal tax matters
        m["tey_vs_tsy"] = m["muni_10y"] / (1 - top)
        # vs corporates: in-state muni dodges federal AND state (SALT cap means no deduction)
        m["tey_vs_corp"] = m["muni_10y"] / (1 - top - STATE_TAX)
        m["corp_10y"] = df["10Y"] + CORP_SPREAD_BP / 100
        out[name] = m
    return out


def run_muni(proj):
    muni = muni_projection(proj)
    top = FED_BRACKETS[-1]
    print("\n========== MUNI / WEALTH MANAGEMENT VIEW ==========")
    print(f"Muni/Treasury ratio today: {MUNI_RATIO_TODAY:.0%} | top bracket {top:.1%} federal "
          f"| state {STATE_TAX:.1%} | corporate spread {CORP_SPREAD_BP}bp")
    print(f"Breakeven: munis beat Treasuries after tax for anyone above a "
          f"{1 - MUNI_RATIO_TODAY:.0%} federal rate (1 minus the ratio).")

    for name, m in muni.items():
        snap = m.iloc[[0, 12, 24, 36]][["treasury_10y", "ratio", "muni_10y", "tey_vs_tsy",
                                         "corp_10y", "tey_vs_corp"]].copy()
        snap.columns = ["10Y Tsy", "Ratio", "Muni 10Y", "TEY vs Tsy", "Corp 10Y", "TEY vs Corp"]
        snap.index = ["Today", "+1Y", "+2Y", "+3Y"]
        print(f"\n{name}  (TEY at {top:.1%} federal{', in-state' if STATE_TAX else ''})")
        print(snap.round(2).to_string())

    today = muni["Base"].iloc[0]
    print("\nToday's muni TEY by federal bracket (vs Treasury):")
    for b in FED_BRACKETS:
        tey = today["muni_10y"] / (1 - b)
        verdict = "MUNI wins" if tey > today["treasury_10y"] else "Treasury wins"
        print(f"  {b:>6.1%}: TEY {tey:.2f}% vs Tsy {today['treasury_10y']:.2f}% -> {verdict}")

    # ---------- chart ----------
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    ax = axes[0]
    ax.plot(muni["Base"].index, muni["Base"]["treasury_10y"], color=INK, linewidth=1.4,
            linestyle="--", label="10Y Treasury (Base)")
    for name, m in muni.items():
        ax.plot(m.index, m["tey_vs_tsy"], color=SCENARIOS[name]["color"], label=f"{name} muni TEY")
        ax.annotate(f"{name} {m['tey_vs_tsy'].iloc[-1]:.2f}%", (m.index[-1], m["tey_vs_tsy"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_title(f"10Y muni tax-equivalent yield at {top:.1%} vs Treasury")
    ax.set_ylabel("Yield (%)")
    ax.set_xlim(muni["Base"].index[0], muni["Base"].index[-1] + pd.DateOffset(months=7))
    ax.legend(loc="upper left", ncol=2)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    labels = [f"{b:.1%}" for b in FED_BRACKETS]
    tey_t = [today["muni_10y"] / (1 - b) for b in FED_BRACKETS]
    tey_c = [today["muni_10y"] / (1 - b - STATE_TAX) for b in FED_BRACKETS]
    x = np.arange(len(FED_BRACKETS))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, tey_t, w, color=BLUE, label="TEY vs Treasury (federal only)")
    ax.bar(x + w / 2 + 0.01, tey_c, w, color=AQUA, label=f"TEY vs corporate (fed + {STATE_TAX:.1%} state)")
    ax.axhline(today["treasury_10y"], color=INK, linestyle="--", linewidth=1.2)
    ax.text(-0.6, today["treasury_10y"], f" 10Y Tsy {today['treasury_10y']:.2f}%",
            ha="left", va="bottom", fontsize=9, color=INK2)
    ax.axhline(today["corp_10y"], color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(-0.6, today["corp_10y"], f" A corp {today['corp_10y']:.2f}%",
            ha="left", va="bottom", fontsize=9, color=INK2)
    for xi, v in zip(x, tey_c):
        ax.text(xi + w / 2 + 0.01, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_xlabel("Client federal tax rate")
    ax.set_ylabel("Yield (%)")
    ax.set_title(f"Today: who should own munis? (muni 10Y {today['muni_10y']:.2f}%)")
    ax.set_ylim(0, max(tey_c) * 1.15)
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    spread_end_labels(axes[0])
    plt.savefig("muni_tey.png", dpi=150)
    plt.show()
    return muni


if __name__ == "__main__":
    proj = run_pro_forma()
    run_muni(proj)
