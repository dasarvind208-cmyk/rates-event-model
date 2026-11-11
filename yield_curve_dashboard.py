"""
Yield Curve Dashboard
---------------------
Pulls daily US Treasury yields from FRED (free, no API key needed),
then makes three charts and prints a quick market summary.

Run it in Google Colab (easiest) or any Python setup with pandas + matplotlib.
"""

import pandas as pd
import matplotlib.pyplot as plt

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

# ---------- SETTINGS (feel free to change these) ----------

YEARS_OF_HISTORY = 5  # how far back the time-series charts go

# FRED series IDs for each maturity, and a label for the chart
MATURITIES = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}

# Events to mark on the charts. Add your own as you follow the news,
# e.g. Fed meetings or big CPI prints. Format: "YYYY-MM-DD": "label"
EVENTS = {
    "2022-03-16": "Fed starts hiking",
    "2024-09-18": "Fed's first cut",
}

# ---------- 1. GET THE DATA ----------

def load_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    return df.iloc[:, 0].rename(series_id)


def load_all():
    print("Downloading Treasury yields from FRED...")
    data = pd.concat([load_series(s) for s in MATURITIES], axis=1)
    data = data.dropna(how="all")          # drop holidays/weekends
    return data.ffill()                    # fill any small gaps


# ---------- 2. CHARTS ----------

def closest_row(data, target_date):
    """Get the last trading day on or before target_date."""
    return data.loc[:target_date].iloc[-1]


def plot_curve_snapshots(data, ax):
    latest_date = data.index[-1]
    snapshots = {
        f"Today ({latest_date:%b %d, %Y})": closest_row(data, latest_date),
        "1 month ago": closest_row(data, latest_date - pd.DateOffset(months=1)),
        "1 year ago": closest_row(data, latest_date - pd.DateOffset(years=1)),
    }
    labels = list(MATURITIES.values())
    for name, row in snapshots.items():
        ax.plot(labels, row.values, marker="o", label=name)
    ax.set_title("US Treasury Yield Curve")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def add_events(ax, start):
    for date, label in EVENTS.items():
        d = pd.Timestamp(date)
        if d >= start:
            ax.axvline(d, color=INK2, linestyle=":", alpha=0.8)
            ax.text(d, ax.get_ylim()[1], " " + label, rotation=90,
                    va="top", fontsize=8, color=INK2)


def plot_spread(data, ax):
    start = data.index[-1] - pd.DateOffset(years=YEARS_OF_HISTORY)
    recent = data.loc[start:]
    spread = recent["DGS10"] - recent["DGS2"]
    ax.plot(spread.index, spread, color=BLUE)
    ax.axhline(0, color=RED, linewidth=1)
    # shade inverted periods (10Y below 2Y)
    ax.fill_between(spread.index, spread, 0, where=spread < 0,
                    color=RED, alpha=0.15, label="Inverted")
    ax.set_title("10Y minus 2Y Spread (below 0 = inverted)")
    ax.set_ylabel("Percentage points")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    add_events(ax, start)


def plot_2y_vs_10y(data, ax):
    start = data.index[-1] - pd.DateOffset(years=YEARS_OF_HISTORY)
    recent = data.loc[start:]
    ax.plot(recent.index, recent["DGS2"], color=ORANGE, label="2Y (tracks Fed policy)")
    ax.plot(recent.index, recent["DGS10"], color=BLUE, label="10Y (growth + inflation outlook)")
    ax.set_title("2Y vs 10Y Treasury Yields")
    ax.set_ylabel("Yield (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    add_events(ax, start)


def plot_dashboard(data):
    """All three charts side by side in one image."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    plot_curve_snapshots(data, axes[0])
    plot_spread(data, axes[1])
    plot_2y_vs_10y(data, axes[2])
    for ax in axes[1:]:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("dashboard.png", dpi=150)
    plt.show()


# ---------- 3. TEXT SUMMARY (use this for your weekly note) ----------

def bp(x):
    return f"{x * 100:+.0f}bp"


def print_summary(data):
    latest_date = data.index[-1]
    now = data.iloc[-1]
    week = closest_row(data, latest_date - pd.DateOffset(weeks=1))
    month = closest_row(data, latest_date - pd.DateOffset(months=1))

    print(f"\n===== Rates summary as of {latest_date:%A, %b %d, %Y} =====\n")
    print(f"{'Tenor':<6}{'Yield':>8}{'1W chg':>10}{'1M chg':>10}")
    for sid, label in MATURITIES.items():
        print(f"{label:<6}{now[sid]:>7.2f}%{bp(now[sid] - week[sid]):>10}"
              f"{bp(now[sid] - month[sid]):>10}")

    spread_now = now["DGS10"] - now["DGS2"]
    spread_week = week["DGS10"] - week["DGS2"]
    shape = "INVERTED" if spread_now < 0 else "upward sloping"
    move = "steepened" if spread_now > spread_week else "flattened"
    print(f"\n2s10s spread: {bp(spread_now)} ({shape})")
    print(f"Over the past week the curve {move} by "
          f"{abs(spread_now - spread_week) * 100:.0f}bp.")


# ---------- RUN EVERYTHING ----------

if __name__ == "__main__":
    data = load_all()
    print_summary(data)
    plot_dashboard(data)
    print("\nDashboard saved as dashboard.png in the current folder.")
