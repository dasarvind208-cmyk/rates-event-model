# Rates Event Model

A Python toolkit for pricing how US economic data releases move Treasury yields, trading around them, and projecting the yield curve forward.

Before each major release (CPI, PCE, jobs report), the model combines analyst consensus with my own odds of a beat or miss. It turns that into a probability-weighted expected move in the 10-year Treasury yield and sizes a trade. After the release, it scores the call against what the market actually did and drafts a post-mortem, which I use for a weekly market note and to recalibrate the next trade.

## What's inside

| File | What it does |
|---|---|
| `yield_curve_dashboard.py` | Pulls live Treasury yields from FRED. Charts the yield curve, the 2s10s spread and 2Y vs 10Y, and prints a weekly rates summary in bp. |
| `rates_models.py` | Event model, walk-forward event backtest, recession probability model and curve trade backtest. |
| `rates_pro_forma.py` | 3-year projection of the 3M, 2Y and 10Y yields under Base, Hawkish and Dovish scenarios, compared against market-implied forward rates. |
| `event_history.csv` | Historical releases (date, event, consensus, actual) used to fit coefficients and run the backtest. |

## Methodology

**Event model**
- Each release type (CPI, PCE, NFP, unemployment) has its own coefficient: the bp move in the 2Y and 10Y per unit of surprise (actual minus consensus).
- Coefficients are fit by weighted regression on the last 3 years of releases, the post-COVID rate regime. Recent prints are weighted more heavily (12-month half-life), so the model adapts as the Fed's focus shifts between inflation and jobs.
- My beat / in-line / miss probabilities produce a probability-weighted expected move and a distribution. Position size scales with expected move relative to that uncertainty, and trades below a minimum edge are skipped.
- The 10Y is the primary trade, since it's the global benchmark rate, drives mortgage rates and sets the discount rate for equity valuations. The 2Y and the 2s10s curve are secondary legs.

**Post-mortem**
- Separates *forecast error* (did I get the data wrong?) from *reaction error* (did the market respond differently than the model expected?).
- Includes the S&P 500's move on the day to track the equity discount-rate channel.

**Backtest**
- Walk-forward, so each release is priced only with data available before that date.
- Tests whether the coefficients predict the realized 10Y reaction, plus a systematic surprise-momentum trading rule.

**Pro forma**
- Building-block model: 2Y = expected average Fed funds over the next 2 years plus a premium, and 10Y = expected average short rate over the next 10 years plus term premium.
- Scenarios are set by Fed path, neutral rate and term premium views, calibrated to start at today's yields.
- Term premium is calibrated once and shared across scenarios, and the market prices each scenario's Fed path gradually rather than on day one.
- Uncertainty bands use realized yield volatility. Market-implied forwards (math on today's curve, not the Fed's dot plot) show what is already priced in, and the gap between my Base case and the forwards is the trade signal.
- Recession risk is tracked with the Sahm rule, driven by each scenario's unemployment path.

**Muni / wealth management module**
- Converts each scenario's 10Y into a projected 10Y muni yield using the muni/Treasury ratio, which drifts by scenario as supply and fund flows shift.
- Computes tax-equivalent yield by client bracket (24% up to 40.8% with the net investment income tax) against Treasuries (federal only) and corporates (federal plus state for in-state munis).
- Shows the breakeven tax rate (1 minus the ratio) where munis beat Treasuries after tax, the relative value call a desk would pitch to HNW and UHNW clients.

**Recession signals**
- Leading: logistic regression of recessions 12 months ahead on the 10Y minus 3M spread, the same approach as the NY Fed model.
- Real time: the Sahm rule (3-month average unemployment vs. its prior 12-month low, with 0.50pp as the trigger).

## Running it

The easiest way is Google Colab:
1. Open a new notebook at colab.research.google.com.
2. Paste each `.py` file into its own code cell.
3. Upload `event_history.csv` through the file panel, then run.

Or locally:
```bash
pip install -r requirements.txt
python yield_curve_dashboard.py
python rates_models.py
python rates_pro_forma.py
```

All market data comes from FRED's public CSV endpoints, so no API key is needed.

## Weekly workflow

1. Check the economic calendar for upcoming releases and their consensus.
2. Add each release with my odds to `EVENT_LOG` and run the model to get a trade ticket.
3. After the release, fill in the actual number and rerun for the post-mortem.
4. Log the result in my trade journal and publish a market note.

## Limitations

- Short-term yield moves are noisy, so a single release often moves less than daily noise. The model is built to show when there's no edge as much as when there is.
- 3 years of history gives roughly 36 observations per release, so coefficients carry real estimation error.
- Releases that land on the same day (jobs and unemployment, or PCE and GDP) share the same yield move.
- The curve-based recession model reads steepening as lower risk, even though the curve often un-inverts right as a recession begins.
