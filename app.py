"""Demand Forecast & Inventory Analytics API (M5 / Walmart data) - single file.

Needs demand_model.pkl in the same folder.
Install:  pip install fastapi "uvicorn[standard]" pydantic scikit-learn==1.9.1 numpy==2.5.4 pandas==3.0.6 joblib==1.6.0
Run:      uvicorn app:app --reload          (docs at http://127.0.0.1:8000/docs)
Public:   uvicorn app:app --host 0.0.0.0 --port $PORT     (Render)  /  port 7860 (Hugging Face Docker Space)
Keep the scikit-learn version above, otherwise the .pkl may not load.

Endpoints
  GET  /health  /products  /stores  /model-info
  POST /predict            1. demand forecasting
  GET  /seasonality        2. seasonal demand analysis
  GET  /regional-analysis  3. regional demand analysis
  POST /volatility         4. demand volatility detection
  POST /stockout-risk      5. stockout risk  6. safety stock  7. recommended order quantity
  POST /depletion          8. inventory depletion estimation
  POST /overstock          9. overstock detection
  POST /inventory-report   all of the above for one product & store in a single call
"""
import math
import os
from statistics import NormalDist
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =====================================================================================================
# Core logic (identical to the notebook)
# =====================================================================================================
STORE_REGION = {"CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
                "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
                "WI_1": "WI", "WI_2": "WI", "WI_3": "WI"}
REGION_NAMES = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}
FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag8", "rm4", "rm8", "rm13", "rstd4",
            "month", "woy", "reg_ca", "reg_tx"]
MIN_HISTORY = 13   # weeks of history the model needs
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VOLATILITY_THRESHOLDS = {"low_max": 0.25, "medium_max": 0.45}   # on the coefficient of variation (last 8 weeks)


# ---------------------------------------------------------------- 1. Demand forecasting
def features_from_history(history, week_start, region):
    h = [float(x) for x in history]
    if len(h) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} weeks of history, got {len(h)}")
    ws = pd.Timestamp(week_start)
    return pd.DataFrame([{
        "lag1": h[-1], "lag2": h[-2], "lag3": h[-3], "lag4": h[-4], "lag8": h[-8],
        "rm4": float(np.mean(h[-4:])), "rm8": float(np.mean(h[-8:])), "rm13": float(np.mean(h[-13:])),
        "rstd4": float(np.std(h[-4:], ddof=1)),
        "month": int(ws.month), "woy": int(ws.isocalendar().week),
        "reg_ca": int(region == "CA"), "reg_tx": int(region == "TX"),
    }])[FEATURES]


def forecast_weeks(model, history, next_week_start, region, n_weeks, cap=None):
    """Recursive multi-week forecast: each forecast is fed back as history for the next week."""
    h, out, ws = list(history), [], pd.Timestamp(next_week_start)
    for _ in range(n_weeks):
        yhat = float(max(0.0, model.predict(features_from_history(h, ws, region))[0]))
        if cap is not None:
            yhat = min(yhat, cap)
        out.append(yhat)
        h.append(yhat)
        ws += pd.Timedelta(weeks=1)
    return out


# ---------------------------------------------------------------- 2. Seasonal demand analysis
def seasonality_analysis(month_index, dow_index, next_week_start, n_weeks=8):
    """month_index: 12 values (Jan..Dec), dow_index: 7 values (Mon..Sun); both average 1.0."""
    mi, di = np.asarray(month_index, float), np.asarray(dow_index, float)
    amp = float(mi.max() - mi.min())
    strength = "STRONG" if amp >= 0.35 else ("MODERATE" if amp >= 0.15 else "WEAK")
    months = [{"month": MONTH_NAMES[i], "index": round(float(mi[i]), 2),
               "vs_average_pct": round(float((mi[i] - 1) * 100), 1)} for i in range(12)]
    days = [{"day": DAY_NAMES[i], "index": round(float(di[i]), 2),
             "vs_average_pct": round(float((di[i] - 1) * 100), 1)} for i in range(7)]
    outlook = []
    for k in range(n_weeks):
        ws = pd.Timestamp(next_week_start) + pd.Timedelta(weeks=k)
        m = (ws + pd.Timedelta(days=3)).month
        idx = float(mi[m - 1])
        outlook.append({"week_start": str(ws.date()), "month": MONTH_NAMES[m - 1], "index": round(idx, 2),
                        "outlook": "ABOVE_AVERAGE" if idx >= 1.07 else ("BELOW_AVERAGE" if idx <= 0.93 else "NORMAL")})
    return {"seasonality_strength": strength, "month_swing": round(amp, 2),
            "peak_month": MONTH_NAMES[int(mi.argmax())], "low_month": MONTH_NAMES[int(mi.argmin())],
            "peak_day": DAY_NAMES[int(di.argmax())], "low_day": DAY_NAMES[int(di.argmin())],
            "monthly_index": months, "day_of_week_index": days, "upcoming_weeks": outlook}


# ---------------------------------------------------------------- 3. Regional demand analysis
def _cv(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() > 0 else 0.0


def _trend_pct(h):
    h = np.asarray(h, float)
    prev = h[-26:-13].mean() if len(h) >= 26 else h[:-13].mean()
    return float((h[-13:].mean() / prev - 1) * 100) if prev > 0 else 0.0


def regional_analysis(histories, store_region, next_week_forecast):
    """histories: {store_id: weekly units}; next_week_forecast: {store_id: units}."""
    n = min(len(h) for h in histories.values())
    regions = {}
    for sid, h in histories.items():
        regions.setdefault(store_region[sid], []).append(sid)
    out, total = [], 0.0
    for reg, sids in regions.items():
        series = np.sum([np.asarray(histories[s][-n:], float) for s in sids], axis=0)
        avg = float(series[-52:].mean())
        total += avg
        stores = [{"store_id": s, "avg_weekly_units": round(float(np.mean(histories[s][-52:])), 1),
                   "trend_pct": round(_trend_pct(histories[s]), 1),
                   "volatility_cv": round(_cv(histories[s][-8:]), 2),
                   "next_week_forecast": round(float(next_week_forecast[s]), 1)} for s in sorted(sids)]
        out.append({"region": reg, "region_name": REGION_NAMES[reg], "n_stores": len(sids),
                    "avg_weekly_units": round(avg, 1), "avg_weekly_units_per_store": round(avg / len(sids), 1),
                    "trend_pct": round(_trend_pct(series), 1), "volatility_cv": round(_cv(series[-8:]), 2),
                    "next_week_forecast": round(sum(next_week_forecast[s] for s in sids), 1), "stores": stores})
    for r in out:
        r["share_of_total_pct"] = round(100 * r["avg_weekly_units"] / total, 1) if total > 0 else 0.0
    out.sort(key=lambda r: -r["avg_weekly_units_per_store"])
    best_store = max((s for r in out for s in r["stores"]), key=lambda s: s["avg_weekly_units"])
    return {"regions": out,
            "highest_demand_region_per_store": out[0]["region"],
            "fastest_growing_region": max(out, key=lambda r: r["trend_pct"])["region"],
            "most_volatile_region": max(out, key=lambda r: r["volatility_cv"])["region"],
            "highest_demand_store": best_store["store_id"],
            "note": "avg_weekly_units = last 52 weeks; trend_pct = last 13 weeks vs the 13 weeks before; "
                    "volatility_cv = std/mean of last 8 weeks; regions ranked by demand per store."}


# ---------------------------------------------------------------- 4. Demand volatility detection
def volatility_analysis(history, thresholds=None):
    th = thresholds or VOLATILITY_THRESHOLDS
    h = np.asarray(history, float)
    if len(h) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} weeks of history, got {len(h)}")
    cv8, cv52 = _cv(h[-8:]), _cv(h[-52:])
    level = "LOW" if cv8 < th["low_max"] else ("MEDIUM" if cv8 < th["medium_max"] else "HIGH")
    ratio = cv8 / cv52 if cv52 > 0 else 1.0
    trend = "RISING" if ratio > 1.3 else ("FALLING" if ratio < 0.77 else "STABLE")

    def z_at(i):   # how unusual is week i compared with the 12 weeks before it
        base = h[i - 12:i]
        s = base.std(ddof=1)
        return 0.0 if s < 1e-9 else float((h[i] - base.mean()) / s)

    z_latest = z_at(len(h) - 1)
    spike_weeks = [i for i in range(max(12, len(h) - 13), len(h)) if abs(z_at(i)) > 2]
    flag = "SPIKE" if z_latest > 2 else ("DROP" if z_latest < -2 else "NORMAL")
    alerts = []
    if level == "HIGH":
        alerts.append("Demand is highly variable: hold extra safety stock and review more often.")
    if trend == "RISING":
        alerts.append("Variability over the last 8 weeks is well above this item's 52-week norm.")
    if flag == "SPIKE":
        alerts.append("Last week's sales were unusually high versus the previous 12 weeks.")
    if flag == "DROP":
        alerts.append("Last week's sales were unusually low versus the previous 12 weeks.")
    return {"volatility_level": level, "cv_last_8_weeks": round(cv8, 3), "cv_last_52_weeks": round(cv52, 3),
            "volatility_trend": trend, "latest_week_flag": flag, "latest_week_z_score": round(z_latest, 2),
            "latest_week_units": float(h[-1]), "avg_units_previous_12_weeks": round(float(h[-13:-1].mean()), 1),
            "unusual_weeks_in_last_13": len(spike_weeks), "alerts": alerts,
            "thresholds": {"LOW": f"CV < {th['low_max']}", "MEDIUM": f"CV < {th['medium_max']}",
                           "HIGH": f"CV >= {th['medium_max']}"}}


# ---------------------------------------------------------------- 5/6/7. Stockout risk, safety stock, order quantity
def stockout_analysis(weekly_forecasts, weekly_sigma, available_stock, lead_time_days,
                      service_level=0.95, review_days=7):
    """Explicit inventory rule.

    lead_time_demand    = forecast demand over the lead time (weekly forecasts pro-rated per day)
    safety_stock        = z(service_level) * weekly_sigma * sqrt(lead_time_days / 7)
    reorder_point (ROP) = lead_time_demand + safety_stock
    risk: HIGH   if stock <= lead_time_demand  (expected to run out before the order arrives)
          MEDIUM if stock <= ROP               (inside the safety-stock zone -> reorder now)
          LOW    otherwise
    suggested_order_qty = max(0, ROP + review_days * daily_demand - stock)
    """
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be > 0")
    z = NormalDist().inv_cdf(service_level)
    remaining, ltd = lead_time_days, 0.0
    for w in weekly_forecasts:
        take = min(7, remaining)
        ltd += w * take / 7.0
        remaining -= take
        if remaining <= 0:
            break
    daily = ltd / lead_time_days
    ss = z * weekly_sigma * math.sqrt(lead_time_days / 7.0)
    rop = ltd + ss
    risk = "HIGH" if available_stock <= ltd else ("MEDIUM" if available_stock <= rop else "LOW")
    cover = available_stock / daily if daily > 0 else None
    return {"stockout_risk": risk, "avg_daily_demand": round(daily, 2),
            "lead_time_demand": round(ltd, 1), "safety_stock": round(ss, 1),
            "reorder_point": round(rop, 1),
            "days_of_cover": None if cover is None else round(cover, 1),
            "reorder_now": bool(available_stock <= rop),
            "suggested_order_qty": int(math.ceil(max(0.0, rop + review_days * daily - available_stock)))}


# ---------------------------------------------------------------- 8. Inventory depletion estimation
def daily_profile(weekly_forecasts, dow_index, start_date):
    """Spread each weekly forecast over its 7 days using the day-of-week index (weekly total preserved)."""
    start = pd.Timestamp(start_date).normalize()
    return np.array([weekly_forecasts[i // 7] / 7.0 * dow_index[(start + pd.Timedelta(days=i)).weekday()]
                     for i in range(len(weekly_forecasts) * 7)])


def depletion_analysis(weekly_forecasts, dow_index, available_stock, start_date,
                       lead_time_days=None, reorder_point=None):
    daily = daily_profile(weekly_forecasts, dow_index, start_date)
    cum = np.concatenate([[0.0], np.cumsum(daily)])        # cum[k] = units demanded in the first k days
    n = len(daily)
    start = pd.Timestamp(start_date).normalize()

    def days_until(level):                                  # days until stock falls to `level`
        need = available_stock - level
        if need <= 0:
            return 0.0
        if cum[-1] < need:
            return None
        k = int(np.argmax(cum >= need))
        step = cum[k] - cum[k - 1]
        return float((k - 1) + ((need - cum[k - 1]) / step if step > 0 else 0.0))

    def date_of(d):
        return None if d is None else str((start + pd.Timedelta(days=int(math.floor(d)))).date())

    d0 = days_until(0.0)
    res = {"start_date": str(start.date()), "horizon_days": n,
           "runs_out_within_horizon": bool(d0 is not None),
           "days_until_stockout": None if d0 is None else round(d0, 1), "stockout_date": date_of(d0)}
    if reorder_point is not None:
        dr = days_until(reorder_point)
        res.update({"days_until_reorder_point": None if dr is None else round(dr, 1),
                    "reorder_by_date": date_of(dr)})
    if lead_time_days is not None:
        sold = float(np.interp(lead_time_days, np.arange(n + 1), cum))
        res.update({"stock_when_order_arrives": round(max(0.0, available_stock - sold), 1),
                    "expected_shortfall_units": round(max(0.0, sold - available_stock), 1),
                    "runs_out_before_delivery": bool(d0 is not None and d0 < lead_time_days)})
    res["daily_forecast_next_7_days"] = [
        {"date": str((start + pd.Timedelta(days=i)).date()), "day": DAY_NAMES[(start + pd.Timedelta(days=i)).weekday()],
         "expected_units": round(float(daily[i]), 1)} for i in range(min(7, n))]
    res["weekly_projection"] = [
        {"week_start": str((start + pd.Timedelta(days=7 * w)).date()),
         "expected_demand": round(float(cum[7 * (w + 1)] - cum[7 * w]), 1),
         "projected_end_stock": round(max(0.0, available_stock - float(cum[7 * (w + 1)])), 1)}
        for w in range(n // 7)]
    return res


# ---------------------------------------------------------------- 9. Overstock detection
def overstock_analysis(weekly_demand, reorder_point, available_stock, extra_weeks=4):
    """max_stock = reorder_point + extra_weeks x weekly_demand.  Overstock if stock > max_stock.
    severity by stock / max_stock:  <=1.5 MILD, <=2.0 MODERATE, >2.0 SEVERE."""
    max_stock = reorder_point + extra_weeks * weekly_demand
    ratio = available_stock / max_stock if max_stock > 0 else 0.0
    if available_stock <= max_stock:
        severity = "NONE"
    else:
        severity = "MILD" if ratio <= 1.5 else ("MODERATE" if ratio <= 2.0 else "SEVERE")
    excess = max(0.0, available_stock - max_stock)
    weeks_cover = available_stock / weekly_demand if weekly_demand > 0 else None
    status = "OVERSTOCK" if severity != "NONE" else ("REORDER" if available_stock <= reorder_point else "HEALTHY")
    advice = {"OVERSTOCK": "Do not reorder. Consider promotions or transfers to other stores.",
              "REORDER": "Stock is at or below the reorder point: place an order.",
              "HEALTHY": "Stock is between the reorder point and the maximum level."}[status]
    return {"stock_status": status, "overstock": bool(severity != "NONE"), "overstock_severity": severity,
            "max_stock_level": round(float(max_stock), 1), "excess_units": round(float(excess), 1),
            "weeks_of_cover": None if weeks_cover is None else round(float(weeks_cover), 1),
            "weeks_to_sell_excess": round(float(excess / weekly_demand), 1) if weekly_demand > 0 else None,
            "avg_weekly_demand": round(float(weekly_demand), 1), "advice": advice}

# =====================================================================================================
# Service layer
# =====================================================================================================
BUNDLE = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demand_model.pkl"))


class NotFound(Exception):
    pass


class BadInput(Exception):
    pass


def _week_start(ws):
    if ws is None:
        return BUNDLE["next_week_start"]
    try:
        t = pd.Timestamp(ws)
    except Exception:
        raise BadInput("week_start must be a date like 2016-05-23")
    if t.weekday() != 0:
        raise BadInput("week_start must be a Monday (weeks run Monday-Sunday)")
    return str(t.date())


def _resolve(product_id, store_id, history):
    """-> (weekly history, region, weekly sigma, series key)."""
    b = BUNDLE
    if store_id not in b["store_region"]:
        raise BadInput(f"store_id must be one of {sorted(b['store_region'])}")
    region = b["store_region"][store_id]
    key = f"{product_id}|{store_id}"
    if history:
        if len(history) < MIN_HISTORY:
            raise BadInput(f"recent_weekly_sales needs at least {MIN_HISTORY} weekly values")
        if any(x < 0 for x in history):
            raise BadInput("recent_weekly_sales cannot contain negative values")
        return list(history), region, float(max(np.std(history[-8:], ddof=1), 1.0)), key
    if key not in b["history"]:
        raise NotFound(f"unknown product/store '{key}'. Use GET /products and GET /stores, "
                       "or send recent_weekly_sales.")
    hist = b["history"][key]
    sigma = b["sigma"].get(key) or b["global_rel_rmse"] * float(np.mean(hist))
    return hist, region, float(max(sigma, 1.0)), key


def _dow(key):
    return BUNDLE["seasonality"].get(key, {}).get("dow", [1.0] * 7)


def _fc(hist, region, ws, n):
    return forecast_weeks(BUNDLE["model"], hist, ws, region, n, cap=BUNDLE["cap"])


def _r1(xs):
    return [round(x, 1) for x in xs]


def do_model_info():
    b = BUNDLE
    return {"model": b["model_name"], "features": b["features"], "n_products": len(b["products"]),
            "stores": b["store_region"], "data_through_week": b["last_history_week"],
            "default_forecast_week": b["next_week_start"], "accuracy": b["accuracy"],
            "comparison": b["comparison"]}


def do_products():
    return {"products": BUNDLE["products"]}


def do_stores():
    return {"stores": [{"store_id": s, "region": r, "region_name": REGION_NAMES[r]}
                       for s, r in BUNDLE["store_region"].items()]}


def do_predict(product_id, store_id, recent_weekly_sales=None, week_start=None, horizon_weeks=1):
    hist, region, _, _ = _resolve(product_id, store_id, recent_weekly_sales)
    ws = _week_start(week_start)
    fc = _fc(hist, region, ws, horizon_weeks)
    return {"product_id": product_id, "store_id": store_id, "region": region, "first_forecast_week": ws,
            "weekly_forecast_units": _r1(fc), "model": BUNDLE["model_name"]}


def do_seasonality(product_id, store_id, n_weeks=8):
    _, region, _, key = _resolve(product_id, store_id, None)
    s = BUNDLE["seasonality"][key]
    out = seasonality_analysis(s["month"], s["dow"], BUNDLE["next_week_start"], n_weeks)
    out.update({"product_id": product_id, "store_id": store_id, "region": region,
                "basis": "Month index = 90% chain-wide pattern + 10% product-region pattern (store-level monthly "
                         "patterns were too noisy to beat 'no seasonality' in a backtest). Day-of-week index is "
                         "estimated per product & store."})
    return out


def do_regional(product_id):
    if product_id not in BUNDLE["products"]:
        raise NotFound(f"unknown product '{product_id}'. Use GET /products.")
    b = BUNDLE
    hs = {s: b["history"][f"{product_id}|{s}"] for s in b["store_region"]}
    nf = {s: _fc(hs[s], b["store_region"][s], b["next_week_start"], 1)[0] for s in hs}
    out = regional_analysis(hs, b["store_region"], nf)
    out.update({"product_id": product_id, "forecast_week": b["next_week_start"]})
    return out


def do_volatility(product_id, store_id, recent_weekly_sales=None):
    hist, region, _, _ = _resolve(product_id, store_id, recent_weekly_sales)
    out = volatility_analysis(hist, BUNDLE["volatility_thresholds"])
    out.update({"product_id": product_id, "store_id": store_id, "region": region})
    return out


def do_stockout(product_id, store_id, available_stock, lead_time_days, recent_weekly_sales=None,
                week_start=None, service_level=0.95, review_days=7):
    hist, region, sigma, _ = _resolve(product_id, store_id, recent_weekly_sales)
    ws = _week_start(week_start)
    fc = _fc(hist, region, ws, int(math.ceil(lead_time_days / 7.0)))
    res = stockout_analysis(fc, sigma, available_stock, lead_time_days, service_level, review_days)
    res.update({"product_id": product_id, "store_id": store_id, "region": region,
                "available_stock": available_stock, "lead_time_days": lead_time_days,
                "service_level": service_level, "weekly_forecast_units": _r1(fc)})
    return res


def do_depletion(product_id, store_id, available_stock, lead_time_days=None, horizon_days=56,
                 recent_weekly_sales=None, week_start=None, service_level=0.95):
    hist, region, sigma, key = _resolve(product_id, store_id, recent_weekly_sales)
    ws = _week_start(week_start)
    horizon = max(horizon_days, lead_time_days or 0)
    fc = _fc(hist, region, ws, int(math.ceil(horizon / 7.0)))
    rop = None
    if lead_time_days:
        rop = stockout_analysis(fc, sigma, available_stock, lead_time_days, service_level)["reorder_point"]
    res = depletion_analysis(fc, _dow(key), available_stock, ws, lead_time_days, rop)
    res.update({"product_id": product_id, "store_id": store_id, "region": region,
                "available_stock": available_stock, "lead_time_days": lead_time_days,
                "reorder_point": rop})
    return res


def do_overstock(product_id, store_id, available_stock, lead_time_days=7, extra_weeks=4,
                 recent_weekly_sales=None, week_start=None, service_level=0.95):
    hist, region, sigma, _ = _resolve(product_id, store_id, recent_weekly_sales)
    ws = _week_start(week_start)
    n = max(4, int(math.ceil(lead_time_days / 7.0)))
    fc = _fc(hist, region, ws, n)
    st = stockout_analysis(fc, sigma, available_stock, lead_time_days, service_level)
    res = overstock_analysis(float(np.mean(fc[:4])), st["reorder_point"], available_stock, extra_weeks)
    res.update({"product_id": product_id, "store_id": store_id, "region": region,
                "available_stock": available_stock, "reorder_point": st["reorder_point"],
                "extra_weeks": extra_weeks, "lead_time_days": lead_time_days})
    return res


def do_report(product_id, store_id, available_stock, lead_time_days, recent_weekly_sales=None,
              week_start=None, service_level=0.95, review_days=7, extra_weeks=4):
    common = dict(product_id=product_id, store_id=store_id, recent_weekly_sales=recent_weekly_sales,
                  week_start=week_start)
    so = do_stockout(available_stock=available_stock, lead_time_days=lead_time_days,
                     service_level=service_level, review_days=review_days, **common)
    dep = do_depletion(available_stock=available_stock, lead_time_days=lead_time_days,
                       service_level=service_level, **common)
    ov = do_overstock(available_stock=available_stock, lead_time_days=lead_time_days,
                      extra_weeks=extra_weeks, service_level=service_level, **common)
    vol = do_volatility(product_id, store_id, recent_weekly_sales)
    fc = do_predict(horizon_weeks=4, **common)
    seas = do_seasonality(product_id, store_id, 4) if f"{product_id}|{store_id}" in BUNDLE["seasonality"] else None

    reasons = []
    if so["stockout_risk"] == "HIGH":
        reasons.append("Stock is below expected demand over the lead time: a stock-out is likely before delivery.")
    elif so["stockout_risk"] == "MEDIUM":
        reasons.append("Stock is inside the safety-stock zone: reorder now.")
    if dep.get("runs_out_before_delivery"):
        reasons.append(f"Stock is estimated to run out on {dep['stockout_date']}, before the order arrives.")
    if ov["overstock"]:
        reasons.append(f"Overstock ({ov['overstock_severity']}): {ov['excess_units']} units above the maximum level.")
    if vol["volatility_level"] == "HIGH":
        reasons.append("Demand is highly volatile.")
    if vol["latest_week_flag"] != "NORMAL":
        reasons.append(f"Last week's sales were an unusual {vol['latest_week_flag'].lower()}.")
    urgent = so["stockout_risk"] == "HIGH" or bool(dep.get("runs_out_before_delivery"))
    return {"product_id": product_id, "store_id": store_id, "region": so["region"],
            "priority": "URGENT" if urgent else ("ATTENTION" if reasons else "OK"),
            "needs_attention": bool(reasons), "reasons": reasons,
            "forecast": fc, "stockout": so, "depletion": dep, "overstock": ov, "volatility": vol,
            "seasonality": seas}


# =====================================================================================================
# HTTP layer
# =====================================================================================================
app = FastAPI(title="Demand Forecast & Inventory Analytics API", version="3.0.0",
              description="Weekly demand forecasts per product/store plus seasonal, regional, volatility, "
                          "stockout-risk, safety-stock, order-quantity, depletion and overstock analysis from "
                          "user-supplied available stock and lead time.")


class Base(BaseModel):
    product_id: str = Field(..., examples=["FOODS_1_018"])
    store_id: str = Field(..., examples=["CA_1"])
    recent_weekly_sales: Optional[List[float]] = Field(
        None, description="Optional last >=13 weekly unit totals, oldest first. Omit to use stored history.")


class PredictRequest(Base):
    week_start: Optional[str] = Field(None, description="Monday of the first forecast week, YYYY-MM-DD")
    horizon_weeks: int = Field(1, ge=1, le=12)


class VolatilityRequest(Base):
    pass


class StockoutRequest(Base):
    available_stock: float = Field(..., ge=0, description="Units on hand")
    lead_time_days: float = Field(..., gt=0, le=120, description="Days until a new order arrives")
    service_level: float = Field(0.95, gt=0.5, lt=0.999)
    review_days: int = Field(7, ge=0, le=60, description="Extra days of cover to order for")
    week_start: Optional[str] = None


class DepletionRequest(Base):
    available_stock: float = Field(..., ge=0)
    lead_time_days: Optional[float] = Field(None, gt=0, le=120, description="Optional: adds reorder date and shortfall")
    horizon_days: int = Field(56, ge=7, le=120, description="How far ahead to simulate")
    service_level: float = Field(0.95, gt=0.5, lt=0.999)
    week_start: Optional[str] = None


class OverstockRequest(Base):
    available_stock: float = Field(..., ge=0)
    lead_time_days: float = Field(7, gt=0, le=120)
    extra_weeks: float = Field(4, ge=0, le=26, description="Weeks of demand allowed above the reorder point")
    service_level: float = Field(0.95, gt=0.5, lt=0.999)
    week_start: Optional[str] = None


class ReportRequest(Base):
    available_stock: float = Field(..., ge=0)
    lead_time_days: float = Field(..., gt=0, le=120)
    service_level: float = Field(0.95, gt=0.5, lt=0.999)
    review_days: int = Field(7, ge=0, le=60)
    extra_weeks: float = Field(4, ge=0, le=26)
    week_start: Optional[str] = None


def _run(fn, **kw):
    try:
        return fn(**kw)
    except NotFound as e:
        raise HTTPException(404, str(e))
    except (BadInput, ValueError) as e:
        raise HTTPException(422, str(e))


@app.get("/")
def root():
    return {"message": "Demand forecast API is running. See /docs",
            "endpoints": ["/health", "/products", "/stores", "/model-info", "/predict", "/seasonality",
                          "/regional-analysis", "/volatility", "/stockout-risk", "/depletion", "/overstock",
                          "/inventory-report"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/products")
def products():
    return do_products()


@app.get("/stores")
def stores():
    return do_stores()


@app.get("/model-info")
def model_info():
    return do_model_info()


@app.post("/predict")
def predict(req: PredictRequest):
    return _run(do_predict, **req.model_dump())


@app.get("/seasonality")
def seasonality(product_id: str, store_id: str, n_weeks: int = 8):
    return _run(do_seasonality, product_id=product_id, store_id=store_id, n_weeks=n_weeks)


@app.get("/regional-analysis")
def regional_analysis_endpoint(product_id: str):
    return _run(do_regional, product_id=product_id)


@app.post("/volatility")
def volatility(req: VolatilityRequest):
    return _run(do_volatility, **req.model_dump())


@app.post("/stockout-risk")
def stockout_risk(req: StockoutRequest):
    return _run(do_stockout, **req.model_dump())


@app.post("/depletion")
def depletion(req: DepletionRequest):
    return _run(do_depletion, **req.model_dump())


@app.post("/overstock")
def overstock(req: OverstockRequest):
    return _run(do_overstock, **req.model_dump())


@app.post("/inventory-report")
def inventory_report(req: ReportRequest):
    return _run(do_report, **req.model_dump())
