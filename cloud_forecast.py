#!/usr/bin/env python3
"""
Aging AI — hourly forecast in the cloud (GitHub Actions), without copper.

Same frozen model (aging_ai_boston_model.npz) and the same live data
as the installation: ISO New England NEMA demand + Open-Meteo Boston weather.
Nothing is sent to any copper. Writes status.json at the site root.

If the Raspberry Pi of the installation has published within the last 3 hours,
this script does nothing: the installation's own record always comes first.

record.json keeps every hour since the weights were saved (22 Sep 2026).
Hours before this site went live are replayed once: the recorded ISO-NE demand
and Boston weather of each past hour, run through the same frozen weights
(as replay_fast in AGING_FINAL.py does). They are marked live = 0.
"""
import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SITE = Path(__file__).resolve().parent            # everything sits at the top of the site repo

# The frozen model is kept next to the code; aging_ai_boston.py reads it from state/.
_MODEL = SITE / "aging_ai_boston_model.npz"
if _MODEL.exists():
    (SITE / "state").mkdir(exist_ok=True)
    if not (SITE / "state" / _MODEL.name).exists():
        shutil.copy2(_MODEL, SITE / "state" / _MODEL.name)

import boston_rpi as R                            # noqa: E402
OUT = SITE / "status.json"
RECORD = SITE / "record.json"
HISTORY_HOURS = 72
PI_PRIORITY_HOURS = 3
TRAINED_UTC = datetime(2026, 9, 22, 11, 38, 58, tzinfo=timezone.utc)   # aging_ai_boston_model.npz written
REPLAY_FROM = TRAINED_UTC.replace(minute=0, second=0) + timedelta(hours=1)   # first whole hour after it


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _weather_since(start):
    """Hourly Boston weather from `start` (a Boston date) to now, from the same
    Open-Meteo endpoint the live forecast uses (it serves up to 92 past days)."""
    import pandas as pd
    B = R.B
    need = (pd.Timestamp.now(tz=B.LOCAL_TZ).normalize() - start).days + 2
    recent = B.fetch_recent_boston_weather(past_days=min(92, need))
    if need <= 92:
        return recent
    older = B.fetch_historical_boston_weather(start.date().isoformat(),
                                              (recent["timestamp_utc"].min() - pd.Timedelta(days=1)).date().isoformat())
    return pd.concat([older, recent]).drop_duplicates("timestamp_utc", keep="last").sort_values("timestamp_utc")


def replay():
    """Every hour from REPLAY_FROM to now, run once through the frozen weights."""
    import pandas as pd
    B = R.B
    net = B.AgingNet.load()
    start = pd.Timestamp(REPLAY_FROM).tz_convert(B.LOCAL_TZ).normalize() - pd.Timedelta(days=8)   # 168 h lag
    days = (pd.Timestamp.now(tz=B.LOCAL_TZ).normalize() - start).days + 1
    print(f"Replaying since {_iso(REPLAY_FROM)}: {days} days of ISO-NE data …")
    rows = B.build_live_feature_rows(R.fetch_nema_isone(days=days, start_date=start), _weather_since(start))
    rows = rows[rows["timestamp_utc"] >= pd.Timestamp(REPLAY_FROM)].reset_index(drop=True)
    if rows.empty:
        return {}
    pred, _, _ = net.predict(rows[B.FEATURES].to_numpy(dtype=float))
    alive = int((~net.dead).sum())
    out = {}
    for t, o, f in zip(rows["timestamp_utc"], rows["demand_now"], pred):
        k = _iso(t.to_pydatetime())
        out[k] = [k, round(float(o), 1), round(float(f), 1), alive, 0]
    print(f"Replayed {len(out)} hours")
    return out


def update_record(st, old_history):
    """Add this hour (live = 1) to record.json. Until the replay has succeeded once,
    try it again each hour; a live hour is never overwritten by its replay."""
    rec = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    hours = {h[0]: h for h in rec.get("hours", [])}
    for h in old_history:                       # hours this site already forecast live
        k = _iso(_t(h["t"]))
        hours.setdefault(k, [k, h["observed"], h["forecast"], h.get("alive", 25), 1])
    k = _iso(_t(st["input_hour_utc"]))
    hours[k] = [k, st["observed_mw"], st["forecast_mw"], st["alive"], 1]
    replayed = bool(rec.get("replayed"))
    if not replayed:
        try:
            for k2, row in replay().items():
                hours.setdefault(k2, row)
            replayed = True
        except Exception as e:
            print("replay failed; trying again next hour:", re.sub(r"://[^/@\s]+@", "://", str(e))[:200])
    RECORD.write_text(json.dumps({
        "trained_utc": _iso(TRAINED_UTC),
        "replayed": replayed,
        "columns": ["input_hour_utc", "observed_mw", "forecast_next_hour_mw", "alive", "live"],
        "hours": [hours[k] for k in sorted(hours)],
    }, separators=(",", ":")))


def main():
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    now = datetime.now(timezone.utc)
    if old.get("publisher") == "pi" and old.get("updated_utc") and \
            now - _t(old["updated_utc"]) < timedelta(hours=PI_PRIORITY_HOURS):
        print("The installation published recently; leaving its record as it is.")
        return

    if not R.B.MODEL_PATH.exists():
        print("No trained model found (aging_ai_boston_model.npz).")
        return

    # The next input hour t+1 is complete only at t+2. Until then there is nothing new to fetch.
    complete = "edges" in old                   # older records lack the per-hour network detail
    if complete and old.get("input_hour_utc") and not old.get("error") and \
            now < _t(old["input_hour_utc"]) + timedelta(hours=2):
        print("The next hour is not complete yet.")
        return

    try:
        try:
            p = R.predict_now()
        except Exception as first:              # ISO-NE or Open-Meteo sometimes time out; try once more
            print("first attempt failed:", str(first)[:160])
            time.sleep(20)
            p = R.predict_now()
    except Exception as e:                      # keep the last good forecast, note the error
        err = re.sub(r"://[^/@\s]+@", "://", str(e))[:200]
        if old.get("error") != err:             # one commit per new error, not one per attempt
            old["error"] = err
            old["updated_utc"] = now.isoformat()
            OUT.write_text(json.dumps(old, indent=1))
        print("forecast failed:", err)
        return

    if complete and old.get("input_hour_utc") == p["ts"].isoformat() and not old.get("error"):
        print("ISO-NE has not posted the next hour yet; trying again in 10 minutes.")
        return

    R.write_status(p, sent={"status": "view only"})
    R.print_result(p)                           # the same table the display shows, kept in this run's log
    st = json.loads(R.STATUS_PATH.read_text())
    if os.environ.get("GITHUB_RUN_ID"):         # where anyone can check this run's log
        rid = os.environ["GITHUB_RUN_ID"]
        st["run"] = {"id": int(rid), "url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                                            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{rid}"}

    hist = {h["t"]: h for h in old.get("history", []) if "t" in h}
    hist[st["input_hour_utc"]] = {"t": st["input_hour_utc"], "observed": st["observed_mw"],
                                  "forecast": st["forecast_mw"], "alive": st["alive"]}
    newest = _t(st["input_hour_utc"])
    st["history"] = [hist[k] for k in sorted(hist)
                     if newest - _t(k) <= timedelta(hours=HISTORY_HOURS - 1)]
    st["copper_ms"] = [0] * R.B.HIDDEN          # no copper attached
    # Pulse time each cell would have received since the site went live (nothing is deposited).
    ms = [int(n["pulse_ms"]) for n in st["neurons"]]
    if "pulse_total_ms" not in old:
        st["pulse_total_ms"], st["pulse_total_since"] = ms, st["input_hour_utc"]
    else:
        new_hour = old.get("input_hour_utc") != st["input_hour_utc"]
        st["pulse_total_ms"] = [a + (b if new_hour else 0) for a, b in zip(old["pulse_total_ms"], ms)]
        st["pulse_total_since"] = old.get("pulse_total_since") or st["input_hour_utc"]
    st["mode"] = "standalone"
    st["publisher"] = "cloud"
    st["has_forecast"] = True
    OUT.write_text(json.dumps(st, indent=1))
    update_record(st, old.get("history", []))
    print(f"{st['input_hour_boston']}: observed {st['observed_mw']:,} MW -> "
          f"forecast {st['forecast_mw']:,} MW, {st['alive']} neurons")


if __name__ == "__main__":
    main()
