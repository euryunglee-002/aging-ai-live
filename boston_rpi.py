#!/usr/bin/env python3
"""
Aging AI — exhibition runner (Raspberry Pi or laptop)

Uses the model, features and training code in aging_ai_boston.py unchanged,
and adds three things around it:

  1. Live data   NEMA (zone 4008) hourly demand from ISO New England Web Services
                 (ISO Express account). Falls back to EIA-930 subregion data if no
                 ISO-NE credentials are set.
  2. Output      the 25 per-neuron pulse lengths are sent to the Arduino Mega over
                 serial, using the JSON protocol of aging_ai_boston.py's ArduinoBridge.
  3. Status      every cycle writes state/status.json, which web_server.py serves
                 to the live website.

Setup (once):
    sudo apt install python3-numpy python3-pandas python3-requests python3-serial
    cp iso_ne.env.example iso_ne.env      # then fill in the ISO Express account

Usage (in the folder that holds aging_ai_boston.py):
    python3 boston_rpi.py train                                 # train once, freeze -> state/aging_ai_boston_model.npz
    python3 boston_rpi.py predict                               # one forecast, nothing sent to the copper
    python3 boston_rpi.py predict --serial-port /dev/ttyUSB0    # one forecast + pulses
    python3 boston_rpi.py run --serial-port /dev/ttyUSB0        # exhibition: one cycle per new hour of data
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

# Credentials: read from iso_ne.env next to this file (exported variables take priority).
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iso_ne.env")
if os.path.exists(_ENV_FILE):
    for _ln in open(_ENV_FILE, encoding="utf-8"):
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            if _v.strip():
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import aging_ai_boston as B  # noqa: E402  (must load after the credentials)

TRAIN_START = "2023-01-01T00"
LIVE_DAYS = 10                  # enough for the 1-week (168 h) lag
STALE_HOURS = 48                # never spend copper on data older than this
STATUS_PATH = B.STATE_DIR / "status.json"
LAST_SENT = B.STATE_DIR / "last_sent_hour.txt"
LAST_LOGGED = B.STATE_DIR / "last_logged_hour.txt"
DEAD_FLAG = B.STATE_DIR / "aging_dead.flag"

# ------------------------------------------------------------------ ISO New England
ISO_USER = (os.environ.get("ISO_NE_USER") or "").strip()
ISO_PASS = (os.environ.get("ISO_NE_PASSWORD") or "").strip()
ISO_BASE = os.environ.get("ISO_NE_BASE", "https://webservices.iso-ne.com/api/v1.1")
NEMA_LOC = os.environ.get("NEMA_LOCATION_ID", "4008")          # NEMA load zone id

# ------------------------------------------------------------------ EIA (fallback)
EIA_KEY = (os.environ.get("EIA_API_KEY") or "").strip()
EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
NEMA_SUBBA = os.environ.get("NEMA_SUBBA", "4008")


def _find_records(obj):
    """Find the list of records carrying BeginDate and Load in an ISO-NE JSON reply."""
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "BeginDate" in obj[0]:
            return obj
        for x in obj:
            r = _find_records(x)
            if r:
                return r
    elif isinstance(obj, dict):
        if "BeginDate" in obj and "Load" in obj:
            return [obj]
        for v in obj.values():
            r = _find_records(v)
            if r:
                return r
    return []


def fetch_isone_day(day):
    """One day of ISO-NE five-minute estimated zonal load, NEMA zone only.
    Same feed the frozen model was trained on (aging_ai_boston run of 22 Sep 2026)."""
    url = f"{ISO_BASE}/fiveminuteestimatedzonalload/day/{day:%Y%m%d}.json"
    r = requests.get(url, auth=(ISO_USER, ISO_PASS), headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    try:
        recs = r.json()["isone_web_services"]["five_min_estimated_zonal_loads"]["five_min_estimated_zonal_load"]
    except (KeyError, TypeError, ValueError):
        return []
    if isinstance(recs, dict):
        recs = [recs]
    out = []
    for it in recs:
        try:
            if int(it.get("load_zone_id", -1)) != int(NEMA_LOC):
                continue
        except (TypeError, ValueError):
            continue
        ts, mw = it.get("interval_begin_date"), it.get("estimated_load_mw")
        if ts is not None and mw is not None:
            out.append((ts, mw))
    return out


def fetch_nema_isone(days, start_date=None, stop_after_fail=7):
    """Walk back day by day from today (Boston date); stop where the API stops serving history."""
    today = pd.Timestamp.now(tz=B.LOCAL_TZ).normalize()
    pts, fails = [], 0
    for k in range(days):
        d = today - pd.Timedelta(days=k)
        if start_date is not None and d < start_date:
            break
        try:
            got = fetch_isone_day(d)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError("ISO-NE rejected the login — check ISO_NE_USER / ISO_NE_PASSWORD")
            got = []
        except Exception:
            got = []
        if got:
            pts += got
            fails = 0
        else:
            fails += 1
            if k > 2 and fails >= stop_after_fail:
                print(f"ISO-NE serves no data before {d:%Y-%m-%d}; using what was received")
                break
        if days > 30:
            if k and k % 30 == 0:
                print(f"  ISO-NE day {k}/{days} … {len(pts):,} hours")
            time.sleep(0.15)
    if not pts:
        raise RuntimeError("ISO-NE returned no NEMA data")
    df = pd.DataFrame(pts, columns=["t", "mw"])
    five = pd.DataFrame({"t": pd.to_datetime(df["t"], utc=True),
                         "demand_mw": pd.to_numeric(df["mw"], errors="coerce")})
    five = five[five["demand_mw"] > 0].drop_duplicates("t")
    five["timestamp_utc"] = five["t"].dt.floor("h")
    g = five.groupby("timestamp_utc")["demand_mw"]
    out = g.mean().to_frame()
    out = out[g.count() >= 11].reset_index().sort_values("timestamp_utc")   # complete hours only
    # Fill missing hours with NaN so the 1 h / 24 h / 168 h lags follow real time, not row position.
    return out.set_index("timestamp_utc").asfreq("h").reset_index()


def fetch_nema_eia(start=None, hours=None):
    if not EIA_KEY:
        raise RuntimeError("EIA_API_KEY is not set")
    rows, offset = [], 0
    while True:
        length = hours or 5000
        p = [("api_key", EIA_KEY), ("frequency", "hourly"), ("data[0]", "value"),
             ("facets[subba][]", NEMA_SUBBA), ("sort[0][column]", "period"),
             ("sort[0][direction]", "desc"), ("offset", offset), ("length", length)]
        if start:
            p.append(("start", start))
        r = requests.get(EIA_URL, params=p, timeout=60)
        r.raise_for_status()
        chunk = r.json()["response"]["data"]
        rows += chunk
        if hours or len(chunk) < length:
            break
        offset += length
    if not rows:
        raise RuntimeError("EIA returned no NEMA data")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"timestamp_utc": pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True),
                        "demand_mw": pd.to_numeric(df["value"], errors="coerce")})
    out = out[out["demand_mw"] > 0].drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
    return out.set_index("timestamp_utc").asfreq("h").reset_index()


def source():
    if ISO_USER and ISO_PASS:
        return "ISO-NE"
    if EIA_KEY:
        return "EIA"
    raise RuntimeError("Set ISO_NE_USER / ISO_NE_PASSWORD in iso_ne.env (or EIA_API_KEY)")


def live_load():
    return fetch_nema_isone(days=LIVE_DAYS) if source() == "ISO-NE" else fetch_nema_eia(hours=24 * 15)


def get_load(args):
    """Replaces aging_ai_boston.get_load so `train` pulls history from the live source."""
    if getattr(args, "load_csv", None):
        return B.load_nema_csv(args.load_csv)
    print("Training data source:", source())
    if source() == "ISO-NE":
        start = pd.Timestamp(TRAIN_START[:10], tz=B.LOCAL_TZ)
        days = (pd.Timestamp.now(tz=B.LOCAL_TZ) - start).days + 1
        load = fetch_nema_isone(days=days, start_date=start)
    else:
        load = fetch_nema_eia(start=TRAIN_START)
    # Historical weather (Open-Meteo archive) lags a few days: train only up to a week ago.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    return load[load["timestamp_utc"] < cutoff].reset_index(drop=True)


B.get_load = get_load


# ------------------------------------------------------------------ forecast
def predict_now():
    net = B.AgingNet.load()
    live = B.build_live_feature_rows(live_load(), B.fetch_recent_boston_weather(past_days=16))
    if live.empty:
        raise RuntimeError("no hour where both demand and weather exist")
    row = live.iloc[-1]
    ts = row["timestamp_utc"]
    X = row[B.FEATURES].to_numpy(dtype=float)[None, :]
    pred, act, contrib = net.predict(X)
    share, ms = B.contribution_to_commands(contrib[0], net.dead)
    return dict(src=source(), net=net, ts=ts, age_h=(B.utc_now() - ts).total_seconds() / 3600,
                observed=float(row["demand_now"]), pred=float(pred[0]),
                act=act[0], contrib=contrib[0], share=share, ms=ms)


def print_result(p):
    net = p["net"]
    print("\n" + "=" * 70)
    print("AGING AI — NEMA / BOSTON")
    print(f"Data source         : {p['src']}")
    print(f"Input hour (UTC)    : {p['ts']}   ({p['age_h']:.1f} h old)")
    print(f"Input hour (Boston) : {p['ts'].tz_convert(B.LOCAL_TZ)}")
    print(f"NEMA demand now     : {p['observed']:,.1f} MW")
    print(f"Next-hour forecast  : {p['pred']:,.1f} MW")
    print(f"Living neurons      : {(~net.dead).sum()} / {B.HIDDEN}")
    print(f"Pulses this hour    : {int(p['ms'].sum())} ms")
    print("=" * 70)
    for i in np.argsort(-p["contrib"]):
        st = "DEAD" if net.dead[i] else "alive"
        print(f"N{i:02d} | {st:5s} | contribution={p['contrib'][i]:8.4f} | "
              f"share={p['share'][i]:6.2%} | pulse={p['ms'][i]:4d} ms")


def write_status(p=None, sent=None, error=None):
    """Snapshot for the website (state/status.json), written atomically."""
    B.ensure_dirs()
    st = {"updated_utc": B.utc_now().isoformat(), "error": error}
    if p is not None:
        ts, net = p["ts"], p["net"]
        st.update({
            "source": p["src"],
            "input_hour_utc": ts.isoformat(),
            "input_hour_boston": ts.tz_convert(B.LOCAL_TZ).isoformat(),
            "forecast_hour_boston": (ts + pd.Timedelta(hours=1)).tz_convert(B.LOCAL_TZ).isoformat(),
            "age_hours": round(p["age_h"], 2),
            "observed_mw": round(p["observed"], 1),
            "forecast_mw": round(p["pred"], 1),
            "alive": int((~net.dead).sum()),
            "dead": [int(i) for i in np.where(net.dead)[0]],
            "neurons": [{"id": i, "activation": round(float(p["act"][i]), 4),
                         "contribution": round(float(p["contrib"][i]), 4),
                         "share": round(float(p["share"][i]), 4), "pulse_ms": int(p["ms"][i])}
                        for i in range(B.HIDDEN)],
            "sent": sent or {"status": "not sent"},
        })
    elif STATUS_PATH.exists():                      # keep the last good forecast, add the error
        try:
            old = json.loads(STATUS_PATH.read_text())
            old.update(st)
            st = old
        except Exception:
            pass
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATUS_PATH)


# ------------------------------------------------------------------ Arduino Mega
def send_json(port, p):
    """aging_ai_boston.ArduinoBridge, JSON over serial at 115200 baud, one cell at a time."""
    net = p["net"]
    try:
        bridge = B.ArduinoBridge(port)
    except Exception as e:
        return "ERR", f"cannot open port: {e}"
    try:
        try:   # before plating: if a cell already conducts, drop its neuron and redistribute
            newly = B.mark_dead(net, bridge.scan_dead())
            if newly:
                print("Conductive cell detected, neuron(s) removed:", newly)
                p["share"], p["ms"] = B.contribution_to_commands(p["contrib"], net.dead)
        except Exception as e:
            print("  (scan skipped:", e, ")")
        delivered = [0] * B.HIDDEN
        for i, m in enumerate(p["ms"]):
            if m > 0 and not net.dead[i]:
                delivered[i] = int(bridge.pulse(i, int(m)).get("delivered_ms", m))
        try:   # after plating: check again
            newly = B.mark_dead(net, bridge.scan_dead())
            if newly:
                print("Conductive after plating, neuron(s) removed:", newly)
        except Exception:
            pass
        return "ACK", delivered
    except Exception as e:
        return "ERR", str(e)
    finally:
        bridge.close()


def send_run(port, ms, baud=9600):
    """Legacy firmware protocol: 'RUN:ms,...' (25 values) -> 'ACK:...' / 'DEAD...' / 'ERR...'."""
    import serial
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(3)                          # the Mega resets when the port opens
    ser.reset_input_buffer()
    ser.write(("RUN:" + ",".join(str(int(x)) for x in ms) + "\n").encode())
    ser.flush()
    deadline = time.time() + sum(ms) / 1000 + 15
    try:
        while time.time() < deadline:
            ln = ser.readline().decode(errors="ignore").strip()
            if ln.startswith("ACK:") and len(ln[4:].split(",")) == B.HIDDEN:
                return "ACK", [int(x) for x in ln[4:].split(",")]
            if ln.startswith("DEAD"):
                return "DEAD", ln
            if ln.startswith("ERR"):
                return "ERR", ln
        return None, "no reply"
    finally:
        ser.close()


# ------------------------------------------------------------------ one cycle
def cycle(port=None, protocol="json", quiet_repeat=False):
    """One forecast (+ pulses if a port is given). Returns the input hour.
    quiet_repeat: in exhibition mode, print only when a new hour arrives."""
    if DEAD_FLAG.exists():
        print("TERMINATED — no more forecasts:", DEAD_FLAG.read_text())
        return None
    p = predict_now()
    ts = p["ts"]
    new_hour = not (LAST_LOGGED.exists() and LAST_LOGGED.read_text().strip() == ts.isoformat())
    if new_hour or not quiet_repeat:
        print_result(p)
    if new_hour:                                    # log each input hour once, however often we poll
        B.append_csv(B.PREDICTION_LOG, {
            "processed_utc": B.utc_now().isoformat(), "input_timestamp_utc": ts.isoformat(),
            "observed_demand_mw": p["observed"], "predicted_next_hour_mw": p["pred"],
            "alive_neurons": int((~p["net"].dead).sum()), "pulse_total_ms": int(p["ms"].sum())})
        LAST_LOGGED.write_text(ts.isoformat())

    if not port:
        sent = {"status": "view only"}
    elif p["age_h"] > STALE_HOURS:
        sent = {"status": "stale", "detail": f"data is {p['age_h']:.0f} h old; nothing sent"}
    elif LAST_SENT.exists() and LAST_SENT.read_text().strip() == ts.isoformat():
        prev = json.loads(STATUS_PATH.read_text()).get("sent", {}) if STATUS_PATH.exists() else {}
        sent = prev if prev.get("status") == "delivered" else {"status": "already sent this hour"}
    else:
        kind, info = send_json(port, p) if protocol == "json" else send_run(port, p["ms"])
        if kind == "ACK":
            LAST_SENT.write_text(ts.isoformat())
            for i in range(B.HIDDEN):
                if p["ms"][i] > 0:
                    B.append_csv(B.PULSE_LOG, {
                        "utc": B.utc_now().isoformat(), "input_timestamp_utc": ts.isoformat(),
                        "cell": i, "requested_ms": int(p["ms"][i]), "delivered_ms": int(info[i]),
                        "contribution": float(p["contrib"][i]), "share": float(p["share"][i])})
            sent = {"status": "delivered", "utc": B.utc_now().isoformat(), "total_ms": int(sum(info))}
            print(f"Delivered: {sum(info)} ms in total")
        elif kind == "DEAD":
            LAST_SENT.write_text(ts.isoformat())
            DEAD_FLAG.write_text(f"{B.utc_now().isoformat()} {info}")
            sent = {"status": "terminated", "detail": str(info)}
            print("TERMINATED — the Mega reported a short:", info)
        else:
            sent = {"status": "error", "detail": str(info)}
            print("Mega did not confirm:", info, "— will retry next cycle")
    write_status(p, sent)
    return ts


def run(port, poll, protocol="json"):
    print(f"Exhibition mode: checking every {poll} s, pulsing once per new hour (protocol: {protocol})")
    while True:
        try:
            cycle(port, protocol, quiet_repeat=True)
        except Exception as e:
            print(f"[{B.utc_now().isoformat()}] cycle error: {e}")
            write_status(error=str(e))
        time.sleep(poll)


def main():
    B.ensure_dirs()
    ap = argparse.ArgumentParser(description="Aging AI exhibition runner")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train once and freeze")
    t.add_argument("--load-csv")
    for name in ("predict", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--serial-port", required=(name == "run"))
        sp.add_argument("--protocol", choices=["json", "run"], default="json")
        if name == "run":
            sp.add_argument("--poll-seconds", type=int, default=B.POLL_SECONDS)
    args = ap.parse_args()
    if args.command == "train":
        B.train_model(args)
    elif args.command == "predict":
        cycle(args.serial_port, args.protocol)
    else:
        run(args.serial_port, args.poll_seconds, args.protocol)


if __name__ == "__main__":
    main()
