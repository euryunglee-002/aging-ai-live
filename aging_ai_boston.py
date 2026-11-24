#!/usr/bin/env python3
"""
AGING AI — Boston / Northeast Massachusetts (NEMA)
===================================================

Goal
----
Predict NEXT-HOUR electricity demand for ISO New England's
Northeast Massachusetts and Boston (NEMA) load zone.

The neural network has EXACTLY 25 hidden neurons. Each hidden neuron maps
1-to-1 to one physical CuSO4/copper cell.

Training:
    Historical NEMA electricity + Boston weather + calendar features
        -> train weights/biases
        -> save/freeze model

Exhibition / inference:
    Latest matched electricity + weather
        -> frozen NN
        -> next-hour NEMA demand prediction
        -> 25 hidden activations/contributions
        -> Arduino receives relative pulse commands
        -> Arduino performs current-limited electrodeposition
        -> Arduino can report cells that have become conductive
        -> conductive cell => corresponding hidden neuron is permanently masked

IMPORTANT HARDWARE NOTE
-----------------------
This Python program NEVER directly drives the copper electrodes.
It only sends abstract per-neuron pulse-duration commands to an Arduino.

The Arduino/power electronics must enforce the experimentally established
safe voltage/current limits, current limiting, isolation, and maximum pulse
duration for your exact CuSO4 cell geometry. Do not connect electrode cells
directly to a computer/Arduino GPIO.

Data note
---------
ISO-NE makes NEMA/load-zone reports available, but unattended access may
require an ISO Express / ISO Data Feed account. Therefore this code supports:

1. CSV mode (works now): export/download NEMA data to data/nema_load.csv
2. ISO-NE API mode: fill in the authenticated endpoint/credentials once
   your ISO Express data-feed access is configured.

Weather is fetched automatically from Open-Meteo for Boston coordinates.

Install:
    pip install numpy pandas requests pyserial

Examples:
    python aging_ai_boston.py train --load-csv data/nema_load.csv
    python aging_ai_boston.py predict --load-csv data/nema_load.csv
    python aging_ai_boston.py run --load-csv data/nema_load.csv
    python aging_ai_boston.py status

Expected electricity CSV
------------------------
The parser is intentionally flexible, but the cleanest file is:

    timestamp,demand_mw
    2026-09-18T13:00:00-04:00,3240.5
    2026-09-18T14:00:00-04:00,3312.8

If timestamps have NO timezone, they are interpreted as America/New_York,
then converted to UTC. This handles EST/EDT before merging with weather.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ----------------------------- CONFIG ---------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
LOG_DIR = ROOT / "logs"

MODEL_PATH = STATE_DIR / "aging_ai_boston_model.npz"
DEAD_PATH = STATE_DIR / "dead_neurons.json"
PREDICTION_LOG = LOG_DIR / "predictions.csv"
PULSE_LOG = LOG_DIR / "pulse_commands.csv"

# Boston point used as a weather proxy for the NEMA/Boston installation.
BOSTON_LAT = 42.3601
BOSTON_LON = -71.0589
LOCAL_TZ = "America/New_York"

HIDDEN = 25
RANDOM_SEED = 42

# Historical weather: consistent hourly reanalysis.
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
# Live/recent weather endpoint.
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
]

# These are NN inputs at time t. Target is demand(t+1).
FEATURES = [
    "demand_now",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
    "weekend",
]

# Training settings
EPOCHS = 1200
LEARNING_RATE = 0.003
L2 = 1e-5
VALIDATION_FRACTION = 0.15

# Physical mapping:
# Python decides RELATIVE activity only. Arduino must enforce electrical safety.
MAX_COMMAND_MS = 1500

# Serial protocol settings
DEFAULT_BAUD = 115200
SERIAL_TIMEOUT = 3.0

# Polling. ISO/load data may not appear exactly on the clock hour.
POLL_SECONDS = 300

# Optional authenticated ISO-NE data feed.
# Put credentials in environment variables, never hard-code passwords.
ISO_NE_USER = os.getenv("ISO_NE_USER")
ISO_NE_PASSWORD = os.getenv("ISO_NE_PASSWORD")
ISO_NE_NEMA_ENDPOINT = os.getenv("ISO_NE_NEMA_ENDPOINT")

# ----------------------------------------------------------------------


def ensure_dirs():
    for p in (DATA_DIR, STATE_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def utc_now():
    return pd.Timestamp.now(tz="UTC")


def append_csv(path: Path, row: dict):
    ensure_dirs()
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ========================== ELECTRICITY DATA ===========================

TIME_CANDIDATES = [
    "timestamp", "datetime", "date_time", "time", "period",
    "local time", "local_time", "interval", "hour",
]
LOAD_CANDIDATES = [
    "demand_mw", "load_mw", "demand", "load", "mw",
    "real time demand", "real_time_demand", "rt demand",
    "estimated load", "estimated_load",
]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    # Loose matching for ISO-NE report naming variations.
    for c in df.columns:
        low = str(c).strip().lower()
        if any(candidate in low for candidate in candidates):
            return c
    return None


def _to_utc(series: pd.Series) -> pd.Series:
    """
    Convert timestamps to UTC.

    If source timestamps already contain offsets/timezones, preserve them.
    Otherwise interpret them as Boston/New York local time, then convert UTC.
    """
    parsed = pd.to_datetime(series, errors="coerce")

    # If pandas produced timezone-aware dtype.
    try:
        if parsed.dt.tz is not None:
            return parsed.dt.tz_convert("UTC")
    except AttributeError:
        pass

    # ISO-NE/Boston local timestamps: explicitly resolve DST.
    localized = parsed.dt.tz_localize(
        LOCAL_TZ,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    return localized.dt.tz_convert("UTC")


def load_nema_csv(path: str) -> pd.DataFrame:
    """
    Read a user-provided/exported NEMA electricity CSV and standardize to:
        timestamp_utc, demand_mw

    For unusual ISO-NE report formats, inspect the CSV once and add its exact
    column name to TIME_CANDIDATES / LOAD_CANDIDATES.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in {path}")

    time_col = _find_column(df, TIME_CANDIDATES)
    load_col = _find_column(df, LOAD_CANDIDATES)

    if time_col is None or load_col is None:
        raise ValueError(
            "Could not identify timestamp/load columns.\n"
            f"Columns found: {list(df.columns)}\n"
            "Recommended clean columns: timestamp,demand_mw"
        )

    out = pd.DataFrame()
    out["timestamp_utc"] = _to_utc(df[time_col])
    out["demand_mw"] = pd.to_numeric(
        df[load_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    out = (
        out.dropna()
        .sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .reset_index(drop=True)
    )

    # Force hourly series. If source is 5-minute data, mean gives an hourly
    # representation. If already hourly, this leaves one value per hour.
    out = (
        out.set_index("timestamp_utc")["demand_mw"]
        .resample("1h")
        .mean()
        .dropna()
        .rename("demand_mw")
        .reset_index()
    )
    return out


def fetch_nema_isone_api() -> pd.DataFrame:
    """
    Optional authenticated ISO-NE data-feed hook.

    ISO-NE's exact JSON schema depends on the data-feed endpoint selected.
    Set:
        ISO_NE_USER
        ISO_NE_PASSWORD
        ISO_NE_NEMA_ENDPOINT

    Once you have the endpoint, this function can be adapted to its returned
    JSON fields. We deliberately do not invent an undocumented endpoint.
    """
    if not (ISO_NE_USER and ISO_NE_PASSWORD and ISO_NE_NEMA_ENDPOINT):
        raise RuntimeError(
            "ISO-NE API mode is not configured. Set ISO_NE_USER, "
            "ISO_NE_PASSWORD and ISO_NE_NEMA_ENDPOINT, or use --load-csv."
        )

    r = requests.get(
        ISO_NE_NEMA_ENDPOINT,
        auth=(ISO_NE_USER, ISO_NE_PASSWORD),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()

    # Accept a common/simple list form if your configured feed returns one.
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    if not rows:
        raise ValueError(
            "ISO-NE endpoint responded, but its JSON schema is not recognized. "
            "Inspect one response and adapt fetch_nema_isone_api()."
        )

    raw = pd.DataFrame(rows)
    time_col = _find_column(raw, TIME_CANDIDATES)
    load_col = _find_column(raw, LOAD_CANDIDATES)
    if time_col is None or load_col is None:
        raise ValueError(
            "ISO-NE JSON received but timestamp/load fields were not recognized. "
            f"Fields: {list(raw.columns)}"
        )

    out = pd.DataFrame({
        "timestamp_utc": _to_utc(raw[time_col]),
        "demand_mw": pd.to_numeric(raw[load_col], errors="coerce"),
    }).dropna()

    return (
        out.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .reset_index(drop=True)
    )


# ============================= WEATHER =================================

def _weather_request(url: str, params: dict) -> pd.DataFrame:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    if "hourly" not in j:
        raise ValueError(f"Open-Meteo returned no hourly data: {j}")

    h = j["hourly"]
    out = pd.DataFrame({"timestamp_utc": pd.to_datetime(h["time"], utc=True)})
    for v in WEATHER_VARS:
        out[v] = pd.to_numeric(h[v], errors="coerce")

    return out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")


def fetch_historical_boston_weather(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Hourly Boston weather. timezone=UTC is intentional: electricity is also
    converted to UTC before the merge.
    """
    params = {
        "latitude": BOSTON_LAT,
        "longitude": BOSTON_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
    }
    return _weather_request(OPEN_METEO_ARCHIVE, params)


def fetch_recent_boston_weather(past_days: int = 8) -> pd.DataFrame:
    params = {
        "latitude": BOSTON_LAT,
        "longitude": BOSTON_LON,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": 2,
    }
    return _weather_request(OPEN_METEO_FORECAST, params)


# ============================ DATA MERGE ================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features are based on Boston LOCAL time, even though merging is
    done in UTC. This makes 5 PM mean 5 PM in Boston, including DST.
    """
    out = df.copy()
    local = out["timestamp_utc"].dt.tz_convert(LOCAL_TZ)

    hour = local.dt.hour + local.dt.minute / 60.0
    dow = local.dt.dayofweek
    doy = local.dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["year_sin"] = np.sin(2 * np.pi * doy / 365.2425)
    out["year_cos"] = np.cos(2 * np.pi * doy / 365.2425)
    out["weekend"] = (dow >= 5).astype(float)
    return out


def build_dataset(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Critical rule:
      Electricity and weather are matched by THE SAME UTC HOUR.

    No nearest-neighbor merge is used; an hour must actually match.
    """
    load = load.copy()
    weather = weather.copy()

    load["timestamp_utc"] = pd.to_datetime(load["timestamp_utc"], utc=True)
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)

    load = load.sort_values("timestamp_utc")
    weather = weather.sort_values("timestamp_utc")

    # Demand lags are created BEFORE weather merge.
    load["demand_now"] = load["demand_mw"]
    load["demand_lag_1h"] = load["demand_mw"].shift(1)
    load["demand_lag_24h"] = load["demand_mw"].shift(24)
    load["demand_lag_168h"] = load["demand_mw"].shift(168)

    # Prediction target = demand exactly one hour later.
    load["target_next_hour_mw"] = load["demand_mw"].shift(-1)
    load["target_timestamp_utc"] = load["timestamp_utc"].shift(-1)

    merged = pd.merge(
        load,
        weather,
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    )

    # Protect against a gap being mistaken for "next hour".
    exact_next = (
        merged["target_timestamp_utc"] - merged["timestamp_utc"]
        == pd.Timedelta(hours=1)
    )
    merged = merged.loc[exact_next].copy()

    merged = add_time_features(merged)
    merged = merged.dropna(subset=FEATURES + ["target_next_hour_mw"])
    return merged.reset_index(drop=True)


def build_live_feature_rows(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Same as training features, but does NOT require the future target.
    Used for exhibition inference.
    """
    load = load.copy().sort_values("timestamp_utc")
    load["timestamp_utc"] = pd.to_datetime(load["timestamp_utc"], utc=True)

    load["demand_now"] = load["demand_mw"]
    load["demand_lag_1h"] = load["demand_mw"].shift(1)
    load["demand_lag_24h"] = load["demand_mw"].shift(24)
    load["demand_lag_168h"] = load["demand_mw"].shift(168)

    weather = weather.copy()
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)

    merged = pd.merge(load, weather, on="timestamp_utc", how="inner")
    merged = add_time_features(merged)
    return merged.dropna(subset=FEATURES).reset_index(drop=True)


# =========================== NEURAL NETWORK =============================

@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean, std)

    def transform(self, x):
        return (x - self.mean) / self.std


class AgingNet:
    """
    One hidden layer, exactly 25 hidden neurons.

        input features -> 25 ReLU hidden neurons -> one demand prediction

    One hidden layer is intentional: it gives a clean 1:1 interpretation
    between software hidden neuron i and physical copper cell i.
    """

    def __init__(self, n_features: int, seed: int = RANDOM_SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(
            0, np.sqrt(2.0 / n_features), (n_features, HIDDEN)
        )
        self.b1 = np.zeros(HIDDEN)
        self.W2 = rng.normal(
            0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 1)
        )
        self.b2 = np.zeros(1)

        self.x_scaler: Optional[Scaler] = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.dead = np.zeros(HIDDEN, dtype=bool)

    def forward_scaled(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)

        # A physically dead copper cell removes that neuron from inference.
        a1[:, self.dead] = 0.0

        y = a1 @ self.W2 + self.b2
        return y[:, 0], a1

    def fit(self, X_raw, y_raw, epochs=EPOCHS, lr=LEARNING_RATE):
        self.x_scaler = Scaler.fit(X_raw)
        X = self.x_scaler.transform(X_raw)

        self.y_mean = float(np.mean(y_raw))
        self.y_std = float(np.std(y_raw))
        if self.y_std < 1e-8:
            self.y_std = 1.0
        y = (y_raw - self.y_mean) / self.y_std

        n = len(X)
        split = max(1, int(n * (1.0 - VALIDATION_FRACTION)))
        Xtr, ytr = X[:split], y[:split]
        Xva, yva = X[split:], y[split:]

        best = None
        best_val = float("inf")

        # Full-batch Adam, implemented in NumPy.
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, epochs + 1):
            pred, a1 = self.forward_scaled(Xtr)
            err = pred - ytr
            loss = np.mean(err ** 2)

            # Backpropagation.
            d_out = (2.0 / len(Xtr)) * err[:, None]
            dW2 = a1.T @ d_out + L2 * self.W2
            db2 = d_out.sum(axis=0)

            da1 = d_out @ self.W2.T
            z1 = Xtr @ self.W1 + self.b1
            dz1 = da1 * (z1 > 0)

            dW1 = Xtr.T @ dz1 + L2 * self.W1
            db1 = dz1.sum(axis=0)
            grads = [dW1, db1, dW2, db2]

            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                mhat = m[i] / (1 - beta1 ** epoch)
                vhat = v[i] / (1 - beta2 ** epoch)
                p -= lr * mhat / (np.sqrt(vhat) + eps)

            if len(Xva):
                vp, _ = self.forward_scaled(Xva)
                val = float(np.mean((vp - yva) ** 2))
            else:
                val = float(loss)

            if val < best_val:
                best_val = val
                best = [p.copy() for p in params]

            if epoch == 1 or epoch % 100 == 0:
                print(
                    f"epoch {epoch:4d} | "
                    f"train MSE(scaled)={loss:.5f} | "
                    f"val MSE(scaled)={val:.5f}"
                )

        if best is not None:
            self.W1[:], self.b1[:], self.W2[:], self.b2[:] = best

    def predict(self, X_raw):
        if self.x_scaler is None:
            raise RuntimeError("Model has no scaler.")
        if self.dead.all():
            raise RuntimeError("All 25 physical neurons are dead.")

        X = self.x_scaler.transform(np.asarray(X_raw, dtype=float))
        y_scaled, activations = self.forward_scaled(X)
        y_mw = y_scaled * self.y_std + self.y_mean

        # For this simple 25-hidden-neuron / one-output architecture,
        # |activation * outgoing weight| is a transparent contribution proxy.
        contributions = np.abs(activations * self.W2[:, 0])

        return y_mw, activations, contributions

    def save(self, path=MODEL_PATH):
        if self.x_scaler is None:
            raise RuntimeError("Cannot save untrained model.")
        ensure_dirs()
        tmp = str(path) + ".tmp.npz"
        np.savez(
            tmp,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            x_mean=self.x_scaler.mean,
            x_std=self.x_scaler.std,
            y_mean=np.array([self.y_mean]),
            y_std=np.array([self.y_std]),
            feature_names=np.array(FEATURES),
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        s = np.load(path, allow_pickle=False)
        saved_features = list(s["feature_names"].astype(str))
        if saved_features != FEATURES:
            raise RuntimeError(
                "Saved model feature list differs from current code.\n"
                f"Saved: {saved_features}\nCurrent: {FEATURES}"
            )

        net = cls(len(FEATURES))
        net.W1 = s["W1"]
        net.b1 = s["b1"]
        net.W2 = s["W2"]
        net.b2 = s["b2"]
        net.x_scaler = Scaler(s["x_mean"], s["x_std"])
        net.y_mean = float(s["y_mean"][0])
        net.y_std = float(s["y_std"][0])
        net.dead = load_dead_state()
        return net


# ============================ COPPER STATE ==============================

def load_dead_state() -> np.ndarray:
    dead = np.zeros(HIDDEN, dtype=bool)
    if DEAD_PATH.exists():
        j = json.loads(DEAD_PATH.read_text())
        for i in j.get("dead_neurons", []):
            if 0 <= int(i) < HIDDEN:
                dead[int(i)] = True
    return dead


def save_dead_state(dead: np.ndarray):
    ensure_dirs()
    tmp = DEAD_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {
            "dead_neurons": np.where(dead)[0].astype(int).tolist(),
            "alive_count": int((~dead).sum()),
            "updated_utc": utc_now().isoformat(),
        },
        indent=2,
    ))
    os.replace(tmp, DEAD_PATH)


def mark_dead(net: AgingNet, indices):
    changed = []
    for i in indices:
        i = int(i)
        if 0 <= i < HIDDEN and not net.dead[i]:
            net.dead[i] = True
            changed.append(i)
    if changed:
        save_dead_state(net.dead)
    return changed


def contribution_to_commands(contrib: np.ndarray, dead: np.ndarray):
    """
    Convert neural contribution to relative physical activity.

    Total command budget is capped at MAX_COMMAND_MS per inference.
    This is NOT a voltage/current specification.
    """
    c = np.asarray(contrib, dtype=float).copy()
    c[dead] = 0.0

    total = c.sum()
    if total <= 0:
        share = np.zeros(HIDDEN)
    else:
        share = c / total

    ms = np.rint(share * MAX_COMMAND_MS).astype(int)
    ms[dead] = 0
    return share, ms


# ============================== ARDUINO ================================

class ArduinoBridge:
    """
    Simple newline-delimited JSON protocol.

    Python -> Arduino:
        {"cmd":"pulse","cell":7,"ms":84}

    Arduino -> Python ACK:
        {"ok":true,"cell":7,"delivered_ms":84}

    Python -> Arduino:
        {"cmd":"scan"}

    Arduino -> Python:
        {"ok":true,"dead":[2,7,18]}

    The Arduino firmware is responsible for the actual switching and for
    enforcing all electrical limits. This class only communicates commands.
    """

    def __init__(self, port: str, baud=DEFAULT_BAUD):
        try:
            import serial
        except ImportError as e:
            raise RuntimeError(
                "pyserial is required for Arduino mode: pip install pyserial"
            ) from e

        self.serial = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=SERIAL_TIMEOUT,
        )
        time.sleep(2.0)  # many Arduino boards reset when serial opens
        self.serial.reset_input_buffer()

    def _exchange(self, message: dict) -> dict:
        wire = (json.dumps(message) + "\n").encode("utf-8")
        self.serial.write(wire)
        self.serial.flush()

        line = self.serial.readline().decode("utf-8", errors="replace").strip()
        if not line:
            raise TimeoutError(f"No Arduino ACK for {message}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bad Arduino response: {line!r}") from e

        if not reply.get("ok", False):
            raise RuntimeError(f"Arduino rejected command: {reply}")
        return reply

    def pulse(self, cell: int, ms: int) -> dict:
        return self._exchange({"cmd": "pulse", "cell": int(cell), "ms": int(ms)})

    def scan_dead(self) -> list[int]:
        reply = self._exchange({"cmd": "scan"})
        return [int(x) for x in reply.get("dead", [])]

    def close(self):
        self.serial.close()


# ========================== TRAIN / PREDICT =============================

def get_load(args) -> pd.DataFrame:
    if args.load_csv:
        return load_nema_csv(args.load_csv)
    return fetch_nema_isone_api()


def training_weather_for_load(load: pd.DataFrame) -> pd.DataFrame:
    start = load["timestamp_utc"].min().date().isoformat()
    end = load["timestamp_utc"].max().date().isoformat()
    print(f"Fetching Boston weather {start} -> {end} ...")
    return fetch_historical_boston_weather(start, end)


def train_model(args):
    ensure_dirs()
    load = get_load(args)
    if len(load) < 24 * 30:
        print(
            "WARNING: less than ~30 days of electricity data. "
            "For a serious model, use substantially more historical data."
        )

    weather = training_weather_for_load(load)
    ds = build_dataset(load, weather)

    if len(ds) < 200:
        raise RuntimeError(
            f"Only {len(ds)} matched training rows after UTC merge/lags."
        )

    X = ds[FEATURES].to_numpy(dtype=float)
    y = ds["target_next_hour_mw"].to_numpy(dtype=float)

    # Chronological evaluation: no random shuffle.
    split = int(len(X) * 0.85)
    net = AgingNet(len(FEATURES))
    net.fit(X[:split], y[:split])

    pred, _, _ = net.predict(X[split:])
    actual = y[split:]
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

    print("\nHoldout performance")
    print(f"  rows: {len(actual)}")
    print(f"  MAE : {mae:,.1f} MW")
    print(f"  RMSE: {rmse:,.1f} MW")

    # Final training on all historical rows with fresh network.
    print("\nTraining final frozen 25-neuron model on all available rows...")
    final = AgingNet(len(FEATURES))
    final.fit(X, y)
    final.save()

    # A new trained model does NOT resurrect physical neurons.
    if DEAD_PATH.exists():
        print(
            "Existing physical death-state retained. "
            "Delete it manually only when installing a new physical body."
        )

    print(f"\nSaved frozen model: {MODEL_PATH}")
    print("Training is finished. Exhibition mode performs inference only.")


def latest_prediction(args, arduino: Optional[ArduinoBridge] = None):
    net = AgingNet.load()
    load = get_load(args)

    # We need at least 168h history for the one-week lag.
    if len(load) < 169:
        raise RuntimeError("Need at least 169 hourly electricity observations.")

    weather = fetch_recent_boston_weather(past_days=8)
    live = build_live_feature_rows(load, weather)

    if live.empty:
        raise RuntimeError(
            "No electricity/weather hours matched exactly in UTC."
        )

    row = live.iloc[-1]
    ts = row["timestamp_utc"]
    X = row[FEATURES].to_numpy(dtype=float)[None, :]

    pred, act, contrib = net.predict(X)
    pred = float(pred[0])
    act = act[0]
    contrib = contrib[0]

    share, command_ms = contribution_to_commands(contrib, net.dead)

    print("\n" + "=" * 70)
    print("AGING AI — NEMA / BOSTON")
    print(f"Input hour UTC      : {ts}")
    print(f"Input hour Boston   : {ts.tz_convert(LOCAL_TZ)}")
    print(f"Observed NEMA load  : {row['demand_now']:,.1f} MW")
    print(f"Predicted next hour : {pred:,.1f} MW")
    print(f"Living neurons      : {(~net.dead).sum()} / {HIDDEN}")
    print("=" * 70)

    # Highest contribution first for readable terminal output.
    order = np.argsort(-contrib)
    for i in order:
        status = "DEAD" if net.dead[i] else "alive"
        print(
            f"N{i:02d} | {status:5s} | "
            f"activation={act[i]:9.4f} | "
            f"contribution={contrib[i]:9.4f} | "
            f"share={share[i]:7.3%} | "
            f"command={command_ms[i]:4d} ms"
        )

    append_csv(PREDICTION_LOG, {
        "processed_utc": utc_now().isoformat(),
        "input_timestamp_utc": ts.isoformat(),
        "input_timestamp_boston": ts.tz_convert(LOCAL_TZ).isoformat(),
        "observed_demand_mw": float(row["demand_now"]),
        "predicted_next_hour_mw": pred,
        "alive_neurons": int((~net.dead).sum()),
    })

    if arduino is not None:
        # First synchronize software with physical copper.
        physical_dead = arduino.scan_dead()
        newly = mark_dead(net, physical_dead)
        if newly:
            print(f"Physical conduction detected; neurons died: {newly}")
            # Recompute prediction without newly dead neurons.
            pred_arr, act_arr, contrib_arr = net.predict(X)
            pred = float(pred_arr[0])
            act = act_arr[0]
            contrib = contrib_arr[0]
            share, command_ms = contribution_to_commands(contrib, net.dead)
            print(f"Prediction after death masking: {pred:,.1f} MW")

        # Sequential commands make ACK/accounting unambiguous.
        for i, ms in enumerate(command_ms):
            if net.dead[i] or ms <= 0:
                continue
            reply = arduino.pulse(i, int(ms))
            delivered = int(reply.get("delivered_ms", ms))
            append_csv(PULSE_LOG, {
                "utc": utc_now().isoformat(),
                "input_timestamp_utc": ts.isoformat(),
                "cell": i,
                "requested_ms": int(ms),
                "delivered_ms": delivered,
                "contribution": float(contrib[i]),
                "share": float(share[i]),
            })

        # Scan again because a deposition event may have bridged a cell.
        physical_dead = arduino.scan_dead()
        newly = mark_dead(net, physical_dead)
        if newly:
            print(f"Newly conductive after deposition: {newly}")

    return ts


def run_exhibition(args):
    """
    Long-running installation mode.

    Poll every few minutes, but only perform ONE neural inference/deposition
    when a NEW matched electricity hour appears. Therefore repeated API polls
    do not repeatedly consume copper for the same observation.
    """
    arduino = None
    if args.serial_port:
        arduino = ArduinoBridge(args.serial_port, args.baud)
        print(f"Arduino connected: {args.serial_port}")
    else:
        print("DRY RUN: no --serial-port supplied; no copper commands will be sent.")

    last_processed = None
    print(f"Polling every {args.poll_seconds} seconds.")

    try:
        while True:
            try:
                ts = latest_prediction(args, arduino=None)

                if last_processed is None:
                    last_processed = ts
                    if arduino is not None:
                        # First observation of this process: perform it once.
                        latest_prediction(args, arduino=arduino)

                elif ts > last_processed:
                    print(f"\nNew hourly observation: {ts}")
                    if arduino is not None:
                        latest_prediction(args, arduino=arduino)
                    last_processed = ts
                else:
                    print(f"No new matched hour. Latest remains {ts}")

            except Exception as e:
                print(f"[{utc_now().isoformat()}] cycle error: {e}")

            time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print("\nStopping exhibition.")
    finally:
        if arduino is not None:
            arduino.close()


def show_status():
    dead = load_dead_state()
    print(f"Model exists : {MODEL_PATH.exists()}")
    print(f"Alive        : {(~dead).sum()} / {HIDDEN}")
    print(f"Dead neurons : {np.where(dead)[0].tolist()}")
    if PREDICTION_LOG.exists():
        p = pd.read_csv(PREDICTION_LOG)
        if len(p):
            print("\nLatest prediction log:")
            print(p.tail(1).to_string(index=False))


# ================================ CLI ==================================

def add_data_args(p):
    p.add_argument(
        "--load-csv",
        help="NEMA electricity CSV. If omitted, configured ISO-NE API is used.",
    )


def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="Aging AI: 25-neuron NEMA/Boston demand predictor"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train once and freeze the model")
    add_data_args(p_train)

    p_predict = sub.add_parser(
        "predict",
        help="One inference; no training and no hardware unless --serial-port",
    )
    add_data_args(p_predict)
    p_predict.add_argument("--serial-port")
    p_predict.add_argument("--baud", type=int, default=DEFAULT_BAUD)

    p_run = sub.add_parser("run", help="Continuous exhibition polling")
    add_data_args(p_run)
    p_run.add_argument("--serial-port")
    p_run.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p_run.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)

    sub.add_parser("status", help="Show model/physical neuron status")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "predict":
        bridge = (
            ArduinoBridge(args.serial_port, args.baud)
            if args.serial_port else None
        )
        try:
            latest_prediction(args, bridge)
        finally:
            if bridge:
                bridge.close()
    elif args.command == "run":
        run_exhibition(args)
    elif args.command == "status":
        show_status()


if __name__ == "__main__":
    main()
