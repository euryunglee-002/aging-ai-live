#!/usr/bin/env python3
"""
Aging AI — hourly forecast in the cloud (GitHub Actions), without copper.

Same frozen model (aging_ai_boston_model.npz) and the same live data
as the installation: ISO New England NEMA demand + Open-Meteo Boston weather.
Nothing is sent to any copper. Writes status.json at the site root.

If the Raspberry Pi of the installation has published within the last 3 hours,
this script does nothing: the installation's own record always comes first.
"""
import json
import re
import shutil
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
HISTORY_HOURS = 72
PI_PRIORITY_HOURS = 3


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    now = datetime.now(timezone.utc)
    if old.get("mode") == "mirror" and old.get("updated_utc") and \
            now - _t(old["updated_utc"]) < timedelta(hours=PI_PRIORITY_HOURS):
        print("The installation published recently; leaving its record as it is.")
        return

    if not R.B.MODEL_PATH.exists():
        print("No trained model found (aging_ai_boston_model.npz).")
        return

    try:
        p = R.predict_now()
    except Exception as e:                      # keep the last good forecast, note the error
        old["error"] = re.sub(r"://[^/@\s]+@", "://", str(e))[:200]
        old["updated_utc"] = now.isoformat()
        OUT.write_text(json.dumps(old, indent=1))
        print("forecast failed:", old["error"])
        return

    R.write_status(p, sent={"status": "view only"})
    st = json.loads(R.STATUS_PATH.read_text())

    hist = {h["t"]: h for h in old.get("history", []) if "t" in h}
    hist[st["input_hour_utc"]] = {"t": st["input_hour_utc"], "observed": st["observed_mw"],
                                  "forecast": st["forecast_mw"], "alive": st["alive"]}
    newest = _t(st["input_hour_utc"])
    st["history"] = [hist[k] for k in sorted(hist)
                     if newest - _t(k) <= timedelta(hours=HISTORY_HOURS - 1)]
    st["copper_ms"] = [0] * R.B.HIDDEN          # no copper attached
    st["mode"] = "standalone"
    st["has_forecast"] = True
    OUT.write_text(json.dumps(st, indent=1))
    print(f"{st['input_hour_boston']}: observed {st['observed_mw']:,} MW -> "
          f"forecast {st['forecast_mw']:,} MW, {st['alive']} neurons")


if __name__ == "__main__":
    main()
