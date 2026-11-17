"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day, from everything recorded up
to the end of its first hour, as five quantiles an hour (5%, 25%, 50%, 75%,
95%). Scored by pinball loss on the log scale; must beat the house
persistence baseline and a house AI model.

Method: hour-1-anchored log-ratio quantiles from comparable launches.
Each comparable launch in the case file contributes its release-day curve
normalised by its own hour-1 print: r_h = viewers_h / viewers_1. For each
forecast hour the program takes the empirical median of ln(r_h) across
donors and spreads the five quantiles around it with the donors' own
log-spread (sample sd), so the forecast is q_tau(h) = p1 * exp(med_h +
z_tau * sd_h). Donors are the clean comparable launches whose launch was
already underway at hour 1 (hour-1 at least 5% of their release-day peak):
rolling-midnight launches like GTA VI itself, whose US audience unlocks
hours 5-8 and whose audience grows into the evening. Early-access
contaminated launches, weak analogues and GTA VI's own pre-release
category history are excluded. Everything is computed at run time from
the case file, so the final case (which adds the four practice launches
as comparables) is handled the same way.

If anything unexpected happens for a case, the program falls back to the
house persistence forecast for that case, so the output stays valid:
a valid mediocre forecast beats an invalid case (an invalid case takes
no prize).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there): JSON {"schema":
    "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint",
    "levels", "known", "forecast", "history", "history_hourly", "streams",
    "categories_hourly", "comparables", "notes"};
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most
    1e9 (viewers; fractions allowed), non-decreasing; at most 65,536
    bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# The house baseline's spread, in natural-log units per square-root hour,
# used only as the fallback when the comparable model cannot run.
BASE_SIGMA = 0.15
# A donor counts as "launch already underway at hour 1" when its hour-1
# print is at least this share of its own release-day peak. Rolling
# midnight launches clear it; afternoon simultaneous unlocks do not.
UNDERWAY_MIN = 0.05
# Minimum donors with a value for an hour before the model trusts it.
MIN_DONORS = 3


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def tier_of(comp: dict) -> str:
    """The house's tier for a comparable, from its unlock note."""
    u = comp.get("unlock") or ""
    if "Tier: early-access-contaminated" in u:
        return "early"
    if "Tier: weak" in u:
        return "weak"
    if "Tier: target-history" in u:
        return "target-history"
    if "Tier: comparable" in u:
        return "comparable"
    return "other"


def hourly_curve(comp: dict) -> dict:
    """A comparable's release-day hourly means: {hour 1..24: viewers}."""
    try:
        ds = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return {}
    buckets: dict = {}
    for p in comp.get("hourly") or []:
        try:
            t = parse_ts(p["from"])
            v = p["viewers"]
        except (KeyError, ValueError, TypeError):
            continue
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and isinstance(v, (int, float)):
            buckets.setdefault(h, []).append(max(0, v))
    return {h: sum(v) / len(v) for h, v in buckets.items()}


def donor_curves(case: dict) -> list:
    """Clean comparable curves, in name order for determinism."""
    out = []
    comps = sorted(case.get("comparables") or [], key=lambda c: str(c.get("name", "")))
    for comp in comps:
        if tier_of(comp) != "comparable":
            continue
        curve = hourly_curve(comp)
        h1 = curve.get(1)
        if not h1 or h1 < 1:
            continue
        peak = max(curve.values())
        if h1 / peak < UNDERWAY_MIN:
            continue
        out.append(curve)
    return out


def quantile(sorted_vals: list, tau: float) -> float:
    """Linear-interpolation (type-7) quantile of a sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def log_ratio_table(donors: list) -> dict:
    """{hour: sorted list of ln(viewers_h / viewers_1)} across donors."""
    table: dict = {h: [] for h in range(2, 25)}
    for curve in donors:
        h1 = curve.get(1)
        if not h1 or h1 < 1:
            continue
        for h in range(2, 25):
            y = curve.get(h)
            if y is not None and y >= 1:
                table[h].append(math.log(y / h1))
    for h in table:
        table[h].sort()
    return table


def persistence_quantiles(p1: float, h: int) -> list:
    """The house baseline: every hour stays at hour 1's print, spread
    widening like a random walk's in log space."""
    qs = [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in Z]
    # The contract caps every quantile at 1e9 viewers.
    return [min(max(1, q), 1e9) for q in qs]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    donors = donor_curves(case)
    table = log_ratio_table(donors)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        vals = table.get(h) or []
        if len(vals) < MIN_DONORS:
            rows.append({"hour": h, "q": persistence_quantiles(p1, h)})
            continue
        med = quantile(vals, 0.5)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        sd = math.sqrt(var)
        qs = sorted(round2(max(1.0, p1 * math.exp(med + z * sd))) for z in Z)
        # Guard the contract: five finite numbers, > 0, at most 1e9,
        # non-decreasing.
        qs = [min(max(1, q), 1e9) for q in qs]
        qs = sorted(qs)
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def fallback_case(case: dict, name: str) -> dict:
    """Persistence forecast, used when the comparable model fails: valid
    output is always better than an invalid case."""
    try:
        first = next((k for k in case["known"] if k["hour"] == 1), None)
        p1 = max(1, first["viewers"]) if first else 1
        hours = [row["hour"] for row in case["forecast"]]
    except (KeyError, TypeError):
        p1 = 1
        hours = list(range(2, 25))
    return {"hours": [{"hour": h, "q": persistence_quantiles(p1, h)} for h in hours]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Any unexpected failure falls back to persistence: a valid
            # mediocre forecast is always better than an invalid case
            # (an invalid case takes no prize).
            forecast = fallback_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
