"""
SAR ship detections <-> AIS matching, VERSION 2
===============================================
Builds on ais_matching.py. What is new:

  1. Scene MIDDLE time     A Sentinel-1 scene takes ~20-25 s to capture. We match at the
                           middle of the scene and count the leftover +/- half-duration
                           as uncertainty.
  2. Doppler correction    Moving ships appear shifted sideways (along the satellite's track)
                           in SAR images. We compute the shift from each ship's speed and
                           course, and move its AIS position to where SAR would show it.
  3. Match SCORES          Every pair gets a 0-1 score: how well this AIS ship explains this
                           detection compared with every other explanation. Labels:
                           matched / possible_dark / likely_dark.
  4. Size check            Detection length (from its pixel count) is compared with the AIS
                           length. A 300 m blob "matched" to a 12 m boat can never be an automatic match.
  5. Operational outputs   Pass time, processing time, a growing "could now be within X km"
                           search radius, an AIS-freshness warning, and a re-run comparison.

Version 2.1: the fake AIS is now built ONLY from your detections (no AIS ships without a
radar partner, unless you ask for them), the summary shows how the counts add up, and
demo mode prints an ANSWER KEY so you can check the matcher against the truth.

AIS data: the script starts with FAKE AIS (USE_DEMO_AIS = True).
For real AIS set USE_DEMO_AIS = False and AIS_CSV_PATH = "your_file.csv".

Notebook use: run your detection cells first (so `features`, `first` and `AOI` exist),
then:   %run -i ais_matching_v2.py
Your results are then in the dict `v2`, e.g. v2['results'] (one row per detection).
Your own notebook variables (such as `ships`) are NOT touched.

Needs:  pip install pandas numpy scipy folium
"""

import html
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ===============================================================
# 1. SETTINGS
# ===============================================================
AOI_BBOX = (103.80, 1.20, 103.90, 1.28)   # (west, south, east, north), same box as the detection step

# --- AIS source ------------------------------------------------
USE_DEMO_AIS = True                  # True = FAKE AIS for testing. False = read AIS_CSV_PATH
AIS_CSV_PATH = "ais_positions.csv"   # real AIS file (NOAA, Danish DMA, or similar columns)
AIS_TIME_FORMAT = None               # None = auto-detect. Example: "%d/%m/%Y %H:%M:%S"
SIMULATE_AIS_AVAILABLE_UNTIL_MIN = None   # e.g. 3 = pretend you ran only 3 min after the pass

# --- Detections source (optional) -------------------------------
DETECTIONS_CSV_PATH = None           # e.g. "detections.csv" (columns lon, lat, optional count). None = use
                                     # `features` / `coords` from the notebook

# --- Fake-AIS world (only used when USE_DEMO_AIS = True) --------
DEMO_AIS_SHARE = 0.65                # share of detections that broadcast AIS (the rest are dark ON PURPOSE)
DEMO_EXTRA_AIS_SHIPS = 0             # AIS ships the radar never saw. 0 = every AIS ship has a detection.
                                     # Try 8 to practise reading the 'AIS ships with no radar partner' line
DEMO_SIZE_MISMATCH_SHARE = 0.0       # share of AIS ships given a wrong length. Try 0.12 to see the size check work

# --- Upgrade 1: scene time ------------------------------------
PLACEHOLDER_SCENE_START = "2026-09-01 22:50:00+00:00"   # used only if no Earth Engine scene is found
DEFAULT_SCENE_DURATION_S = 25        # used only if Earth Engine gives no usable scene end time

# --- Upgrade 2: Doppler correction ----------------------------
APPLY_DOPPLER_CORRECTION = True
DIAGNOSE_DOPPLER = True              # also match WITHOUT the correction and print a comparison
FALLBACK_PASS = "ASCENDING"          # used if the scene's pass direction is unknown
FALLBACK_INCIDENCE_DEG = 38.0        # used if the incidence angle is unknown

# --- Where was each AIS ship at the scene time? ---------------
SEARCH_WINDOW_MIN = 30     # only use AIS reports within +/- this many minutes of the scene time
MAX_INTERP_SPAN_MIN = 20   # interpolate if the two reports around the pass are at most this far apart
MAX_EXTRAP_MIN = 10        # otherwise project the nearest report, if it is within this many minutes
DEFAULT_SOG_KN = 10        # speed assumed when an AIS report has none
AOI_PAD_M = 3000           # keep AIS ships slightly outside the box so edge ships can still match

# --- Upgrade 3: match scores (an "uncertainty budget", in metres) ---
SIGMA_BASE_M = 40          # SAR geolocation + pixel size + AIS GPS error
DEFAULT_LENGTH_M = 100     # ship length assumed when AIS has none
LENGTH_SIGMA_FRAC = 0.29   # the bright spot can be anywhere along the hull (~ length / sqrt(12))
INTERP_ERR_FRAC = 0.03     # position error per metre travelled between reports, when interpolating
EXTRAP_ERR_FRAC = 0.15     # same, when projecting a report forward/backward (less reliable)
DOPPLER_ERR_FRAC = 0.25    # how much of the computed Doppler shift we do not trust
CLUTTER_PER_KM2 = 0.1      # unrelated (non-AIS) ships expected per km2: higher = stricter matching
RADAR_DETECT_PROB = 0.9    # chance that radar detects an AIS ship that is really there (higher = pairs more eagerly)
MATCH_SCORE = 0.5          # score >= this -> "matched"
MIN_SCORE = 0.05           # score below this -> "no plausible AIS ship"

# --- Upgrade 4: size check ------------------------------------
PIXEL_SIZE_M = 10          # Sentinel-1 GRD pixel size
SHIP_ASPECT_RATIO = 6      # typical length : width, used to turn pixel count into length
SIZE_RATIO_LIMIT = 3       # SAR and AIS lengths may differ by up to this factor
SIZE_MISMATCH_PENALTY = 0.1    # lowers the likelihood of a pair whose sizes disagree
SIZE_MISMATCH_MAX_SCORE = 0.4  # ...and caps its score, so a size mismatch is never an automatic 'matched'

# --- Upgrade 5: operational outputs ---------------------------
PROCESSED_AT = "assumed"   # "assumed" = scene time + ASSUMED_DELAY_MIN (for learning); "now" = real clock
ASSUMED_DELAY_MIN = 30     # roughly the delivery time of near-real-time SAR services
TYPICAL_SPEED_KN = 12      # a dark ship's likely cruising speed
MAX_SPEED_KN = 25          # a dark ship's plausible top speed
MAX_DRAW_RADIUS_KM = 100   # do not draw search circles bigger than this on the map
FRESH_AIS_MIN = 10         # warn if the AIS data ends less than this many minutes after the pass
PREVIOUS_STATUS_CSV = None  # path of a status_*.csv from an earlier run, to see what changed

# --- Physical constants ---------------------------------------
KNOT_TO_MS = 0.514444
M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON_EQ = 111_320.0
EARTH_RADIUS_M = 6_371_000.0
S1_ALTITUDE_M = 693_000.0
S1_SPEED_MS = 7_500.0
S1_INCLINATION_DEG = 98.18
EARTH_ROT_RAD_S = 7.2921159e-5

RESULT_COLS = [
    "det_id", "det_key", "lon", "lat", "status", "score", "note", "mmsi", "name", "ship_type",
    "ais_true_lon", "ais_true_lat", "ais_img_lon", "ais_img_lat", "distance_m", "sigma_m", "z",
    "doppler_shift_m", "sar_length_m", "ais_length_m", "size_check", "sog_kn", "cog", "method",
    "gap_s", "nearest_ais_m",
]


# ===============================================================
# 2. SMALL HELPERS
# ===============================================================
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (numpy arrays welcome, they broadcast)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin((np.radians(lon2) - np.radians(lon1)) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def move_m(lat, lon, east_m, north_m):
    """Shift a position by some metres east and north (fine for distances of a few km)."""
    return (lat + north_m / M_PER_DEG_LAT,
            lon + east_m / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat))))


def project(lat, lon, sog_kn, cog_deg, dt_s):
    """Dead reckoning: slide a position along its course for dt seconds (dt may be negative)."""
    dist = np.asarray(sog_kn, float) * KNOT_TO_MS * dt_s
    brg = np.radians(cog_deg)
    return move_m(lat, lon, dist * np.sin(brg), dist * np.cos(brg))


def in_bbox(lat, lon, bbox, pad_m=0.0):
    west, south, east, north = bbox
    dlat = pad_m / M_PER_DEG_LAT
    dlon = pad_m / (M_PER_DEG_LON_EQ * math.cos(math.radians((south + north) / 2)))
    return ((lat >= south - dlat) & (lat <= north + dlat)
            & (lon >= west - dlon) & (lon <= east + dlon))


# ===============================================================
# 3. SCENE TIME AND SAR GEOMETRY   (upgrades 1 and 2)
# ===============================================================
def s1_heading_deg(lat_deg, pass_dir):
    """Direction the satellite flies over the ground at this latitude (degrees from north)."""
    phi = math.radians(lat_deg)
    sin_a = math.cos(math.radians(S1_INCLINATION_DEG)) / math.cos(phi)
    a = math.asin(max(-1.0, min(1.0, sin_a)))                        # heading in space (ascending)
    rot = EARTH_ROT_RAD_S * (EARTH_RADIUS_M + S1_ALTITUDE_M) * math.cos(phi)   # Earth turns under it
    v_east = S1_SPEED_MS * math.sin(a) - rot
    v_north = S1_SPEED_MS * math.cos(a)
    if str(pass_dir).upper().startswith("DESC"):
        v_north = -v_north
    return math.degrees(math.atan2(v_east, v_north)) % 360.0


def slant_range_m(incidence_deg):
    """Distance from the satellite to the ground target (curved-Earth geometry)."""
    th = math.radians(incidence_deg)
    r_sat = EARTH_RADIUS_M + S1_ALTITUDE_M
    sin_look = EARTH_RADIUS_M * math.sin(th) / r_sat
    return r_sat * math.sqrt(1 - sin_look ** 2) - EARTH_RADIUS_M * math.cos(th)


def doppler_k_s(incidence_deg):
    """Seconds. Metres of along-track shift per m/s of ship speed straight toward/away from the satellite."""
    th = math.radians(incidence_deg)
    return slant_range_m(incidence_deg) * math.sin(th) / S1_SPEED_MS


def doppler_shift_m(sog_kn, cog_deg, look_deg, incidence_deg):
    """
    How far a moving ship appears displaced ALONG THE SATELLITE'S TRACK (metres).
    Positive = forward (the way the satellite flies), negative = backward.
    A ship moving TOWARD the satellite shifts forward; AWAY shifts backward.
    """
    v = np.asarray(sog_kn, float) * KNOT_TO_MS
    v_away = v * np.cos(np.radians(np.asarray(cog_deg, float) - look_deg))   # + = moving away from the track
    return -doppler_k_s(incidence_deg) * v_away


def apply_azimuth_shift(lat, lon, shift_m, heading_deg):
    """Move positions by shift_m along the satellite's flight direction."""
    h = np.radians(heading_deg)
    return move_m(lat, lon, shift_m * np.sin(h), shift_m * np.cos(h))


def read_scene_metadata(scene=None, aoi=None):
    """
    Read scene times, pass direction and incidence angle from an Earth Engine image.
    Anything missing falls back to the settings at the top (and a note says so).
    """
    notes = []
    t_start = t_end = None
    pass_dir, incidence = FALLBACK_PASS, FALLBACK_INCIDENCE_DEG

    if scene is not None:
        try:
            props = scene.toDictionary().getInfo()
            ms0 = props["system:time_start"]
            t_start = pd.to_datetime(ms0, unit="ms", utc=True)
            ms1 = props.get("system:time_end")
            if ms1 is not None and 5 <= (ms1 - ms0) / 1000.0 <= 180:
                t_end = pd.to_datetime(ms1, unit="ms", utc=True)
            if props.get("orbitProperties_pass"):
                pass_dir = props["orbitProperties_pass"]
            else:
                notes.append("pass direction not found: using " + FALLBACK_PASS)
        except Exception as err:                                   # noqa: BLE001
            t_start = None
            notes.append(f"could not read the scene properties ({err})")
        try:
            if aoi is not None:
                import ee
                stats = scene.select("angle").reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=aoi, scale=1000, bestEffort=True).getInfo()
                if stats and stats.get("angle") is not None:
                    incidence = float(stats["angle"])
                else:
                    notes.append("incidence angle not found: using the fallback value")
            else:
                notes.append("no AOI given: using the fallback incidence angle")
        except Exception as err:                                   # noqa: BLE001
            notes.append(f"could not read the incidence angle ({err})")
    else:
        notes.append("no Earth Engine scene found: using placeholder time, pass and incidence angle")

    if t_start is None:
        t_start = pd.Timestamp(PLACEHOLDER_SCENE_START)
    if t_end is None:
        t_end = t_start + pd.Timedelta(seconds=DEFAULT_SCENE_DURATION_S)
        notes.append(f"scene end time unavailable: assumed a {DEFAULT_SCENE_DURATION_S} s scene")

    lat_c = (AOI_BBOX[1] + AOI_BBOX[3]) / 2
    heading = s1_heading_deg(lat_c, pass_dir)
    return dict(
        t_start=t_start, t_end=t_end, t_mid=t_start + (t_end - t_start) / 2,
        duration_s=(t_end - t_start).total_seconds(), pass_dir=str(pass_dir).upper(),
        incidence_deg=incidence, heading_deg=heading, look_deg=(heading + 90.0) % 360.0, notes=notes)


def print_scene_info(meta):
    k = doppler_k_s(meta["incidence_deg"])
    print(f"Scene start (UTC):    {meta['t_start']:%Y-%m-%d %H:%M:%S}")
    print(f"Scene end   (UTC):    {meta['t_end']:%Y-%m-%d %H:%M:%S}   ({meta['duration_s']:.0f} s long)")
    print(f"Matching time (mid):  {meta['t_mid']:%Y-%m-%d %H:%M:%S}")
    print(f"Pass: {meta['pass_dir']}, incidence {meta['incidence_deg']:.1f} deg, satellite heading "
          f"{meta['heading_deg']:.0f} deg, looking toward {meta['look_deg']:.0f} deg")
    state = "ON" if APPLY_DOPPLER_CORRECTION else "OFF"
    print(f"Doppler correction {state}: a ship moving 5 m/s (~10 kn) straight toward/away from "
          f"the satellite shifts ~{k * 5:.0f} m")
    for n in meta["notes"]:
        print("  note:", n)


# ===============================================================
# 4. LOAD + CLEAN AIS DATA
# ===============================================================
COLUMN_ALIASES = {
    "mmsi": ["mmsi"],
    "timestamp": ["timestamp", "# timestamp", "basedatetime", "time", "datetime", "time_utc"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude", "long"],
    "sog": ["sog", "speed", "speed_knots"],
    "cog": ["cog", "course"],
    "name": ["name", "vesselname", "vessel_name", "shipname"],
    "ship_type": ["ship type", "vesseltype", "ship_type"],
    "length": ["length", "ship_length", "loa"],
}


def clean_ais(df, time_format=None):
    """Standardise an AIS table: mmsi, timestamp, lat, lon, sog, cog (+ optional name, ship_type, length)."""
    df = df.copy()

    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    else:
        if time_format is None and re.match(r"^\d{2}/\d{2}/\d{4}", str(df["timestamp"].iloc[0])):
            time_format = "%d/%m/%Y %H:%M:%S"                      # Danish style: dd/mm/YYYY
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format=time_format, errors="coerce")

    for col in ("mmsi", "lat", "lon", "sog", "cog", "length"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    for col in ("name", "ship_type"):
        if col not in df:
            df[col] = ""

    df = df.dropna(subset=["mmsi", "timestamp", "lat", "lon"])
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]   # drops 91 / 181 = "not available"
    df.loc[df["sog"] >= 102.3, "sog"] = np.nan                           # 102.3 kn = "not available"
    df.loc[df["cog"] >= 360, "cog"] = np.nan                             # 360 deg = "not available"
    df.loc[df["length"] <= 0, "length"] = np.nan                         # 0 = unknown

    df["mmsi"] = df["mmsi"].astype("int64")
    df = df[df["mmsi"].between(200_000_000, 799_999_999)]                # real ships only
    df = df.drop_duplicates(subset=["mmsi", "timestamp"])
    return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)


def load_ais_csv(path, time_format=None):
    """Read an AIS CSV (NOAA, Danish DMA, or any file with similar column names)."""
    raw = pd.read_csv(path, low_memory=False)
    lower = {c.lower().strip(): c for c in raw.columns}
    out = pd.DataFrame()
    for std, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                out[std] = raw[lower[alias]]
                break
    missing = [c for c in ("mmsi", "timestamp", "lat", "lon") if c not in out]
    if missing:
        raise ValueError(f"AIS file is missing {missing}. Columns found: {list(raw.columns)}")
    return clean_ais(out, time_format)


def check_ais_freshness(ais, meta):
    """Upgrade 5: warn if the AIS data stops too soon after the pass."""
    if ais.empty:
        print("WARNING: no AIS reports at all.")
        return
    gap_min = (ais["timestamp"].max() - meta["t_mid"]).total_seconds() / 60
    if gap_min < FRESH_AIS_MIN:
        print(f"WARNING: AIS data ends {gap_min:.0f} min after the pass. Ships will be projected "
              f"forward instead of interpolated. Re-run later with fuller data and set "
              f"PREVIOUS_STATUS_CSV to see what changed.")
    else:
        print(f"AIS check: data runs {gap_min:.0f} min past the pass (good for interpolation).")


# ===============================================================
# 5. WHERE WAS EACH AIS SHIP AT THE SCENE TIME?
# ===============================================================
def estimate_positions(ais, t0):
    """One row per ship: estimated position at scene time t0 (interpolated or projected)."""
    lo, hi = t0 - pd.Timedelta(minutes=SEARCH_WINDOW_MIN), t0 + pd.Timedelta(minutes=SEARCH_WINDOW_MIN)
    ais = ais[(ais["timestamp"] >= lo) & (ais["timestamp"] <= hi)]
    rows = []

    for mmsi, g in ais.groupby("mmsi"):
        g = g.sort_values("timestamp")
        t = (g["timestamp"] - t0).dt.total_seconds().to_numpy()      # seconds vs the pass (negative = before)
        i = int(np.searchsorted(t, 0, side="right"))                 # first report AFTER the pass
        before = g.iloc[i - 1] if i > 0 else None
        after = g.iloc[i] if i < len(g) else None

        if before is not None and after is not None and (t[i] - t[i - 1]) <= MAX_INTERP_SPAN_MIN * 60:
            w = (0 - t[i - 1]) / (t[i] - t[i - 1])                   # straight line between the two reports
            lat = before["lat"] + w * (after["lat"] - before["lat"])
            lon = before["lon"] + w * (after["lon"] - before["lon"])
            gap = min(-t[i - 1], t[i])
            ref = before if -t[i - 1] <= t[i] else after
            method = "interpolated"
        else:
            cands = []
            if before is not None:
                cands.append((-t[i - 1], before, t[i - 1]))
            if after is not None:
                cands.append((t[i], after, t[i]))
            if not cands:
                continue
            gap, ref, t_ref = min(cands, key=lambda c: c[0])
            if gap > MAX_EXTRAP_MIN * 60:
                continue
            if np.isnan(ref["sog"]) or np.isnan(ref["cog"]):
                if gap > 60:
                    continue
                lat, lon = ref["lat"], ref["lon"]
            else:
                lat, lon = project(ref["lat"], ref["lon"], ref["sog"], ref["cog"], -t_ref)
            method = "extrapolated"

        rows.append(dict(
            mmsi=int(mmsi), lat=float(lat), lon=float(lon), sog_kn=ref["sog"], cog=ref["cog"],
            method=method, gap_s=float(gap), name=ref.get("name", ""), ship_type=ref.get("ship_type", ""),
            length=ref.get("length", np.nan)))

    cols = ["mmsi", "lat", "lon", "sog_kn", "cog", "method", "gap_s", "name", "ship_type", "length"]
    ships = pd.DataFrame(rows, columns=cols)
    ships["length"] = pd.to_numeric(ships["length"], errors="coerce")
    if ships.empty:
        ships["in_aoi"] = pd.Series(dtype=bool)
        return ships
    ships = ships[in_bbox(ships["lat"], ships["lon"], AOI_BBOX, pad_m=AOI_PAD_M)].reset_index(drop=True)
    ships["in_aoi"] = in_bbox(ships["lat"], ships["lon"], AOI_BBOX)
    return ships


# ===============================================================
# 6. DETECTIONS TABLE   (upgrade 4 needs the pixel counts)
# ===============================================================
def _finish_detections(df):
    df = df.reset_index(drop=True)
    df.insert(0, "det_id", range(1, len(df) + 1))
    df["det_key"] = df["lon"].map("{:.5f}".format) + "," + df["lat"].map("{:.5f}".format)
    df["pixel_count"] = pd.to_numeric(df["pixel_count"], errors="coerce")
    df["length_est_m"] = np.sqrt(SHIP_ASPECT_RATIO * df["pixel_count"]) * PIXEL_SIZE_M
    return df


def detections_from_features(features):
    """Build the detections table from Earth Engine `features` (keeps each detection's pixel count)."""
    rows = []
    for f in features:
        lon, lat = f["geometry"]["coordinates"][:2]
        props = f.get("properties") or {}
        rows.append(dict(lon=lon, lat=lat, pixel_count=props.get("count", np.nan)))
    return _finish_detections(pd.DataFrame(rows, columns=["lon", "lat", "pixel_count"]))


def detections_from_coords(coords):
    """Build the detections table from a plain list of [lon, lat] (no sizes, so no size check)."""
    df = pd.DataFrame(coords, columns=["lon", "lat"])
    df["pixel_count"] = np.nan
    return _finish_detections(df)


def detections_from_csv(path):
    """Read detections from a CSV with columns lon, lat (and optionally count = pixels per detection)."""
    raw = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in raw.columns}

    def pick(*names):
        return next((lower[n] for n in names if n in lower), None)

    lon_c, lat_c = pick("lon", "lng", "longitude", "long", "x"), pick("lat", "latitude", "y")
    cnt_c = pick("count", "pixel_count", "pixels", "npix")
    if lon_c is None or lat_c is None:
        raise ValueError(f"Detections file needs lon and lat columns. Columns found: {list(raw.columns)}")
    df = pd.DataFrame({
        "lon": pd.to_numeric(raw[lon_c], errors="coerce"),
        "lat": pd.to_numeric(raw[lat_c], errors="coerce"),
        "pixel_count": pd.to_numeric(raw[cnt_c], errors="coerce") if cnt_c else np.nan,
    }).dropna(subset=["lon", "lat"])
    if df.empty:
        raise ValueError("The detections file has no usable rows.")
    return _finish_detections(df)


def add_demo_pixel_counts(det, seed=11):
    """DEMO ONLY: invent pixel counts so the size check has something to work with."""
    rng = np.random.default_rng(seed)
    det = det.copy()
    det["pixel_count"] = np.clip(np.round(rng.lognormal(2.6, 0.8, len(det))), 2, 150)
    det["length_est_m"] = np.sqrt(SHIP_ASPECT_RATIO * det["pixel_count"]) * PIXEL_SIZE_M
    return det


# ===============================================================
# 7. MATCHING   (upgrades 2, 3 and 4 meet here)
# ===============================================================
def match_detections_to_ais(det, ships, meta, apply_doppler=True):
    """
    Returns (results, ais_unseen).
      results   : one row per detection, with status matched / possible_dark / likely_dark and a score
      ais_unseen: AIS ships inside the AOI that no detection was matched to
    """
    n_det, n_ais = len(det), len(ships)
    if n_det == 0:
        return pd.DataFrame(columns=RESULT_COLS), (ships[ships["in_aoi"]] if n_ais else ships)
    if n_ais == 0:
        res = det[["det_id", "det_key", "lon", "lat"]].copy()
        res["status"], res["score"] = "likely_dark", 0.0
        res["note"] = "no AIS ships at the time of the pass"
        res["sar_length_m"] = det["length_est_m"].to_numpy()
        return res.reindex(columns=RESULT_COLS), ships

    rows = np.arange(n_det)
    heading, look, inc = meta["heading_deg"], meta["look_deg"], meta["incidence_deg"]
    k = doppler_k_s(inc)

    # --- each AIS ship: where would SAR show it? (upgrade 2)
    sog = ships["sog_kn"].to_numpy(float)
    cog = ships["cog"].to_numpy(float)
    has_motion = ~np.isnan(sog) & ~np.isnan(cog)
    v = np.where(np.isnan(sog), DEFAULT_SOG_KN, sog) * KNOT_TO_MS                  # m/s
    shift = np.where(has_motion, doppler_shift_m(np.nan_to_num(sog), np.nan_to_num(cog), look, inc), 0.0)
    shift_used = shift if apply_doppler else np.zeros(n_ais)
    img_lat, img_lon = apply_azimuth_shift(ships["lat"].to_numpy(float), ships["lon"].to_numpy(float),
                                           shift_used, heading)

    # --- how uncertain is that predicted position? (upgrade 3, "uncertainty budget")
    l_ais = ships["length"].to_numpy(float)
    l_ais = np.where(l_ais > 0, l_ais, np.nan)
    sig_len = LENGTH_SIGMA_FRAC * np.where(np.isnan(l_ais), DEFAULT_LENGTH_M, l_ais)
    sig_scene = v * (meta["duration_s"] / 2) / math.sqrt(3)                        # leftover timing error (upgrade 1)
    err_frac = np.where(ships["method"].to_numpy() == "extrapolated", EXTRAP_ERR_FRAC, INTERP_ERR_FRAC)
    sig_track = err_frac * v * ships["gap_s"].to_numpy(float)
    if apply_doppler:
        sig_dopp = np.where(has_motion, DOPPLER_ERR_FRAC * np.abs(shift), 0.7 * k * v)
    else:
        sig_dopp = 0.7 * k * v                                                     # shift ignored: allow for it
    sigma = np.sqrt(SIGMA_BASE_M ** 2 + sig_len ** 2 + sig_scene ** 2 + sig_track ** 2 + sig_dopp ** 2)

    # --- distances and size check (upgrade 4)
    dist = haversine_m(det["lat"].to_numpy(float)[:, None], det["lon"].to_numpy(float)[:, None],
                       img_lat[None, :], img_lon[None, :])
    ratio = det["length_est_m"].to_numpy(float)[:, None] / l_ais[None, :]          # NaN if either is unknown
    known = ~np.isnan(ratio)
    mismatch = known & ((ratio > SIZE_RATIO_LIMIT) | (ratio < 1.0 / SIZE_RATIO_LIMIT))
    size_factor = np.where(mismatch, SIZE_MISMATCH_PENALTY, 1.0)

    # --- score: how well does ship j explain detection i, compared with all the alternatives?
    like = size_factor * np.exp(-0.5 * (dist / sigma[None, :]) ** 2) / (2 * np.pi * sigma[None, :] ** 2)
    clutter = CLUTTER_PER_KM2 / 1e6                                                # unrelated ships per m2
    score = like / (clutter + like.sum(axis=1, keepdims=True))
    score = np.where(mismatch, np.minimum(score, SIZE_MISMATCH_MAX_SCORE), score)   # size mismatch: needs a human look

    # --- one-to-one pairing. A pair is only worth making if it explains the detection BETTER than
    #     "this is a dark ship" (benefit > 0). Among all allowed pairings we take the one with the
    #     greatest total benefit, so a barely-plausible pair can never push out a much better one.
    odds = RADAR_DETECT_PROB / (1.0 - RADAR_DETECT_PROB)
    benefit = np.log(np.maximum(odds * like / clutter, 1e-300))
    r, c = linear_sum_assignment(-np.maximum(benefit, 0.0))
    ok = (benefit[r, c] > 0) & (score[r, c] >= MIN_SCORE)
    partner = np.full(n_det, -1)
    partner[r[ok]] = c[ok]

    best_j, best_s = score.argmax(axis=1), score.max(axis=1)
    status = np.full(n_det, "likely_dark", dtype=object)
    note = np.full(n_det, "no plausible AIS ship nearby", dtype=object)
    cand = np.full(n_det, -1)
    taken = set(partner[partner >= 0].tolist())
    holder = {int(partner[k]): k for k in range(n_det) if partner[k] >= 0}      # ship index -> detection index
    det_lat_a, det_lon_a = det["lat"].to_numpy(float), det["lon"].to_numpy(float)
    det_ids = det["det_id"].to_numpy()
    for i in range(n_det):
        if partner[i] >= 0:
            cand[i] = partner[i]
            if score[i, partner[i]] >= MATCH_SCORE:
                status[i], note[i] = "matched", ""
            elif mismatch[i, partner[i]]:
                status[i], note[i] = "possible_dark", "size mismatch: SAR and AIS lengths disagree"
            else:
                status[i], note[i] = "possible_dark", "weak match: score below the match threshold"
        elif best_s[i] >= MIN_SCORE:
            cand[i], status[i] = best_j[i], "possible_dark"
            if best_j[i] in taken:
                k_ = holder[int(best_j[i])]
                gap_m = float(haversine_m(det_lat_a[i], det_lon_a[i], det_lat_a[k_], det_lon_a[k_]))
                note[i] = (f"AIS ship already matched to detection #{det_ids[k_]}, {gap_m:.0f} m away "
                           f"(the same ship split in two, or a second vessel next to it)")
            else:
                note[i] = "AIS candidate is plausible but not chosen"
            if mismatch[i, best_j[i]]:
                note[i] = "size mismatch: SAR and AIS lengths disagree"

    has = cand >= 0
    cj = np.clip(cand, 0, None)
    cdf = ships.reset_index(drop=True).reindex(cand)                  # -1 -> empty row

    def pair(mat):
        return np.where(has, mat[rows, cj], np.nan)

    def per_ship(vec):
        return np.where(has, np.asarray(vec, float)[cj], np.nan)

    size_check = np.where(~has, "", np.where(has & mismatch[rows, cj], "mismatch",
                                             np.where(has & known[rows, cj], "ok", "unknown")))
    res = pd.DataFrame({
        "det_id": det["det_id"].to_numpy(), "det_key": det["det_key"].to_numpy(),
        "lon": det["lon"].to_numpy(), "lat": det["lat"].to_numpy(),
        "status": status, "score": np.where(has, score[rows, cj], best_s), "note": note,
        "mmsi": cdf["mmsi"].to_numpy(), "name": cdf["name"].to_numpy(), "ship_type": cdf["ship_type"].to_numpy(),
        "ais_true_lon": cdf["lon"].to_numpy(), "ais_true_lat": cdf["lat"].to_numpy(),
        "ais_img_lon": per_ship(img_lon), "ais_img_lat": per_ship(img_lat),
        "distance_m": pair(dist), "sigma_m": per_ship(sigma), "z": pair(dist / sigma[None, :]),
        "doppler_shift_m": per_ship(shift_used),
        "sar_length_m": det["length_est_m"].to_numpy(), "ais_length_m": per_ship(l_ais),
        "size_check": size_check, "sog_kn": cdf["sog_kn"].to_numpy(), "cog": cdf["cog"].to_numpy(),
        "method": cdf["method"].to_numpy(), "gap_s": cdf["gap_s"].to_numpy(),
        "nearest_ais_m": dist.min(axis=1),
    })[RESULT_COLS]
    res["mmsi"] = res["mmsi"].astype("Int64")

    matched_cols = {int(partner[i]) for i in range(n_det) if status[i] == "matched"}
    unseen = ships["in_aoi"].to_numpy(bool) & ~np.isin(np.arange(n_ais), list(matched_cols))
    ais_unseen = ships[unseen].copy()
    ais_unseen["nearest_det_m"] = dist.min(axis=0)[unseen]
    return res, ais_unseen.reset_index(drop=True)


def compare_doppler(det, ships, meta):
    """
    Upgrade 2 safety check: match with and without the Doppler correction and compare.
    NOTE: more matches is NOT better. A looser matcher can 'explain' a dark ship with an unrelated
    AIS ship. What should improve is how CLOSE the matched pairs are, and the disagreements below
    are the pairs worth looking at on the map.
    """
    on, _ = match_detections_to_ais(det, ships, meta, apply_doppler=True)
    off, _ = match_detections_to_ais(det, ships, meta, apply_doppler=False)
    m_on, m_off = on[on["status"] == "matched"], off[off["status"] == "matched"]
    print("Doppler correction check (a sanity test on YOUR data):")
    for label, m in (("with correction   ", m_on), ("without correction", m_off)):
        if len(m):
            print(f"   {label}: {len(m):3d} matched, median distance {m['distance_m'].median():.0f} m")
        else:
            print(f"   {label}: no matches")

    both = m_on.merge(m_off, on="det_id", suffixes=("_on", "_off"))
    if len(both):
        d_on, d_off = both["distance_m_on"].median(), both["distance_m_off"].median()
        print(f"   On the {len(both)} detections matched both ways: median distance {d_on:.0f} m with, "
              f"{d_off:.0f} m without.")
        if d_on <= d_off:
            print("   -> the correction is tightening the matches (good sign).")
        else:
            print("   -> WARNING: the correction makes matches LOOSER. Check the pass direction and incidence\n"
                  "      angle above, or switch APPLY_DOPPLER_CORRECTION off.")
    a = on.set_index("det_id")
    b = off.set_index("det_id")
    diff = [i for i in a.index
            if (a.loc[i, "status"] == "matched" or b.loc[i, "status"] == "matched")
            and (a.loc[i, "status"] != b.loc[i, "status"] or a.loc[i, "mmsi"] != b.loc[i, "mmsi"])]
    print(f"   Detections whose result changes with the correction: {len(diff)}"
          + (f"  (ids: {diff[:12]}{'...' if len(diff) > 12 else ''})  <- check these on the map" if diff else ""))
    print()


# ===============================================================
# 8. OPERATIONAL OUTPUTS   (upgrade 5)
# ===============================================================
def add_operational_columns(results, meta, processed_at):
    """Pass time, processing time, and how far a dark ship could have travelled since."""
    res = results.copy()
    age_s = max(0.0, (processed_at - meta["t_mid"]).total_seconds())
    res["acq_time_utc"] = meta["t_mid"].strftime("%Y-%m-%d %H:%M:%S")
    res["processed_utc"] = processed_at.strftime("%Y-%m-%d %H:%M:%S")
    dark = (res["status"] != "matched").to_numpy()
    res["age_min"] = np.where(dark, round(age_s / 60, 1), np.nan)
    res["typical_radius_km"] = np.where(dark, age_s * TYPICAL_SPEED_KN * KNOT_TO_MS / 1000, np.nan)
    res["max_radius_km"] = np.where(dark, age_s * MAX_SPEED_KN * KNOT_TO_MS / 1000, np.nan)
    return res


def save_status(results, meta, path=None):
    """Save each detection's status so a later run can be compared with this one."""
    run_utc = pd.Timestamp.now(tz="UTC")
    path = path or f"status_{meta['t_mid']:%Y%m%dT%H%M%S}_run{run_utc:%Y%m%dT%H%M%S}.csv"
    out = results[["det_key", "status", "mmsi", "score"]].copy()
    out["run_utc"] = run_utc.isoformat()
    out.to_csv(path, index=False)
    return path


def compare_with_previous(results, previous_csv):
    """What changed since an earlier run (for example, because late AIS arrived)?"""
    if not previous_csv:
        return
    if not os.path.exists(previous_csv):
        print(f"(previous status file not found: {previous_csv})")
        return
    prev = pd.read_csv(previous_csv)
    merged = results[["det_key", "status"]].merge(
        prev[["det_key", "status"]], on="det_key", how="outer", suffixes=("_now", "_before"))
    print(f"Compared with the earlier run ({previous_csv}):")
    changed = merged[(merged["status_now"] != merged["status_before"])
                     & merged["status_now"].notna() & merged["status_before"].notna()]
    if changed.empty:
        print("   no detection changed status.")
    else:
        for (old, new), n in changed.groupby(["status_before", "status_now"]).size().items():
            print(f"   {old:>13} -> {new:<13}: {n}")
    n_new = int(merged["status_before"].isna().sum())
    n_gone = int(merged["status_now"].isna().sum())
    if n_new or n_gone:
        print(f"   ({n_new} detections are new, {n_gone} are no longer present)")


def print_summary(results, ships, ais_unseen, meta, processed_at):
    n = len(results)
    counts = results["status"].value_counts() if n else pd.Series(dtype=int)
    n_match = int(counts.get("matched", 0))
    n_possible = int(counts.get("possible_dark", 0))
    n_likely = int(counts.get("likely_dark", 0))
    n_ais = int(ships["in_aoi"].sum()) if len(ships) else 0
    matched_mmsi = results.loc[results["status"] == "matched", "mmsi"].dropna()
    n_match_in = int(matched_mmsi.isin(ships.loc[ships["in_aoi"], "mmsi"]).sum()) if len(ships) else 0
    outside = n_match - n_match_in
    n_unmatched_ais = len(ais_unseen)

    print("=" * 66)
    print("RADAR LIST (everything SAR detected)")
    print(f"  Detections:                              {n}")
    print(f"    matched (score >= {MATCH_SCORE}):                {n_match}")
    print(f"    possible dark (weak/ambiguous AIS):    {n_possible}")
    if n_possible:
        pdk = results.loc[results["status"] == "possible_dark", "note"].fillna("").astype(str)
        k_taken = int(pdk.str.startswith("AIS ship already matched").sum())
        k_weak = int(pdk.str.startswith("weak match").sum())
        k_size = int(pdk.str.startswith("size mismatch").sum())
        parts = [f"{c} {t}" for c, t in ((k_taken, "next to an AIS ship already matched to another detection"),
                                         (k_weak, "weak position match"), (k_size, "size mismatch"),
                                         (n_possible - k_taken - k_weak - k_size, "other")) if c]
        print("        why: " + "; ".join(parts))
    print(f"    likely dark (no plausible AIS):        {n_likely}")
    print("AIS LIST (ships broadcasting inside the AOI at the pass)")
    print(f"  AIS ships:                               {n_ais}")
    print(f"    matched to a detection:                {n_match_in}")
    print(f"    with no radar partner:                 {n_unmatched_ais}")
    print("BOTH LISTS")
    note = (f"   ({outside} {'sits' if outside == 1 else 'sit'} just outside the AOI box, so "
            f"{'it is' if outside == 1 else 'they are'} not in the AIS count)") if outside else ""
    print(f"  Ships found by both (matched):           {n_match}{note}")
    if n:
        print(f"  Detections that have AIS:                {n_match / n:.0%}   ({n_match} of {n})")
    if n_ais:
        print(f"  AIS ships found by radar (recall):       {n_match_in / n_ais:.0%}   ({n_match_in} of {n_ais})")
    if n_match:
        sc = results.loc[results["status"] == "matched", "size_check"].value_counts()
        print(f"  Size check among matches:                ok {sc.get('ok', 0)}, "
              f"mismatch {sc.get('mismatch', 0)}, unknown {sc.get('unknown', 0)}")
    print("=" * 66)

    if n_unmatched_ais:
        d = ais_unseen["nearest_det_m"].to_numpy(float)
        near, mid, far = int((d <= 500).sum()), int(((d > 500) & (d <= 2000)).sum()), int((d > 2000).sum())
        print(f"The {n_unmatched_ais} AIS ships with no radar partner: {near} lie within 500 m of a detection "
              f"(size mismatch, crowding or a swapped pair),")
        print(f"   {mid} within 2 km, and {far} farther than 2 km (the radar saw nothing near them).")
    if n_possible + n_likely:
        age_min = max(0.0, (processed_at - meta["t_mid"]).total_seconds() / 60)
        typ = age_min * 60 * TYPICAL_SPEED_KN * KNOT_TO_MS / 1000
        top = age_min * 60 * MAX_SPEED_KN * KNOT_TO_MS / 1000
        basis = "assumed" if PROCESSED_AT != "now" else "real"
        print(f"Operational note: results are {age_min:.0f} min old ({basis} clock). A dark candidate could")
        print(f"now be up to ~{typ:.1f} km ({TYPICAL_SPEED_KN} kn) to ~{top:.1f} km ({MAX_SPEED_KN} kn) "
              f"from where it was seen.")
    print("A dark candidate is only a CANDIDATE: it may be a small boat, a navy ship, an AIS gap,")
    print("a false alarm, or a fixed structure. AIS ships near the coast can be 'missed' simply")
    print("because the land mask hides them.")


# ===============================================================
# 9. MAP
# ===============================================================
STATUS_COLOURS = {"matched": "lime", "possible_dark": "orange", "likely_dark": "red"}


def _popup(r):
    lines = [f"<b>Detection #{r.det_id}</b>: {r.status} (score {r.score:.2f})"]
    if pd.notna(r.mmsi):
        lines.append(f"AIS: MMSI {int(r.mmsi)} {html.escape(str(r.name or ''))}")
        lines.append(f"distance {r.distance_m:.0f} m, uncertainty {r.sigma_m:.0f} m (z = {r.z:.1f})")
        lines.append(f"Doppler shift applied: {r.doppler_shift_m:.0f} m, size check: {r.size_check}")
    if r.note:
        lines.append(html.escape(str(r.note)))
    return "<br>".join(lines)


def make_map(results, ais_unseen, aoi_bbox=AOI_BBOX):
    import folium
    west, south, east, north = aoi_bbox
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=13, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street map", overlay=False, show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False, show=True).add_to(m)
    folium.Rectangle([[south, west], [north, east]], color="cyan", weight=2, fill=False,
                     tooltip="AOI").add_to(m)

    groups = {
        "matched": folium.FeatureGroup(name="Matched (green = SAR, blue = where AIS predicts it)"),
        "possible_dark": folium.FeatureGroup(name="Possible dark (orange)"),
        "likely_dark": folium.FeatureGroup(name="Likely dark (red)"),
    }
    g_unseen = folium.FeatureGroup(name="AIS ships not matched (purple)")
    g_shift = folium.FeatureGroup(name="Doppler shift: AIS true position -> where SAR shows it", show=False)
    g_search = folium.FeatureGroup(name="Search area for dark candidates (grows with time)", show=False)

    for r in results.itertuples():
        g = groups[r.status]
        col = STATUS_COLOURS[r.status]
        folium.CircleMarker([r.lat, r.lon], radius=7 if r.status == "matched" else 8, color=col,
                            fill=True, fill_opacity=0.9,
                            popup=folium.Popup(_popup(r), max_width=280)).add_to(g)
        if pd.notna(r.mmsi):
            folium.PolyLine([[r.lat, r.lon], [r.ais_img_lat, r.ais_img_lon]], color="white",
                            weight=1.5, dash_array="4").add_to(g)
            folium.CircleMarker([r.ais_img_lat, r.ais_img_lon], radius=4, color="deepskyblue",
                                fill=True, fill_opacity=0.9, tooltip=f"AIS {int(r.mmsi)}").add_to(g)
            folium.PolyLine([[r.ais_true_lat, r.ais_true_lon], [r.ais_img_lat, r.ais_img_lon]],
                            color="gray", weight=2).add_to(g_shift)
            folium.CircleMarker([r.ais_true_lat, r.ais_true_lon], radius=3, color="gray",
                                fill=True, tooltip=f"AIS true position {int(r.mmsi)}").add_to(g_shift)
        if r.status != "matched" and "max_radius_km" in results.columns and pd.notna(r.max_radius_km):
            if r.max_radius_km <= MAX_DRAW_RADIUS_KM:
                folium.Circle([r.lat, r.lon], radius=r.max_radius_km * 1000, color="red", weight=1,
                              dash_array="6", fill=False,
                              tooltip=f"#{r.det_id}: up to {r.max_radius_km:.1f} km at {MAX_SPEED_KN} kn"
                              ).add_to(g_search)
                folium.Circle([r.lat, r.lon], radius=r.typical_radius_km * 1000, color="orange", weight=1,
                              dash_array="2", fill=False,
                              tooltip=f"#{r.det_id}: ~{r.typical_radius_km:.1f} km at {TYPICAL_SPEED_KN} kn"
                              ).add_to(g_search)

    for r in ais_unseen.itertuples():
        info = (f"<b>AIS only</b><br>MMSI {r.mmsi} {html.escape(str(r.name or ''))}<br>"
                f"{r.sog_kn} kn<br>nearest detection {r.nearest_det_m:.0f} m away")
        folium.CircleMarker([r.lat, r.lon], radius=6, color="mediumpurple", fill=True, fill_opacity=0.9,
                            popup=folium.Popup(info, max_width=250)).add_to(g_unseen)

    for g in (*groups.values(), g_unseen, g_shift, g_search):
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


# ===============================================================
# 10. FAKE AIS DATA (only for testing the code)
# ===============================================================
def make_demo_ais(det, meta, seed=7, return_truth=False):
    """
    FAKE AIS, built from YOUR detections:
      * DEMO_AIS_SHARE of the detections get an AIS ship; the rest are dark ON PURPOSE.
      * DEMO_EXTRA_AIS_SHIPS extra AIS ships are added that the radar never saw (default 0).
      * DEMO_SIZE_MISMATCH_SHARE of the AIS ships get a wrong length (default 0).
    It is built with the same physics the matcher assumes, so it only proves the code runs.
    It says NOTHING about how well the method works on real data.
    With return_truth=True it also returns the answer key (who is really who).
    """
    rng = np.random.default_rng(seed)
    t0, half = meta["t_mid"], meta["duration_s"] / 2
    west, south, east, north = AOI_BBOX
    rows, state = [], {"mmsi": 563000000}
    truth = dict(planted={}, dark=[], size_mismatch=[], extras=[])    # planted: {det_id: mmsi}

    def add_track(lat0, lon0, sog, cog, length):
        state["mmsi"] += 1
        pattern = str(rng.choice(["both", "before", "after"]))
        offsets = {"both": [-14, -8, -3, 4, 9], "before": [-9, -5, -2], "after": [3, 6, 11]}[pattern]
        for minutes in offsets:
            t = minutes * 60 + rng.uniform(-20, 20)
            lat, lon = project(lat0, lon0, sog, cog, t)
            rows.append(dict(
                mmsi=state["mmsi"], timestamp=t0 + pd.Timedelta(seconds=t),
                lat=float(lat + rng.normal(0, 2e-5)), lon=float(lon + rng.normal(0, 2e-5)),
                sog=max(0.0, sog + rng.normal(0, 0.2)), cog=(cog + rng.normal(0, 2)) % 360,
                name=f"DEMO SHIP {state['mmsi']}", length=length))

    # (1) AIS ships that belong to real detections
    for d in det.itertuples():
        if rng.random() >= DEMO_AIS_SHARE:
            truth["dark"].append(int(d.det_id))                   # left dark on purpose
            continue
        sog, cog = rng.uniform(0.5, 16), rng.uniform(0, 360)
        dt = rng.uniform(-half, half)                             # when inside the scene the radar saw it
        shift = float(doppler_shift_m(sog, cog, meta["look_deg"], meta["incidence_deg"]))
        lat1, lon1 = apply_azimuth_shift(d.lat, d.lon, -shift, meta["heading_deg"])      # true spot when imaged
        lat0, lon0 = project(lat1, lon1, sog, cog, -dt)                                  # true spot at mid time
        lat0, lon0 = move_m(lat0, lon0, rng.normal(0, 20), rng.normal(0, 20))
        if pd.notna(d.length_est_m):
            length = d.length_est_m * rng.uniform(0.7, 1.4)
            if rng.random() < DEMO_SIZE_MISMATCH_SHARE:           # a deliberate size mismatch
                length = d.length_est_m / 5 if rng.random() < 0.5 else d.length_est_m * 5
                truth["size_mismatch"].append(int(d.det_id))
        else:
            length = float(rng.lognormal(4.5, 0.6))
        add_track(float(lat0), float(lon0), sog, cog, round(float(length)))
        truth["planted"][int(d.det_id)] = state["mmsi"]

    # (2) optional AIS ships that the radar never saw
    for _ in range(DEMO_EXTRA_AIS_SHIPS):
        add_track(rng.uniform(south, north), rng.uniform(west, east), rng.uniform(0, 14),
                  rng.uniform(0, 360), round(float(rng.lognormal(4.5, 0.6))))
        truth["extras"].append(state["mmsi"])

    out = pd.DataFrame(rows, columns=["mmsi", "timestamp", "lat", "lon", "sog", "cog", "name", "length"])
    return (out, truth) if return_truth else out


def print_demo_answer_key(results, truth):
    """Demo only: compare what the matcher decided with what we built into the fake world."""
    R = results.set_index("det_id")
    cnt = dict(correct=0, wrong=0, flagged=0, lost=0)
    for det_id, mmsi in truth["planted"].items():
        r = R.loc[det_id]
        got = int(r["mmsi"]) if pd.notna(r["mmsi"]) else None
        if r["status"] == "matched":
            cnt["correct" if got == mmsi else "wrong"] += 1
        elif got == mmsi:
            cnt["flagged"] += 1
        else:
            cnt["lost"] += 1
    dark = truth["dark"]
    wrongly = sum(R.loc[i, "status"] == "matched" for i in dark)
    print("-" * 66)
    print("DEMO ANSWER KEY (only possible because the AIS is fake)")
    print(f"  Built into this fake world: {len(truth['planted'])} detections given an AIS ship, "
          f"{len(dark)} left dark on purpose,")
    print(f"  {len(truth['extras'])} extra AIS ships the radar never saw, "
          f"{len(truth['size_mismatch'])} AIS ships given a wrong length.")
    print(f"  Result for the {len(truth['planted'])} ships that have AIS:")
    print(f"     {cnt['correct']:3d}  matched to the correct ship")
    print(f"     {cnt['wrong']:3d}  matched to the WRONG ship")
    print(f"     {cnt['flagged']:3d}  flagged possible_dark but pointing at the correct ship "
          f"(weak score, crowding or size mismatch)")
    print(f"     {cnt['lost']:3d}  lost (no candidate, or a different one)")
    print(f"  Result for the {len(dark)} dark ships:")
    print(f"     {len(dark) - wrongly:3d}  correctly left without a match")
    print(f"     {wrongly:3d}  wrongly 'explained' by an unrelated AIS ship")
    print("-" * 66)


# ===============================================================
# 11. RUN IT
# ===============================================================
SAMPLE_COORDS = [[103.80300, 1.24043], [103.80397, 1.24319], [103.80589, 1.23362],
                 [103.80643, 1.23890], [103.81017, 1.24469]]


def run(features=None, coords=None, scene=None, aoi=None, detections_csv=None):
    """
    Everything the script does, in one function (so it cannot overwrite variables in your notebook).
      features : Earth Engine detections (list of GeoJSON features; their 'count' enables the size check)
      coords   : fallback list of [lon, lat] if you have no `features`
      detections_csv : path of a CSV with lon, lat (and optionally count); takes priority over the above
      scene    : the Earth Engine image the detections came from (for time, pass direction, incidence)
      aoi      : the Earth Engine geometry of your area (for the incidence angle)
    Returns a dict: det, meta, ais, ships, results, ais_unseen, map, truth (demo only)
    """
    # --- Detections: a CSV file if given, else Earth Engine `features` (they carry pixel counts), else coords
    if detections_csv:
        det = detections_from_csv(detections_csv)
        print(f"Detections read from {detections_csv}: {len(det)}")
    elif isinstance(features, list) and features and isinstance(features[0], dict) and "geometry" in features[0]:
        det = detections_from_features(features)
        print(f"Detections read from `features`: {len(det)}")
        if det["pixel_count"].isna().all():
            print("  note: no 'count' property found in `features`. Properties seen:",
                  list((features[0].get("properties") or {}).keys()))
    else:
        det = detections_from_coords(coords if coords is not None else SAMPLE_COORDS)
        print(f"Detections read from `coords`: {len(det)}")

    # --- Scene: time, pass direction, incidence angle
    meta = read_scene_metadata(scene, aoi)
    print_scene_info(meta)

    # --- AIS
    if USE_DEMO_AIS:
        print("\n!!! DEMO MODE: FAKE AIS data. The numbers below mean nothing. !!!\n")
        if det["pixel_count"].isna().all():
            det = add_demo_pixel_counts(det)
            print("(demo: pixel counts were invented so the size check can be shown)")
        ais, truth = make_demo_ais(det, meta, return_truth=True)
    else:
        if det["pixel_count"].isna().all():
            print("Note: the detections have no pixel counts, so the size check is off.")
        ais, truth = load_ais_csv(AIS_CSV_PATH, AIS_TIME_FORMAT), None
    if SIMULATE_AIS_AVAILABLE_UNTIL_MIN is not None:
        cutoff = meta["t_mid"] + pd.Timedelta(minutes=SIMULATE_AIS_AVAILABLE_UNTIL_MIN)
        ais = ais[ais["timestamp"] <= cutoff]
        print(f"Simulating an early run: AIS reports after {cutoff:%H:%M:%S} UTC are hidden.")
    check_ais_freshness(ais, meta)

    # --- Where was each ship at the scene's middle time, and who matches whom?
    ships = estimate_positions(ais, meta["t_mid"])
    results, ais_unseen = match_detections_to_ais(det, ships, meta, apply_doppler=APPLY_DOPPLER_CORRECTION)
    processed_at = (pd.Timestamp.now(tz="UTC") if PROCESSED_AT == "now"
                    else meta["t_mid"] + pd.Timedelta(minutes=ASSUMED_DELAY_MIN))
    results = add_operational_columns(results, meta, processed_at)

    print_summary(results, ships, ais_unseen, meta, processed_at)
    if DIAGNOSE_DOPPLER and APPLY_DOPPLER_CORRECTION and len(ships):
        compare_doppler(det, ships, meta)
    if truth is not None:
        print_demo_answer_key(results, truth)
    compare_with_previous(results, PREVIOUS_STATUS_CSV)

    # --- Save everything
    status_file = save_status(results, meta)
    results.to_csv("results.csv", index=False)
    results[results["status"] == "matched"].to_csv("matched.csv", index=False)
    results[results["status"] != "matched"].to_csv("dark_candidates.csv", index=False)
    ais_unseen.to_csv("ais_not_detected.csv", index=False)
    print(f"Saved: results.csv, matched.csv, dark_candidates.csv, ais_not_detected.csv, {status_file}")
    print("To compare a later re-run with this one, set PREVIOUS_STATUS_CSV =", repr(status_file))

    # --- Map
    fmap = make_map(results, ais_unseen)
    fmap.save("ais_match_map.html")
    try:
        from IPython.display import display
        display(fmap)
    except ImportError:
        print("Map saved to ais_match_map.html")
    return dict(det=det, meta=meta, ais=ais, ships=ships, results=results, ais_unseen=ais_unseen,
                map=fmap, truth=truth)


if __name__ == "__main__":
    # In a notebook (%run -i) this picks up `features`, `coords`, `first` and `AOI` from your detection cells.
    _g = globals()
    v2 = run(features=_g.get("features"), coords=_g.get("coords"), scene=_g.get("first"), aoi=_g.get("AOI"),
             detections_csv=DETECTIONS_CSV_PATH)
