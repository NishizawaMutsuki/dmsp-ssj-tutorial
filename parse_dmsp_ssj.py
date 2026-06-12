#!/usr/bin/env python3
"""
src/parse_dmsp_ssj.py

Parser for DMSP SSJ "Simple Format" binary files (NCEI distribution).

Reference: AFRL-RV-PS-TR-2014-0174, Table 19 (SSJ data file format).

File layout
-----------
- 16-bit unsigned big-endian integers
- Each minute = 2640 words = 5280 bytes:
    Words  1-15  : header (date / position / MLT for first second of minute)
    Words 16-58  : first  per-second block (43 words)
    Words 59-101 : second per-second block
    ...
    Words 2553-2595: 60th per-second block
    Word  2596   : 1 → word 18 (sec-of-min) is in milliseconds (SSJ5)
    Words 2597-2640: zero fill
- Each per-second block (43 words):
    word  1 : hour of day
    word  2 : minute of hour
    word  3 : second of minute (or millisecond if SSJ5 + flag)
    words  4-23 : 20 electron channel raw values
    words 24-43 : 20 ion channel raw values
- Channel ordering inside each block is NOT 1..20.  The raw order is:
    pos:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    chan: 4  3  2  1  8  7  6  5 12 11 10  9 16 15 14 13 20 19 18 17
- For SSJ5 (F16-F18) the slot for channel 11 holds status word 1 (electrons)
  / status word 2 (ions) instead of a channel reading — so we get 19 real
  electron and 19 real ion channels.

Decompression: raw 16-bit value I encodes 5-bit mantissa X and 4-bit
exponent Y as
    X = I mod 32
    Y = (I - X) / 32
    counts = (X + 32) * 2**Y - 33
Special raw values: 0 → missing (-1), 1 → 0 counts.

Differential energy flux:  JE_i = (counts_i / (G_i * Δt)) * E_i
where Δt = 0.05 s for SSJ5, G_i is the per-channel geometric factor
(eV·cm²·sr) from Tables 23-25 (electrons) / 26-28 (ions).
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

WORDS_PER_MINUTE = 2640
BYTES_PER_MINUTE = WORDS_PER_MINUTE * 2
WORDS_PER_SECOND = 43
HEADER_WORDS = 15

# Energies (eV) and channel widths (eV) for channels 1..20
CHANNEL_ENERGY_EV = np.array([
    30000, 20400, 13900, 9450, 6460, 4400, 3000, 2040, 1392, 949,
    949, 646, 440, 300, 204, 139, 95, 65, 44, 30,
], dtype=np.float64)

CHANNEL_DELTA_EV = np.array([
    9600, 8050, 5475, 3720, 2525, 1730, 1180, 804, 545.5, 373,
    373, 254.5, 173, 118, 80.5, 54.5, 37, 25.5, 17.5, 14,
], dtype=np.float64)

# File-order index (1..20) → channel number
FILE_ORDER_TO_CHANNEL = np.array(
    [4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17],
    dtype=np.int64,
)
# Inverse permutation: channel i → its position (0-based) in the 20-word slot
CHANNEL_TO_FILE_ORDER = np.argsort(FILE_ORDER_TO_CHANNEL)

# SSJ5 status word lives where channel 11 would normally sit
STATUS_FILE_POS = int(np.where(FILE_ORDER_TO_CHANNEL == 11)[0][0])  # = 9

# Geometric factors (eV·cm²·sr) for electrons and ions, channels 1..20.
# Tables 24-25 (electrons) and 27-28 (ions).  Channel 11 marked '-' (NaN)
# only for SSJ5 sensors (F16-F20) where it is replaced with status word.
ELECTRON_GEOM_FACTORS = {
    "F15": np.array([
        0.3124, 0.2528, 0.2046, 0.1657, 0.1341, 0.1086, 0.0879, 0.07116,
        0.0576, 0.04662, 0.05072, 0.03647, 0.02488, 0.01733, 0.01205,
        0.006807, 0.003672, 0.001928, 0.0009189, 0.000356,
    ], dtype=np.float64),
    "F16": np.array([
        1.781, 1.477, 1.188, 0.935, 0.722, 0.551, 0.416, 0.306, 0.225, 0.166,
        np.nan, 0.123, 0.0876, 0.0613, 0.0429, 0.0289, 0.0182, 0.0113,
        0.00621, 0.00307,
    ], dtype=np.float64),
    "F17": np.array([
        1.044, 0.808, 0.602, 0.458, 0.349, 0.262, 0.191, 0.142, 0.103, 0.0727,
        np.nan, 0.0541, 0.0394, 0.0276, 0.0188, 0.0134, 0.00901, 0.00645,
        0.00445, 0.00294,
    ], dtype=np.float64),
    "F18": np.array([
        0.725, 0.534, 0.412, 0.315, 0.266, 0.199, 0.147, 0.107, 0.0803, 0.0562,
        np.nan, 0.041, 0.0296, 0.0203, 0.014, 0.0104, 0.00708, 0.00562,
        0.00386, 0.00239,
    ], dtype=np.float64),
}

ION_GEOM_FACTORS = {
    "F15": np.array([
        0.9016, 0.6392, 0.435, 0.296, 0.2014, 0.1371, 0.09328, 0.06347,
        0.0432, 0.02939, 0.9439, 0.6443, 0.4398, 0.3002, 0.2049, 0.1398,
        0.09454, 0.06515, 0.04447, 0.03035,
    ], dtype=np.float64),
    "F16": np.array([
        13.3, 8.51, 5.43, 3.43, 2.19, 1.4, 0.903, 0.575, 0.368, 0.244,
        np.nan, 0.162, 0.105, 0.0718, 0.0505, 0.0342, 0.023, 0.0157,
        0.00745, 0.00394,
    ], dtype=np.float64),
    "F17": np.array([
        5.71, 3.81, 2.54, 1.7, 1.13, 0.715, 0.47, 0.306, 0.199, 0.122,
        np.nan, 0.0899, 0.0581, 0.0307, 0.017, 0.0101, 0.005, 0.00302,
        0.00158, 0.000911,
    ], dtype=np.float64),
    "F18": np.array([
        10.6, 6.9, 4.51, 2.81, 1.82, 1.19, 0.774, 0.485, 0.296, 0.208,
        np.nan, 0.15, 0.105, 0.0725, 0.0448, 0.0324, 0.0215, 0.0113,
        0.00448, 0.00182,
    ], dtype=np.float64),
}

# Accumulation time (s) — SSJ5 = 0.05, SSJ4 = 0.098
DT_BY_GEN = {"SSJ4": 0.098, "SSJ5": 0.05}


@dataclass(frozen=True)
class SSJDay:
    """One day of decoded SSJ data, second-resolution."""
    sat: str                       # "F17" / "F18" / ...
    sensor: str                    # "SSJ4" / "SSJ5"
    times: np.ndarray              # datetime64[ns], shape (N,)
    glat: np.ndarray               # geographic latitude, deg, shape (N,)
    glon: np.ndarray               # geographic longitude, deg
    altitude_km: np.ndarray        # spacecraft altitude, km
    cgm_lat: np.ndarray            # corrected geomagnetic latitude at 110 km
    cgm_lon: np.ndarray            # CGM longitude at 110 km
    mlt_hours: np.ndarray          # magnetic local time, hours
    e_counts: np.ndarray           # decompressed electron counts, shape (N,20)
    i_counts: np.ndarray           # decompressed ion counts, shape (N,20)
    e_diff_energy_flux: np.ndarray  # eV/(cm² s sr eV), shape (N,20)
    i_diff_energy_flux: np.ndarray
    status1: np.ndarray            # SSJ5 status word 1 (uint16)
    status2: np.ndarray            # SSJ5 status word 2 (uint16)


def _decompress(raw: np.ndarray) -> np.ndarray:
    """Apply log decompression to 16-bit raw values.

    raw == 0 → missing (returned as -1); raw == 1 → 0 counts.
    """
    raw = raw.astype(np.int64)
    x = raw % 32
    y = (raw - x) // 32
    counts = (x + 32) * (2 ** y) - 33
    counts = np.where(raw == 0, -1, counts)
    return counts.astype(np.float64)


def _decode_lat(raw: np.ndarray) -> np.ndarray:
    """Word 6/9/11 latitude conversion: (i-900)/10, with i>1800 branch."""
    raw = raw.astype(np.float64)
    return np.where(raw > 1800, (raw - 4995) / 10.0, (raw - 900) / 10.0)


def _read_words(path: Path) -> np.ndarray:
    """Return a flat uint16 big-endian array from a (possibly gzipped) file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=">u2").astype(np.uint16)
    return arr


def _detect_sensor(path: Path) -> tuple[str, str]:
    """Return ('F17', 'SSJ5') from filename like 'j5f1722001.gz'."""
    name = path.name
    sensor_digit = name[1]   # '4' or '5'
    sat_num = name[3:5]
    sensor = "SSJ4" if sensor_digit == "4" else "SSJ5"
    sat = f"F{sat_num}"
    return sat, sensor


def parse(path: str | Path) -> SSJDay:
    path = Path(path)
    sat, sensor = _detect_sensor(path)
    words = _read_words(path)
    nminutes = words.size // WORDS_PER_MINUTE
    if nminutes == 0:
        raise ValueError(f"{path}: no full minute blocks")
    words = words[: nminutes * WORDS_PER_MINUTE].reshape(nminutes, WORDS_PER_MINUTE)

    # ---- Per-minute header (words 1..15, 0-indexed 0..14) ----------------
    doy   = words[:, 0].astype(np.int64)
    hr0   = words[:, 1].astype(np.int64)
    mn0   = words[:, 2].astype(np.int64)
    sc0   = words[:, 3].astype(np.int64)
    yr_raw = words[:, 4].astype(np.int64)         # 22 for 2022 (i.e. year-1950 ... or 2-digit)
    glat0  = _decode_lat(words[:, 5])
    glon0  = words[:, 6].astype(np.float64) / 10.0
    alt_nmi = words[:, 7].astype(np.float64)
    cgm_lat0 = _decode_lat(words[:, 10])
    cgm_lon0 = words[:, 11].astype(np.float64) / 10.0
    mlt_h  = words[:, 12].astype(np.int64)
    mlt_m  = words[:, 13].astype(np.int64)
    mlt_s  = words[:, 14].astype(np.int64)
    mlt_hours0 = mlt_h + mlt_m / 60.0 + mlt_s / 3600.0

    # Year — spec says conversion is "i - 50". Empirically the stored value
    # is `year - 1950`: i=72 → 2022 (range 37..99 covers 1987..2049).
    year_full = 1950 + yr_raw

    # ---- Millisecond flag ------------------------------------------------
    ms_flag = words[:, 2595]  # word 2596 (0-indexed 2595)

    # ---- Per-second blocks (60 × 43 words starting at index 15) ----------
    sec_block = words[:, HEADER_WORDS:HEADER_WORDS + 60 * WORDS_PER_SECOND]
    sec_block = sec_block.reshape(nminutes, 60, WORDS_PER_SECOND)

    sec_hour = sec_block[..., 0].astype(np.int64)
    sec_min  = sec_block[..., 1].astype(np.int64)
    sec_sec_raw = sec_block[..., 2].astype(np.int64)

    e_raw = sec_block[..., 3:23].copy()   # (nmin, 60, 20)  — file order
    i_raw = sec_block[..., 23:43].copy()  # (nmin, 60, 20)

    # Pull SSJ5 status words out before decompressing
    status1 = e_raw[..., STATUS_FILE_POS].astype(np.uint16)
    status2 = i_raw[..., STATUS_FILE_POS].astype(np.uint16)
    if sensor == "SSJ5":
        # Mark the channel-11 slot as missing (will not be a real reading)
        e_raw[..., STATUS_FILE_POS] = 0
        i_raw[..., STATUS_FILE_POS] = 0

    e_counts_file = _decompress(e_raw)
    i_counts_file = _decompress(i_raw)

    # Reorder into channel-1..20 layout
    e_counts = e_counts_file[..., CHANNEL_TO_FILE_ORDER]
    i_counts = i_counts_file[..., CHANNEL_TO_FILE_ORDER]

    # ---- Build absolute timestamps --------------------------------------
    # Use the per-second hour/minute words; second is 0..59 (or ms*1000).
    use_ms = ms_flag == 1                              # (nminutes,)
    ms_per_sec = np.where(use_ms[:, None], sec_sec_raw, sec_sec_raw * 1000)
    # Compose UT seconds since start of day
    ut_sec = sec_hour.astype(np.float64) * 3600.0 + sec_min.astype(np.float64) * 60.0 + ms_per_sec.astype(np.float64) / 1000.0

    # Day-start datetime per minute
    base = []
    for k in range(nminutes):
        base.append(datetime(int(year_full[k]), 1, 1, tzinfo=timezone.utc)
                    + timedelta(days=int(doy[k]) - 1))
    base_arr = np.array([np.datetime64(b.replace(tzinfo=None), "ns") for b in base])
    times = base_arr[:, None] + (ut_sec * 1e9).astype("timedelta64[ns]")

    # ---- Position / MLT: only one sample per minute is stored.  Repeat
    # the header values across the 60-sec block (good to a few percent;
    # F17 moves ~7.5 km/s ≈ 0.07° latitude per second).
    glat = np.repeat(glat0, 60).reshape(nminutes, 60)
    glon = np.repeat(glon0, 60).reshape(nminutes, 60)
    alt_km = np.repeat(alt_nmi * 1.852, 60).reshape(nminutes, 60)
    cgm_lat = np.repeat(cgm_lat0, 60).reshape(nminutes, 60)
    cgm_lon = np.repeat(cgm_lon0, 60).reshape(nminutes, 60)
    mlt = np.repeat(mlt_hours0, 60).reshape(nminutes, 60)

    # ---- Differential energy flux ---------------------------------------
    g_e = ELECTRON_GEOM_FACTORS.get(sat)
    g_i = ION_GEOM_FACTORS.get(sat)
    dt = DT_BY_GEN[sensor]
    if g_e is None or g_i is None:
        e_dje = np.full_like(e_counts, np.nan)
        i_dje = np.full_like(i_counts, np.nan)
    else:
        # JE_i = (C_i / (G_i * dt)) * E_i  units: eV /(cm² s sr eV)
        with np.errstate(invalid="ignore", divide="ignore"):
            e_dje = (e_counts / (g_e[None, None, :] * dt)) * CHANNEL_ENERGY_EV[None, None, :]
            i_dje = (i_counts / (g_i[None, None, :] * dt)) * CHANNEL_ENERGY_EV[None, None, :]
        e_dje[e_counts < 0] = np.nan
        i_dje[i_counts < 0] = np.nan

    # ---- Flatten (nminutes × 60) → (N,) -----------------------------------
    flat = lambda a: a.reshape(-1, *a.shape[2:])
    return SSJDay(
        sat=sat,
        sensor=sensor,
        times=flat(times),
        glat=flat(glat),
        glon=flat(glon),
        altitude_km=flat(alt_km),
        cgm_lat=flat(cgm_lat),
        cgm_lon=flat(cgm_lon),
        mlt_hours=flat(mlt),
        e_counts=flat(e_counts),
        i_counts=flat(i_counts),
        e_diff_energy_flux=flat(e_dje),
        i_diff_energy_flux=flat(i_dje),
        status1=flat(status1),
        status2=flat(status2),
    )


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: parse_dmsp_ssj.py <file.gz> [...]", file=sys.stderr)
        sys.exit(1)
    for p in sys.argv[1:]:
        d = parse(p)
        print(f"\n=== {p} ===")
        print(f"  sat={d.sat} sensor={d.sensor} N={len(d.times)}")
        print(f"  time range: {d.times[0]} .. {d.times[-1]}")
        valid = d.cgm_lat[~np.isnan(d.cgm_lat)]
        if valid.size:
            print(f"  cgm_lat range: [{valid.min():.1f}, {valid.max():.1f}]")
        polar = np.abs(d.cgm_lat) > 75
        print(f"  polar-cap (|cgm_lat|>75°) seconds: {int(polar.sum())}")
        # Total electron energy flux (integrated over channels)
        je_total = np.nansum(
            d.e_diff_energy_flux * CHANNEL_DELTA_EV[None, :], axis=1
        )
        if polar.any():
            print(
                f"  median JE_e on polar cap: "
                f"{np.nanmedian(je_total[polar]):.2e} eV/(cm² s sr)"
            )


if __name__ == "__main__":
    main()
