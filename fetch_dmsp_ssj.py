#!/usr/bin/env python3
"""
fetch_dmsp_ssj.py — DMSP/SSJ データのダウンロード

Download DMSP/SSJ Simple-Format files from NCEI for a given date.
Tries F15..F18 in turn and keeps every file the server returns.

NCEI directory layout:
  https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/access/{sat}/ssj/{YYYY}/{MM}/{file}.gz

Filename convention: jNFNNYYDDD.gz
  N  = 4 (SSJ4) for F12-F15;  5 (SSJ5) for F16-F20
  NN = spacecraft (e.g. 17, 18)
  YY = 2-digit year
  DDD = day-of-year
"""
from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

BASE_URL = "https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/access"
# ダウンロード先（このチュートリアルフォルダの下に data/YYYY-MM-DD/ ができる）
ROOT = Path(__file__).resolve().parent / "data"

# (sat, sensor_digit, valid_start, valid_end)
# 注: 2024年以降のNCEIアーカイブは断続的（2024年は1月と5月、2025年は5月のみ等）。
# valid_end は「これ以降は絶対ない」保証ではなく探索範囲の目安。ない日は404でスキップされる。
SATS = [
    ("f15", "4", date(1999, 12, 17), date(2009,  3, 25)),
    ("f16", "5", date(2003, 10, 24), date(2025, 12, 31)),
    ("f17", "5", date(2006, 11,  7), date(2025, 12, 31)),
    ("f18", "5", date(2009, 10, 21), date(2025, 12, 31)),
]


def _filename(sat: str, sensor_digit: str, d: date) -> str:
    yy = f"{d.year % 100:02d}"
    doy = d.timetuple().tm_yday
    return f"j{sensor_digit}{sat}{yy}{doy:03d}.gz"


def _url(sat: str, sensor_digit: str, d: date) -> str:
    return f"{BASE_URL}/{sat}/ssj/{d.year}/{d.month:02d}/{_filename(sat, sensor_digit, d)}"


def fetch_for_date(d: date) -> list[Path]:
    """Download all available F-series SSJ files for the given UTC date.

    Returns the list of local paths that exist after the call (cache + new).
    """
    out_dir = ROOT / d.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    found: list[Path] = []

    for sat, sd, t0, t1 in SATS:
        if not (t0 <= d <= t1):
            continue
        local = out_dir / _filename(sat, sd, d)
        if local.exists() and local.stat().st_size > 0:
            found.append(local)
            continue
        try:
            urlretrieve(_url(sat, sd, d), local)
            if local.stat().st_size == 0:
                local.unlink(missing_ok=True)
                continue
            found.append(local)
            print(f"  [fetched] {local.name}")
        except (HTTPError, URLError):
            local.unlink(missing_ok=True)
        except Exception as e:
            print(f"  [error]   {sat} {d}: {e}")
            local.unlink(missing_ok=True)
    return found


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: fetch_dmsp_ssj.py YYYY-MM-DD [...]", file=sys.stderr)
        raise SystemExit(1)
    for s in sys.argv[1:]:
        d = datetime.fromisoformat(s).date()
        files = fetch_for_date(d)
        print(f"{d}: {len(files)} files -> {[p.name for p in files]}")


if __name__ == "__main__":
    main()
