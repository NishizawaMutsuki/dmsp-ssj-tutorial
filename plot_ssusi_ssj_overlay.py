#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "netCDF4"]
# ///
"""
plot_ssusi_ssj_overlay.py — SSUSI画像 + SSJ粒子スペクトログラムの6パネル図

使い方:
  python plot_ssusi_ssj_overlay.py 2022-12-25 f18

これだけで SSJ と SSUSI を取得し、北/南半球のSSUSI極投影、軌道トラック、
イオン/電子エネルギースペクトログラムを1枚のPNGにまとめる。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "dmsp_ssj_tutorial_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "dmsp_ssj_tutorial_xdg_cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

try:
    from netCDF4 import Dataset
except ImportError as exc:  # pragma: no cover - user-facing setup failure
    raise SystemExit("netCDF4 が必要です。先に `pip install netCDF4` を実行してください。") from exc

import fetch_dmsp_ssj
import fetch_ssusi
from parse_dmsp_ssj import CHANNEL_ENERGY_EV, parse


HERE = Path(__file__).resolve().parent
SSJ_LOCAL_ROOT = HERE / "data"
SSUSI_LOCAL_ROOT = HERE / "data" / "ssusi"
SSJ_HEMISPHERE_PASS_LAT_FLOOR_DEG = 45.0
SSJ_HEMISPHERE_PASS_MAX_GAP_SECONDS = 120.0
SSUSI_RADIANCE_TICKS = np.arange(0, 1601, 200)


@dataclass(frozen=True)
class SSUSIObservation:
    hemisphere: str
    time_utc: datetime
    nc_path: Path


@dataclass(frozen=True)
class HemisphereConfig:
    hemisphere: str
    ssusi_nc_path: Path
    image_time_utc: str


@dataclass(frozen=True)
class PassData:
    config: HemisphereConfig
    times: np.ndarray
    minutes: np.ndarray
    cgm_lat: np.ndarray
    mlt_hours: np.ndarray
    energies_ev: np.ndarray
    ion_log_flux: np.ndarray
    electron_log_flux: np.ndarray


def parse_date(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"日付は YYYY-MM-DD で指定してください: {text}") from exc


def normalize_satellite(text: str) -> str:
    sat = text.strip().lower()
    if sat.startswith("dmsp"):
        sat = sat.removeprefix("dmsp")
    if not sat.startswith("f"):
        sat = f"f{sat}"
    if sat not in {"f15", "f16", "f17", "f18"}:
        raise SystemExit(f"このチュートリアルのSSJ図は f15/f16/f17/f18 に対応しています: {text}")
    return sat


def normalize_hemisphere(text: str) -> str:
    hemi = text.strip().upper()
    if hemi in {"N", "NORTH"}:
        return "N"
    if hemi in {"S", "SOUTH"}:
        return "S"
    raise SystemExit(f"半球は N または S で指定してください: {text}")


def parse_target_time(text: str) -> time:
    try:
        return datetime.strptime(text, "%H:%M").time()
    except ValueError as exc:
        raise SystemExit(f"--target-time は HH:MM で指定してください: {text}") from exc


def utc_np(value: str) -> np.datetime64:
    return np.datetime64(value.replace("Z", ""), "ns")


def np_datetime_to_datetime(value: np.datetime64) -> datetime:
    milliseconds = value.astype("datetime64[ms]").astype(np.int64)
    return datetime(1970, 1, 1) + timedelta(milliseconds=int(milliseconds))


def datetime_to_utc_text(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def format_short_date(value: date) -> str:
    return value.strftime("%Y %b %d")


def format_hhmm(value: np.datetime64 | datetime) -> str:
    if isinstance(value, np.datetime64):
        value = np_datetime_to_datetime(value)
    return value.strftime("%H%M")


def format_hhmm_colon(value: datetime) -> str:
    return value.strftime("%H:%M")


def format_hhmmss(value: np.datetime64) -> str:
    return np_datetime_to_datetime(value).strftime("%H:%M:%S")


def doy_to_month_day(year: int, doy: int) -> tuple[int, int]:
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return dt.month, dt.day


def month_to_abbreviation(month: int) -> str:
    names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    return names.get(month, "Invalid")


def time_to_angle(decimal_hour: np.ndarray | float) -> np.ndarray:
    time_adjusted = 12 - np.asarray(decimal_hour, dtype=float)
    angle = (time_adjusted / 24) * 360
    return np.where(angle < 0, angle + 360, angle)


def decimal_time_to_hh_mm(decimal_time: float) -> str:
    hours = int(decimal_time)
    minutes = int(round((decimal_time - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours % 24:02d}:{minutes:02d}"


def first_ut_to_time(array_2d: np.ndarray) -> str:
    values = np.asarray(array_2d, dtype=float)
    nz = values[np.isfinite(values) & (values > 0)]
    if nz.size == 0:
        return "--:--"
    return decimal_time_to_hh_mm(float(nz.min()))


def first_ut_to_datetime(year: int, doy: int, array_2d: np.ndarray) -> datetime | None:
    values = np.asarray(array_2d, dtype=float)
    nz = values[np.isfinite(values) & (values > 0)]
    if nz.size == 0:
        return None
    seconds = int(round(float(nz.min()) * 3600.0))
    base = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return base + timedelta(seconds=seconds)


def load_data(fname: str | Path) -> dict[str, np.ndarray]:
    dataset = Dataset(fname, "r")
    try:
        return {variable: dataset[variable][:] for variable in dataset.variables}
    finally:
        dataset.close()


def create_custom_colormap() -> tuple[colors.Colormap, colors.BoundaryNorm]:
    rgb = [
        [24, 0, 29],
        [10, 42, 218],
        [86, 185, 253],
        [115, 246, 202],
        [115, 245, 87],
        [145, 247, 76],
        [255, 251, 84],
        [241, 117, 48],
        [234, 51, 35],
    ]
    cmap = colors.LinearSegmentedColormap.from_list("ssusi_lbhs", np.array(rgb) / 255.0)
    norm = colors.BoundaryNorm([0, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600], ncolors=256)
    return cmap, norm


def ssusi_polar_coordinates(mlat: np.ndarray, mlt: np.ndarray, absolute_latitude: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(mlat, dtype=float)
    if absolute_latitude:
        lat = np.abs(lat)
    radius = 90.0 - lat
    theta = np.deg2rad(time_to_angle(np.asarray(mlt, dtype=float)))
    return theta, radius


def plot_ssusi_radiance(ax: plt.Axes, lat_grid: np.ndarray, mlt_grid: np.ndarray, radiance: np.ndarray) -> object:
    theta, radius = ssusi_polar_coordinates(lat_grid, mlt_grid)
    cmap, norm = create_custom_colormap()
    return ax.scatter(theta, radius, c=radiance, cmap=cmap, norm=norm, alpha=1, s=0.8, marker="s")


def add_ssusi_colorbar(fig: plt.Figure, mappable: object, cax: plt.Axes) -> None:
    cbar = fig.colorbar(mappable, cax=cax, ticks=SSUSI_RADIANCE_TICKS)
    cbar.set_label("SSUSI LBHS radiance (R)", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)


def configure_ssusi_polar_axes(ax: plt.Axes, year: int, month: int, day: int, ut_label: str) -> None:
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.xaxis.set_tick_params(pad=-20)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor("#666666")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticks([10, 20])
    ax.yaxis.grid(True, linestyle=(0, (10, 10)), color="black", lw=0.4)

    for ang_deg in range(0, 360, 30):
        ang = np.deg2rad(ang_deg)
        ax.plot([ang, ang], [10, 90], linestyle=(0, (10, 10)), color="black", lw=0.4, zorder=0)

    ax.set_yticklabels(["80°", "70°"], color="white", fontsize=7)
    ax.set_rlim(0, 40)

    for ang, label, rotation in zip([0, 90, 180, 270], ["12 MLT", "06 MLT", "24 MLT", "18 MLT"], [0, -90, 0, 90], strict=True):
        ax.text(
            np.deg2rad(ang),
            37,
            label,
            color="white",
            fontsize=9,
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
        )

    date_text = f"{day:02d} {month_to_abbreviation(month)} {year:04d} {ut_label} UT"
    ax.text(0.97, 0.97, date_text, transform=ax.transAxes, ha="right", va="bottom", color="black", fontsize=9)


def ssj_filename_for_sat(d: date, sat: str) -> str:
    sat = normalize_satellite(sat)
    yy = f"{d.year % 100:02d}"
    doy = f"{d.timetuple().tm_yday:03d}"
    sensor_digit = "4" if sat == "f15" else "5"
    return f"j{sensor_digit}{sat}{yy}{doy}.gz"


def ssj_file_matches(path: Path, d: date, sat: str) -> bool:
    return path.name.lower() == ssj_filename_for_sat(d, sat).lower()


def find_local_ssj_file(d: date, sat: str) -> Path | None:
    folder = SSJ_LOCAL_ROOT / d.isoformat()
    if not folder.exists():
        return None
    for path in sorted(folder.glob("*.gz")):
        if ssj_file_matches(path, d, sat) and path.stat().st_size > 0:
            return path
    return None


def copy_ssj_from_repo_cache(d: date, sat: str) -> Path | None:
    for parent in HERE.parents:
        source_dir = parent / "data" / "raw" / "dmsp_ssj" / d.isoformat()
        if not source_dir.exists():
            continue
        for source in sorted(source_dir.glob("*.gz")):
            if not ssj_file_matches(source, d, sat) or source.stat().st_size == 0:
                continue
            out_dir = SSJ_LOCAL_ROOT / d.isoformat()
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / source.name
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            print(f"  [cache] {source_dir} から {target.name} をコピーしました")
            return target
    return None


def ensure_ssj_file(d: date, sat: str, allow_fetch: bool, include_repo_cache: bool) -> Path:
    cached = find_local_ssj_file(d, sat)
    if cached is not None:
        return cached

    if include_repo_cache:
        copied = copy_ssj_from_repo_cache(d, sat)
        if copied is not None:
            return copied

    if allow_fetch:
        fetch_dmsp_ssj.ROOT = SSJ_LOCAL_ROOT
        fetch_dmsp_ssj.fetch_for_date(d)
        cached = find_local_ssj_file(d, sat)
        if cached is not None:
            return cached

    raise SystemExit(
        f"{d.isoformat()} {sat.upper()} のSSJファイルがありません。\n"
        f"NCEIにその日の{sat.upper()}データがない、またはネットワークから取得できません。"
    )


def ensure_ssusi_files(
    d: date,
    sat: str,
    allow_fetch: bool,
    include_repo_cache: bool,
    base_urls: Iterable[str],
) -> list[Path]:
    cached = fetch_ssusi.existing_files(d, sat, SSUSI_LOCAL_ROOT)
    if cached:
        return cached

    if allow_fetch:
        fetch_ssusi.ROOT = SSUSI_LOCAL_ROOT
        files = fetch_ssusi.fetch_for_date(d, sat, include_repo_cache=include_repo_cache, extra_bases=base_urls)
    else:
        files = []

    if not files:
        raise SystemExit(
            f"{d.isoformat()} {sat.upper()} のSSUSI NetCDFがありません。\n"
            f"JHU/APL公開サーバにその日の{sat.upper()}データがない、または現在サーバが利用できません。"
        )
    return files


def hemisphere_name(hemisphere: str) -> str:
    return "NORTH" if hemisphere == "N" else "SOUTH"


def nc_satellite(path: Path) -> str | None:
    """APL命名とSPDF命名の両方から衛星名 (f18等) を取り出す。"""
    name = path.name
    # APL: PS.APL_..._GP.F18-SSUSI_...
    marker = "_GP."
    if marker in name:
        rest = name.split(marker, 1)[1]
        if "-SSUSI" in rest:
            return rest.split("-SSUSI", 1)[0].lower()
        return None
    # SPDF: dmspf18_ssusi_edr-aurora_...
    lower = name.lower()
    if lower.startswith("dmsp") and "_ssusi_" in lower:
        return lower.split("_", 1)[0].removeprefix("dmsp")
    return None


def observation_from_nc(path: Path, hemisphere: str) -> SSUSIObservation | None:
    if nc_satellite(path) is None:
        return None
    ssusi = load_data(path)
    year = int(ssusi["YEAR"])
    doy = int(ssusi["DOY"])
    ut_key = "UT_N" if hemisphere == "N" else "UT_S"
    if ut_key not in ssusi:
        return None
    obs_time = first_ut_to_datetime(year, doy, ssusi[ut_key])
    if obs_time is None:
        return None
    return SSUSIObservation(hemisphere=hemisphere, time_utc=obs_time, nc_path=path)


def discover_observations(files: Iterable[Path], d: date, sat: str) -> list[SSUSIObservation]:
    observations: list[SSUSIObservation] = []
    for path in sorted(files):
        if nc_satellite(path) != sat:
            continue
        if not fetch_ssusi.file_matches(path, d, sat):
            continue
        for hemisphere in ("N", "S"):
            obs = observation_from_nc(path, hemisphere)
            if obs is not None:
                observations.append(obs)
    return sorted(observations, key=lambda obs: (obs.hemisphere, obs.time_utc, obs.nc_path.name))


def select_pair(
    observations: list[SSUSIObservation],
    d: date,
    primary: str,
    target: time,
    opposite_policy: str,
    max_pair_minutes: float,
) -> dict[str, SSUSIObservation]:
    by_hemi = {"N": [], "S": []}
    for obs in observations:
        by_hemi[obs.hemisphere].append(obs)
    if not by_hemi["N"] or not by_hemi["S"]:
        raise SystemExit(f"{d.isoformat()} は N/S 両半球のSSUSI画像がそろっていません。")

    primary_list = by_hemi[primary]
    target_dt = datetime.combine(d, target)
    anchor = min(primary_list, key=lambda obs: abs((obs.time_utc - target_dt).total_seconds()))
    opposite = "S" if primary == "N" else "N"
    candidates = by_hemi[opposite]

    def minutes_delta(obs: SSUSIObservation) -> float:
        return abs((obs.time_utc - anchor.time_utc).total_seconds()) / 60.0

    if opposite_policy == "after":
        preferred = [obs for obs in candidates if obs.time_utc >= anchor.time_utc and minutes_delta(obs) <= max_pair_minutes]
    elif opposite_policy == "before":
        preferred = [obs for obs in candidates if obs.time_utc <= anchor.time_utc and minutes_delta(obs) <= max_pair_minutes]
    else:
        preferred = [obs for obs in candidates if minutes_delta(obs) <= max_pair_minutes]

    if not preferred:
        preferred = [obs for obs in candidates if minutes_delta(obs) <= max_pair_minutes]
    if not preferred:
        raise SystemExit(
            f"{anchor.time_utc:%H:%M} UT の{primary}半球画像に対して、"
            f"{max_pair_minutes:.0f}分以内の反対半球SSUSI画像がありません。"
        )

    mate = min(preferred, key=minutes_delta)
    return {"N": anchor if primary == "N" else mate, "S": mate if primary == "N" else anchor}


def edges_from_centers(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    inner = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (inner[0] - values[0])
    last = values[-1] + (values[-1] - inner[-1])
    return np.concatenate([[first], inner, [last]])


def log_edges_from_centers(values: np.ndarray) -> np.ndarray:
    return 10.0 ** edges_from_centers(np.log10(np.asarray(values, dtype=float)))


def ordered_energy_and_flux(flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_channel = np.any(np.isfinite(flux) & (flux > 0), axis=0)
    energies = CHANNEL_ENERGY_EV[valid_channel]
    clean_flux = np.where(np.isfinite(flux[:, valid_channel]) & (flux[:, valid_channel] > 0), flux[:, valid_channel], np.nan)
    order = np.argsort(energies)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_flux = np.log10(clean_flux[:, order])
    return energies[order], log_flux


def hemisphere_pass_mask(times: np.ndarray, cgm_lat: np.ndarray, hemisphere: str) -> np.ndarray:
    lat = np.asarray(cgm_lat, dtype=float)
    finite = np.isfinite(lat) & ~np.isnat(times)
    if hemisphere == "N":
        return finite & (lat >= SSJ_HEMISPHERE_PASS_LAT_FLOOR_DEG)
    return finite & (lat <= -SSJ_HEMISPHERE_PASS_LAT_FLOOR_DEG)


def contiguous_hemisphere_pass_indices(day: object, image_time_utc: str, hemisphere: str) -> np.ndarray:
    timestamp = utc_np(image_time_utc)
    mask = hemisphere_pass_mask(day.times, day.cgm_lat, hemisphere)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        return candidates

    distances = np.abs((day.times[candidates] - timestamp) / np.timedelta64(1, "s"))
    anchor = int(candidates[int(np.nanargmin(distances))])

    left = anchor
    while left > 0 and mask[left - 1]:
        gap_seconds = (day.times[left] - day.times[left - 1]) / np.timedelta64(1, "s")
        if float(gap_seconds) < 0 or float(gap_seconds) > SSJ_HEMISPHERE_PASS_MAX_GAP_SECONDS:
            break
        left -= 1

    right = anchor
    while right + 1 < len(day.times) and mask[right + 1]:
        gap_seconds = (day.times[right + 1] - day.times[right]) / np.timedelta64(1, "s")
        if float(gap_seconds) < 0 or float(gap_seconds) > SSJ_HEMISPHERE_PASS_MAX_GAP_SECONDS:
            break
        right += 1

    return np.arange(left, right + 1)


def extract_pass(day: object, config: HemisphereConfig) -> PassData:
    indices = contiguous_hemisphere_pass_indices(day, config.image_time_utc, config.hemisphere)
    if indices.size == 0:
        raise SystemExit(f"{config.image_time_utc} 付近に同半球のSSJパスが見つかりません。")

    times = day.times[indices]
    start = times[0]
    minutes = (times - start).astype("timedelta64[ms]").astype(float) / 60000.0
    ion_energies, ion_log_flux = ordered_energy_and_flux(day.i_diff_energy_flux[indices])
    electron_energies, electron_log_flux = ordered_energy_and_flux(day.e_diff_energy_flux[indices])
    if not np.allclose(ion_energies, electron_energies):
        raise SystemExit("イオン/電子のエネルギーグリッドが一致しません。")

    return PassData(
        config=config,
        times=times,
        minutes=minutes,
        cgm_lat=day.cgm_lat[indices],
        mlt_hours=day.mlt_hours[indices],
        energies_ev=electron_energies,
        ion_log_flux=ion_log_flux,
        electron_log_flux=electron_log_flux,
    )


def evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=int)
    count = min(count, size)
    return np.unique(np.linspace(0, size - 1, count).round().astype(int))


def render_ssusi_background(ax: plt.Axes, config: HemisphereConfig) -> object:
    ssusi = load_data(config.ssusi_nc_path)
    year = int(ssusi["YEAR"])
    doy = int(ssusi["DOY"])
    month, day = doy_to_month_day(year, doy)
    hemi_name = hemisphere_name(config.hemisphere)
    radiance = ssusi[f"DISK_RADIANCEDATA_INTENSITY_{hemi_name}"][3, :, :].copy()
    radiance[radiance == 0] = np.nan
    artist = plot_ssusi_radiance(
        ax,
        ssusi["LATITUDE_GEOMAGNETIC_GRID_MAP"],
        ssusi["MLT_GRID_MAP"],
        radiance,
    )
    ut_key = "UT_N" if config.hemisphere == "N" else "UT_S"
    configure_ssusi_polar_axes(ax, year, month, day, first_ut_to_time(ssusi[ut_key]))
    return artist


def visible_track_points(pass_data: PassData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, radius = ssusi_polar_coordinates(pass_data.cgm_lat, pass_data.mlt_hours, absolute_latitude=True)
    valid = np.isfinite(theta) & np.isfinite(radius) & (radius >= 0.0) & (radius <= 40.0)
    indices = np.flatnonzero(valid)
    return theta[indices], radius[indices], indices


def hemisphere_title(hemisphere: str) -> str:
    return "Northern Hemisphere" if hemisphere == "N" else "Southern Hemisphere"


def plot_ssusi_panel(ax: plt.Axes, pass_data: PassData, sat: str) -> object:
    artist = render_ssusi_background(ax, pass_data.config)
    theta, radius, source_indices = visible_track_points(pass_data)
    if theta.size:
        line = ax.plot(theta, radius, color="white", lw=2.0, zorder=5)[0]
        line.set_path_effects([pe.Stroke(linewidth=3.3, foreground="black", alpha=0.7), pe.Normal()])

        marker_indices = evenly_spaced_indices(theta.size, 6)
        ax.scatter(
            theta[marker_indices],
            radius[marker_indices],
            s=23,
            facecolor="#202020",
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )

        label_indices = evenly_spaced_indices(theta.size, 5)
        for idx in label_indices:
            source_idx = source_indices[idx]
            ax.annotate(
                format_hhmmss(pass_data.times[source_idx]),
                (theta[idx], radius[idx]),
                xytext=(4, 5),
                textcoords="offset points",
                color="white",
                fontsize=7.5,
                ha="left",
                va="bottom",
                zorder=7,
                path_effects=[pe.Stroke(linewidth=2.2, foreground="black", alpha=0.85), pe.Normal()],
            )

    image_time = datetime.strptime(pass_data.config.image_time_utc.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
    title = f"{hemisphere_title(pass_data.config.hemisphere)}  {sat.upper()} SSUSI  {image_time:%H%M} UT"
    ax.text(0.5, 1.012, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    return artist


def setup_energy_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_facecolor("black")
    ax.set_yscale("log")
    ax.set_ylim(30, 30000)
    ax.set_yticks([100, 1000, 10000])
    ax.set_yticklabels(["100", "1000", "10,000"])
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=9, length=3)
    ax.tick_params(axis="x", which="minor", bottom=False)


def plot_spectrogram(
    ax: plt.Axes,
    pass_data: PassData,
    log_flux: np.ndarray,
    norm: colors.Normalize,
    cmap: colors.Colormap,
    ylabel: str,
) -> object:
    x_edges = edges_from_centers(pass_data.minutes)
    y_edges = log_edges_from_centers(pass_data.energies_ev)
    mesh = ax.pcolormesh(x_edges, y_edges, log_flux.T, cmap=cmap, norm=norm, shading="flat", rasterized=True)
    setup_energy_axis(ax, ylabel)
    ax.set_xlim(x_edges[0], x_edges[-1])
    return mesh


def tick_positions(pass_data: PassData, count: int = 7) -> tuple[np.ndarray, list[int]]:
    positions = np.linspace(pass_data.minutes[0], pass_data.minutes[-1], count)
    indices = [int(np.nanargmin(np.abs(pass_data.minutes - pos))) for pos in positions]
    return positions, indices


def apply_pass_tick_labels(ax: plt.Axes, pass_data: PassData, event_date: date) -> None:
    positions, indices = tick_positions(pass_data)
    labels = []
    for idx in indices:
        labels.append(f"{pass_data.mlt_hours[idx]:.1f}\n{pass_data.cgm_lat[idx]:.1f}\n{format_hhmm(pass_data.times[idx])}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, linespacing=1.05)
    ax.tick_params(axis="x", pad=2)
    ax.text(
        -0.045,
        -0.035,
        f"MLT\nMLAT\nhhmm\n{format_short_date(event_date)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        linespacing=1.05,
        clip_on=False,
    )


def hide_x_labels(ax: plt.Axes, pass_data: PassData) -> None:
    positions, _ = tick_positions(pass_data)
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.05, label, transform=ax.transAxes, ha="right", va="top", fontsize=13, fontweight="bold", clip_on=False)


def build_pair_figure(
    d: date,
    sat: str,
    ssj_path: Path,
    pair: dict[str, SSUSIObservation],
    out_dir: Path,
    dpi: int,
) -> Path:
    day = parse(ssj_path)
    configs = {
        hemi: HemisphereConfig(hemisphere=hemi, ssusi_nc_path=obs.nc_path, image_time_utc=datetime_to_utc_text(obs.time_utc))
        for hemi, obs in pair.items()
    }
    passes = [extract_pass(day, configs["N"]), extract_pass(day, configs["S"])]

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    ion_norm = colors.Normalize(vmin=3, vmax=8)
    electron_norm = colors.Normalize(vmin=5, vmax=9)

    fig = plt.figure(figsize=(12.8, 9.4), facecolor="white")
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 1.0, 0.028],
        height_ratios=[2.55, 1.05, 1.12],
        wspace=0.26,
        hspace=0.12,
    )

    ax_ssusi = [fig.add_subplot(gs[0, col], projection="polar") for col in range(2)]
    ax_ion = [fig.add_subplot(gs[1, col]) for col in range(2)]
    ax_electron = [fig.add_subplot(gs[2, col]) for col in range(2)]
    cax_ssusi = fig.add_subplot(gs[0, 2])
    cax_ion = fig.add_subplot(gs[1, 2])
    cax_electron = fig.add_subplot(gs[2, 2])

    ssusi_artist = None
    for col, pass_data in enumerate(passes):
        ssusi_artist = plot_ssusi_panel(ax_ssusi[col], pass_data, sat)
    add_ssusi_colorbar(fig, ssusi_artist, cax_ssusi)

    ion_mesh = None
    electron_mesh = None
    for col, pass_data in enumerate(passes):
        ion_mesh = plot_spectrogram(
            ax_ion[col],
            pass_data,
            pass_data.ion_log_flux,
            ion_norm,
            cmap,
            "Energy of ions\n(eV)",
        )
        electron_mesh = plot_spectrogram(
            ax_electron[col],
            pass_data,
            pass_data.electron_log_flux,
            electron_norm,
            cmap,
            "Energy of electrons\n(eV)",
        )
        hide_x_labels(ax_ion[col], pass_data)
        apply_pass_tick_labels(ax_electron[col], pass_data, d)

    for ax in [ax_ion[1], ax_electron[1]]:
        ax.set_ylabel("")

    ion_cbar = fig.colorbar(ion_mesh, cax=cax_ion, ticks=np.arange(3, 9))
    ion_cbar.set_label("Log energy flux\n(eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$)", fontsize=9)
    ion_cbar.ax.tick_params(labelsize=8.5)

    electron_cbar = fig.colorbar(electron_mesh, cax=cax_electron, ticks=np.arange(5, 10))
    electron_cbar.set_label("Log energy flux\n(eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$)", fontsize=9)
    electron_cbar.ax.tick_params(labelsize=8.5)

    for label, ax in zip(["A", "B", "C", "E", "D", "F"], [*ax_ssusi, *ax_ion, *ax_electron], strict=True):
        add_panel_label(ax, label)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ssusi_ssj_overlay_{sat.upper()}_{d:%Y%m%d}_N{pair['N'].time_utc:%H%M}_S{pair['S'].time_utc:%H%M}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="UTC日付: YYYY-MM-DD")
    parser.add_argument("satellite", help="衛星: f15/f16/f17/f18")
    parser.add_argument("--target-time", default="12:00", help="選ぶSSUSI画像の目標UT時刻。既定: 12:00")
    parser.add_argument("--primary-hemisphere", default="N", help="目標時刻に合わせる半球。N または S。既定: N")
    parser.add_argument(
        "--opposite",
        choices=["after", "before", "nearest"],
        default="after",
        help="反対半球画像の選び方。既定: after",
    )
    parser.add_argument("--max-pair-minutes", type=float, default=70.0, help="N/Sペアとして許す最大時刻差 [分]")
    parser.add_argument("--out-dir", type=Path, default=HERE, help="PNG出力先。既定: このフォルダ")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-fetch", action="store_true", help="ネットワーク取得をせず、既存キャッシュだけを使う")
    parser.add_argument("--no-repo-cache", action="store_true", help="このリポジトリ内の raw データキャッシュを使わない")
    parser.add_argument("--base-url", action="append", default=[], help="追加のSSUSI公開ミラーURL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = parse_date(args.date)
    sat = normalize_satellite(args.satellite)
    target = parse_target_time(args.target_time)
    primary = normalize_hemisphere(args.primary_hemisphere)

    ssj_path = ensure_ssj_file(d, sat, allow_fetch=not args.no_fetch, include_repo_cache=not args.no_repo_cache)
    ssusi_files = ensure_ssusi_files(
        d,
        sat,
        allow_fetch=not args.no_fetch,
        include_repo_cache=not args.no_repo_cache,
        base_urls=args.base_url,
    )
    observations = discover_observations(ssusi_files, d, sat)
    if not observations:
        raise SystemExit(f"{d.isoformat()} {sat.upper()} のSSUSI観測時刻をNetCDFから読めませんでした。")

    pair = select_pair(observations, d, primary, target, args.opposite, args.max_pair_minutes)
    print(
        f"SSUSI pair: N {format_hhmm_colon(pair['N'].time_utc)} ({pair['N'].nc_path.name}), "
        f"S {format_hhmm_colon(pair['S'].time_utc)} ({pair['S'].nc_path.name})"
    )
    print(f"SSJ file: {ssj_path}")

    png_path = build_pair_figure(d, sat, ssj_path, pair, args.out_dir, args.dpi)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
