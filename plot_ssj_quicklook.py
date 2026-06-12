#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
plot_ssj_quicklook.py — DMSP/SSJ クイックルック図の作成

指定した日付・衛星の SSJ データをダウンロード（済みなら再利用）して、
  1) 降下電子のエネルギースペクトログラム
  2) 全エネルギーフラックス（チャネル積分値）の時系列
  3) 衛星軌道（地理緯度・補正地磁気緯度の時系列）
  4) 極域から見た軌道トラック（MLT-磁気緯度ダイヤル）
を1枚のPNGに描く。

使い方:
  python plot_ssj_quicklook.py 2022-01-01 f18
  python plot_ssj_quicklook.py 2022-01-01 f18 --hours 6 9   # UTで6時〜9時だけ描く

依存: numpy, matplotlib （pip install numpy matplotlib）
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# 日本語フォント（mac: Hiragino / Windows: Yu Gothic, Meiryo）
plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]

import fetch_dmsp_ssj
import parse_dmsp_ssj
from parse_dmsp_ssj import CHANNEL_DELTA_EV, CHANNEL_ENERGY_EV


def find_or_fetch(date_str: str, sat: str) -> Path:
    """ローカルにファイルがあればそれを、なければNCEIから取得して返す。"""
    d = datetime.fromisoformat(date_str).date()
    files = fetch_dmsp_ssj.fetch_for_date(d)
    matches = [p for p in files if sat.lower() in p.name]
    if not matches:
        raise SystemExit(
            f"{date_str} の {sat.upper()} データが見つかりません。"
            f"取得できた衛星: {[p.name for p in files]}"
        )
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="DMSP/SSJ quicklook")
    ap.add_argument("date", help="YYYY-MM-DD (UT)")
    ap.add_argument("sat", nargs="?", default="f18", help="f15/f16/f17/f18")
    ap.add_argument("--hours", nargs=2, type=float, metavar=("H0", "H1"),
                    help="描画するUT時間帯（例: --hours 6 9）省略時は1日全部")
    args = ap.parse_args()

    path = find_or_fetch(args.date, args.sat)
    print(f"読み込み: {path}")
    day = parse_dmsp_ssj.parse(path)

    # ---- 時間帯の切り出し --------------------------------------------------
    t = day.times  # datetime64[ns], 1秒ごと
    ut_hours = (t - t[0]).astype("timedelta64[s]").astype(float) / 3600.0
    if args.hours:
        sel = (ut_hours >= args.hours[0]) & (ut_hours <= args.hours[1])
    else:
        sel = np.ones(t.shape, dtype=bool)
    t_sel = t[sel].astype("datetime64[ms]").astype("O")  # matplotlib用にdatetimeへ

    # ---- 図の組み立て ------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], hspace=0.35, wspace=0.25)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_flux = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_orb = fig.add_subplot(gs[2, 0], sharex=ax_spec)
    ax_dial = fig.add_subplot(gs[:, 1], projection="polar")

    # (1) 電子スペクトログラム
    # e_diff_energy_flux: (N, 20) 単位 eV/(cm^2 s sr eV)。チャネル11はSSJ5では欠測。
    je = day.e_diff_energy_flux[sel].T  # (20, Nsel)
    mesh = ax_spec.pcolormesh(
        t_sel, CHANNEL_ENERGY_EV, je,
        norm=LogNorm(vmin=1e5, vmax=1e10), cmap="viridis", shading="nearest",
    )
    ax_spec.set_yscale("log")
    ax_spec.set_ylabel("電子エネルギー [eV]")
    ax_spec.set_title(f"DMSP {day.sat} / {day.sensor}  {args.date}  降下電子")
    fig.colorbar(mesh, ax=ax_spec, label="JE [eV/(cm² s sr eV)]", pad=0.01)

    # (2) 全エネルギーフラックス（チャネル幅で積分）
    je_total = np.nansum(day.e_diff_energy_flux[sel] * CHANNEL_DELTA_EV[None, :], axis=1)
    ax_flux.semilogy(t_sel, np.where(je_total > 0, je_total, np.nan), lw=0.5, color="C1")
    ax_flux.set_ylabel("JE合計\n[eV/(cm² s sr)]")
    ax_flux.grid(alpha=0.3)

    # (3) 軌道：地理緯度と補正地磁気緯度
    ax_orb.plot(t_sel, day.glat[sel], lw=0.8, label="地理緯度")
    ax_orb.plot(t_sel, day.cgm_lat[sel], lw=0.8, label="補正地磁気緯度 (110 km)")
    ax_orb.set_ylabel("緯度 [deg]")
    ax_orb.set_xlabel("UT")
    ax_orb.legend(loc="upper right", fontsize=8)
    ax_orb.grid(alpha=0.3)
    ax_orb.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # (4) 極域ダイヤル（北半球, |cgm_lat|>50°のみ）— MLTを角度、余緯度を半径に
    polar_sel = sel & (day.cgm_lat > 50)
    theta = day.mlt_hours[polar_sel] / 24.0 * 2 * np.pi
    r = 90.0 - day.cgm_lat[polar_sel]
    jt = np.nansum(
        day.e_diff_energy_flux[polar_sel] * CHANNEL_DELTA_EV[None, :], axis=1
    )
    sc = ax_dial.scatter(theta, r, c=np.where(jt > 0, jt, np.nan),
                         s=2, norm=LogNorm(vmin=1e9, vmax=1e12), cmap="plasma")
    ax_dial.set_theta_zero_location("S")  # MLT=0(真夜中)を下に
    ax_dial.set_rlim(0, 40)
    ax_dial.set_rgrids([10, 20, 30, 40], ["80°", "70°", "60°", "50°"], fontsize=7)
    ax_dial.set_thetagrids([0, 90, 180, 270], ["00", "06", "12", "18"], fontsize=8)
    ax_dial.set_title("北半球 軌道トラック\n(磁気緯度-MLT, 色=JE合計)", fontsize=10)
    fig.colorbar(sc, ax=ax_dial, label="JE [eV/(cm² s sr)]", shrink=0.5, pad=0.1)

    out = Path(f"ssj_quicklook_{day.sat}_{args.date}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
