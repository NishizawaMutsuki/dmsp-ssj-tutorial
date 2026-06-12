# DMSP/SSJ 入門 — 軌道データと降下電子フラックスの取り方

DMSP衛星の**軌道（衛星位置）**と**降下電子・イオンのフラックス**は、
SSJセンサーのデータファイル1つに両方入っています。
このフォルダのスクリプトだけで「ダウンロード → 読み出し → 図にする」が完結します。

## 0. 必要なもの

**Pythonを入れていなくてもOK**な方法（推奨）: `uv` を1回だけインストールします。

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

以降は `python xxx.py ...` の代わりに `uv run xxx.py ...` と打つだけで、
Python本体も必要なライブラリ（numpy, matplotlib, netCDF4）も初回に自動で揃います。

すでにPython環境がある人は、従来どおりでも動きます:

```bash
pip install numpy matplotlib netCDF4
```

認証・アカウント登録は不要です（SSJはNOAA NCEI、SSUSIはNASA SPDF/CDAWebの公開アーカイブから取ります。
SSUSIの開発元JHU/APLのサイトはデータ配布を停止しているため、NASAのミラーを使います）。

## 1. まず動かしてみる

```bash
cd このフォルダ
uv run plot_ssj_quicklook.py 2022-01-01 f18
# （自分のPython環境を使う場合: python plot_ssj_quicklook.py 2022-01-01 f18）
```

これだけで以下が自動で行われます。

1. NOAA NCEI から 2022-01-01 のSSJファイルをダウンロード（`data/2022-01-01/` に保存）
2. バイナリを展開して秒値のフラックスと衛星位置を取り出す
3. `ssj_quicklook_F18_2022-01-01.png` を出力
   - 上段: 降下電子のエネルギースペクトログラム（30 eV〜30 keV）
   - 中段: 全エネルギーフラックス（チャネル積分）
   - 下段: 衛星の地理緯度・補正地磁気緯度 = **軌道**
   - 右: 北極上空から見た軌道トラック（磁気緯度-MLT、色がフラックス）

時間帯を絞るには `--hours 6 9`（UT 6時〜9時）のように指定します。

## 2. データはどこから来るか

```
https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/access/{衛星}/ssj/{年}/{月}/
```

- 1日1ファイル、gzip圧縮バイナリ（例: `j5f1822001.gz` = SSJ5, F18, 2022年, 通日001）
- 利用できる衛星と期間（`fetch_dmsp_ssj.py` の `SATS` に定義）:

| 衛星 | センサー | 期間 |
|------|---------|------|
| F15 | SSJ4 | 1999-12 〜 2009-03 |
| F16 | SSJ5 | 2003-10 〜（2024年以降は断続的）|
| F17 | SSJ5 | 2006-11 〜（同上）|
| F18 | SSJ5 | 2009-10 〜（同上）|

**2024年以降の注意**: NCEIアーカイブは2024年から断続的になり、
2024年は1月・5月、2025年は5月分のみ公開されています（2026-06時点）。
一方SSUSI（§6）の公開は2025年2月中旬までなので、
**SSJ+SSUSIの両方が揃う最新時期は2024年5月**です。

ブラウザで上のURLを開くと普通のディレクトリ一覧が見えるので、
「本当にここから取っているのか」を自分の目で確認できます。

## 3. 自分の解析コードからデータを使う

```python
import fetch_dmsp_ssj, parse_dmsp_ssj
from datetime import date

# ダウンロード（済みならキャッシュを再利用）
files = fetch_dmsp_ssj.fetch_for_date(date(2022, 1, 1))

# パース → 1秒ごとのデータが numpy 配列で得られる
day = parse_dmsp_ssj.parse(files[0])

day.times                # 時刻 (datetime64[ns])
day.glat, day.glon       # ★軌道: 地理緯度・経度 [deg]
day.altitude_km          # ★軌道: 高度 [km]（DMSPは約840-860 km）
day.cgm_lat, day.cgm_lon # 補正地磁気座標（高度110 km にマップした値）
day.mlt_hours            # 磁気地方時 [hour]
day.e_diff_energy_flux   # ★降下電子の微分エネルギーフラックス (N, 20ch)
                         #   単位: eV/(cm² s sr eV)
day.i_diff_energy_flux   # 降下イオン（同上）
day.e_counts             # 生カウント値（-1 は欠測）
```

チャネルの中心エネルギーは `parse_dmsp_ssj.CHANNEL_ENERGY_EV`
（30 keV → 30 eV の降順、20チャネル）にあります。

## 4. ハマりどころ（重要）

- **チャネル11は使えない（SSJ5）**: F16以降はチャネル11の場所に
  ステータス語が入っており、フラックスは NaN になります。19チャネルが実データです。
- **位置は1分分解能**: ファイルには衛星位置が1分に1点しか入っていません。
  パーサは同じ値を60秒に複製しています。衛星は約7.5 km/s（緯度0.07°/s）で
  動くので、秒精度の位置が必要なら時刻で線形内挿してください。
- **フラックスの単位**: `e_diff_energy_flux` は微分**エネルギー**フラックス
  eV/(cm² s sr eV) です。数フラックス (個/(cm² s sr eV)) が欲しければ
  各チャネルのエネルギーで割ります。
- **全エネルギーフラックスへの積分**: チャネル幅 `CHANNEL_DELTA_EV` を掛けて
  和を取ります（`plot_ssj_quicklook.py` の中段パネルが実例）。
- **欠測**: 生カウント 0 は「データなし」を意味し、counts=-1 / flux=NaN に
  変換されます。`np.nansum` / `np.nanmedian` 系で扱ってください。

## 5. ファイル形式の出典

バイナリ形式・チャネルエネルギー・幾何因子はすべて
**AFRL-RV-PS-TR-2014-0174**（"DMSP SSJ Simple Format"、Table 19, 23-28）
に基づいています。`parse_dmsp_ssj.py` の docstring に対応表があります。
論文に書くときのデータ出典は NOAA NCEI
(DMSP Space Weather Sensors archive) を引用してください。

## 6. SSUSI画像とSSJ粒子を1枚に重ねる

SSUSI（紫外オーロラ画像）とSSJ（粒子の直接観測）を合わせた6パネル図は、
次の1コマンドで作れます。

```bash
cd このフォルダ
python plot_ssusi_ssj_overlay.py 2022-12-25 f18
```

これだけで以下を自動で行います。

1. NOAA NCEI から 2022-12-25 の F18 SSJ ファイルを取得（`data/2022-12-25/`）
2. NASA SPDF (CDAWeb) またはローカルキャッシュから F18 SSUSI EDR-AURORA NetCDF を取得（`data/ssusi/20221225/`）
3. 北半球・南半球のSSUSI極投影にSSJ軌道を重ねる
4. 下段にイオン・電子の log flux スペクトログラムを描く
5. `ssusi_ssj_overlay_F18_20221225_N1241_S1332.png` のようなPNGを出力する

既定では、北半球画像のうち 12:00 UT に近いものを選び、その後に来る南半球画像を
70分以内で探します。別の時刻を見たい場合は、例えば次のようにします。

```bash
python plot_ssusi_ssj_overlay.py 2022-12-25 f18 --target-time 18:00
python plot_ssusi_ssj_overlay.py 2022-12-25 f18 --primary-hemisphere S --opposite before
```

SPDFのアーカイブはここからブラウザでも見られます（自分の目で確認できます）:
`https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf18/ssusi/data/edr-aurora/{年}/{通日}/`

ダウンロードに失敗した場合は、エラー文に「SSUSI NetCDFがありません」と出ます。
その場合は EDR-AURORA の `.NC` ファイルを `data/ssusi/YYYYMMDD/` に置くか、
既に持っているキャッシュを指定してください。

```bash
SSUSI_CACHE_ROOT=/path/to/data/raw/ssusi python plot_ssusi_ssj_overlay.py 2022-12-25 f18
```

このリポジトリ内で実行している場合は、`data/raw/ssusi/YYYYMMDD/` と
`data/raw/dmsp_ssj/YYYY-MM-DD/` があれば自動でチュートリアル側の `data/` にコピーします。

## 7. 関連: オーロラ画像と合わせたいとき

同じDMSPに載っているSSUSI（紫外オーロライメージャ）のNetCDFにも
衛星位置（`LATITUDE`/`LONGITUDE`/`ALTITUDE`、スキャンごと）が入っています。
ただしSSUSI内の「電子フラックス」(`ENERGY_FLUX_*_MAP`) はUV発光からの
**モデル推定値**であり、粒子の直接観測はSSJだけです。
オーロラの2次元像と軌道に沿った実測フラックスを重ねたい場合は
SSUSI + SSJ の組み合わせになります（§6のスクリプトがそれです。
データ配布: NASA SPDF https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/ 。
開発元JHU/APLのサイト https://ssusi.jhuapl.edu/ は解説資料の参照用）。
