#!/usr/bin/env python3
"""
fetch_ssusi.py — DMSP/SSUSI EDR-AURORA NetCDF の取得

指定した日付・衛星の SSUSI EDR-AURORA NetCDF をこのチュートリアル
フォルダの data/ssusi/YYYYMMDD/ に保存する。

まずローカルキャッシュを使い、次に環境変数 SSUSI_CACHE_ROOT または
このリポジトリ内の data/raw/ssusi を read-through cache として使う。
最後に NASA SPDF (CDAWeb) の公開アーカイブからダウンロードする。
注: JHU/APL の SSUSI サイトはデータ配布を停止しているため、
SPDF ミラー https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/ を既定にしている。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
ROOT = HERE / "data" / "ssusi"
# NASA SPDF (CDAWeb) の DMSP/SSUSI ミラー。テンプレート形式（{sat} 等は
# format_template で展開）。JHU/APL 本家は配布停止のため使わない。
DEFAULT_PUBLIC_BASES = [
    "https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmsp{sat}/ssusi/data/edr-aurora/{YYYY}/{DDD}",
]
USER_AGENT = "dmsp-ssusi-tutorial/1.0"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class FetchResult:
    files: list[Path]
    messages: list[str]


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
    if sat not in {"f16", "f17", "f18", "f19"}:
        raise SystemExit(f"SSUSI衛星は f16/f17/f18/f19 の形式で指定してください: {text}")
    return sat


def date_text(d: date) -> str:
    return d.strftime("%Y%m%d")


def file_matches(path_or_name: str | Path, d: date, sat: str) -> bool:
    """APL命名（PS.APL_..._GP.F18-SSUSI_..._DD.YYYYMMDD_...）と
    SPDF命名（dmspf18_ssusi_edr-aurora_YYYYDDDT...）の両方を受け付ける。"""
    name = Path(path_or_name).name
    upper = name.upper()
    if not upper.endswith(".NC"):
        return False
    # APL distribution naming
    if (
        f"_GP.{sat.upper()}-SSUSI_" in upper
        and "_PA.APL-EDR-AURORA_" in upper
        and f"_DD.{date_text(d)}_" in upper
    ):
        return True
    # SPDF (CDAWeb) mirror naming
    lower = name.lower()
    doy_tag = f"{d.year:04d}{d.timetuple().tm_yday:03d}t"
    return (
        lower.startswith(f"dmsp{sat.lower()}_ssusi_edr-aurora_")
        and doy_tag in lower
    )


def local_dir(d: date, root: Path = ROOT) -> Path:
    return root / date_text(d)


def existing_files(d: date, sat: str, root: Path = ROOT) -> list[Path]:
    folder = local_dir(d, root)
    if not folder.exists():
        return []
    candidates = [p for p in folder.iterdir() if p.suffix.upper() == ".NC"]
    return sorted(path for path in candidates if file_matches(path, d, sat) and path.stat().st_size > 0)


def candidate_cache_roots(include_repo_cache: bool) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("SSUSI_CACHE_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    if include_repo_cache:
        for parent in HERE.parents:
            candidate = parent / "data" / "raw" / "ssusi"
            if candidate.exists():
                roots.append(candidate)
                break

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen and resolved != ROOT.resolve():
            seen.add(resolved)
            deduped.append(root)
    return deduped


def copy_from_cache(d: date, sat: str, cache_root: Path, out_root: Path = ROOT) -> list[Path]:
    source_dir = local_dir(d, cache_root)
    if not source_dir.exists():
        return []

    out_dir = local_dir(d, out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    sources = sorted(p for p in source_dir.iterdir() if p.suffix.upper() == ".NC")
    for source in sources:
        if not file_matches(source, d, sat) or source.stat().st_size == 0:
            continue
        target = out_dir / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        copied.append(target)
    return copied


def public_bases(extra_bases: Iterable[str] | None = None) -> list[str]:
    bases: list[str] = []
    env = os.environ.get("SSUSI_PUBLIC_BASES")
    if env:
        bases.extend(item.strip() for item in env.split(",") if item.strip())
    if extra_bases:
        bases.extend(extra_bases)
    bases.extend(DEFAULT_PUBLIC_BASES)

    deduped: list[str] = []
    seen: set[str] = set()
    for base in bases:
        normalized = base.rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def format_template(text: str, d: date, sat: str) -> str:
    mapping = {
        "YYYY": f"{d.year:04d}",
        "yyyy": f"{d.year:04d}",
        "MM": f"{d.month:02d}",
        "mm": f"{d.month:02d}",
        "DD": f"{d.day:02d}",
        "dd": f"{d.day:02d}",
        "DDD": f"{d.timetuple().tm_yday:03d}",
        "doy": f"{d.timetuple().tm_yday:03d}",
        "date": date_text(d),
        "sat": sat.lower(),
        "SAT": sat.upper(),
    }
    return text.format(**mapping)


def listing_urls_for_base(base: str, d: date, sat: str) -> list[str]:
    if "{" in base and "}" in base:
        return [format_template(base, d, sat).rstrip("/") + "/"]

    year = f"{d.year:04d}"
    month = f"{d.month:02d}"
    day = f"{d.day:02d}"
    doy = f"{d.timetuple().tm_yday:03d}"
    ymd = date_text(d)
    sat_l = sat.lower()
    sat_u = sat.upper()
    patterns = [
        f"{base}/{ymd}/",
        f"{base}/{year}/{ymd}/",
        f"{base}/{year}/{month}/{day}/",
        f"{base}/{year}/{doy}/",
        f"{base}/{sat_l}/{year}/{month}/{day}/",
        f"{base}/{sat_l}/{year}/{doy}/",
        f"{base}/{sat_l}/edr-aurora/{year}/{doy}/",
        f"{base}/{sat_u}/EDR-AURORA/{year}/{doy}/",
    ]
    return patterns


def read_url_text(url: str, timeout: float = 25.0) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if "text" not in content_type and b"<html" not in data[:1024].lower():
        return ""
    return data.decode("utf-8", errors="replace")


def links_from_listing(url: str) -> list[str]:
    html = read_url_text(url)
    parser = LinkParser()
    parser.feed(html)
    return [urljoin(url, link) for link in parser.links]


def download_nc(url: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url.split("?", 1)[0]).name
    if not filename.upper().endswith(".NC"):
        return None
    target = out_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return target

    request = Request(url, headers={"User-Agent": USER_AGENT})
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(request, timeout=60.0) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
        if "html" in content_type.lower() or data[:1024].lstrip().lower().startswith(b"<!doctype html"):
            tmp.unlink(missing_ok=True)
            return None
        if not (data.startswith(b"CDF") or data.startswith(b"\x89HDF")):
            tmp.unlink(missing_ok=True)
            return None
        tmp.write_bytes(data)
        tmp.replace(target)
        return target
    except (HTTPError, URLError, TimeoutError, OSError):
        tmp.unlink(missing_ok=True)
        return None


def fetch_from_public(d: date, sat: str, extra_bases: Iterable[str] | None = None) -> FetchResult:
    out_dir = local_dir(d)
    messages: list[str] = []
    found: list[Path] = []
    tried = 0

    for base in public_bases(extra_bases):
        for listing_url in listing_urls_for_base(base, d, sat):
            tried += 1
            try:
                links = links_from_listing(listing_url)
            except HTTPError as exc:
                if exc.code not in {403, 404}:
                    messages.append(f"  [skip] {listing_url}: HTTP {exc.code}")
                continue
            except (URLError, TimeoutError, OSError) as exc:
                messages.append(f"  [skip] {listing_url}: {exc}")
                continue

            matches = [link for link in links if file_matches(link, d, sat)]
            if not matches:
                continue
            for link in matches:
                local = download_nc(link, out_dir)
                if local is not None:
                    found.append(local)
                    messages.append(f"  [fetched] {local.name}")
            if found:
                return FetchResult(sorted(set(found)), messages)

    if not found:
        messages.append(
            f"  [not found] NASA SPDF公開アーカイブで {d.isoformat()} {sat.upper()} の"
            f" EDR-AURORA NetCDF を取得できませんでした（探索URL {tried} 件）。"
        )
    return FetchResult([], messages)


def fetch_for_date(
    d: date,
    sat: str,
    *,
    include_repo_cache: bool = True,
    extra_bases: Iterable[str] | None = None,
) -> list[Path]:
    sat = normalize_satellite(sat)
    cached = existing_files(d, sat)
    if cached:
        return cached

    for cache_root in candidate_cache_roots(include_repo_cache):
        copied = copy_from_cache(d, sat, cache_root)
        if copied:
            print(f"  [cache] {cache_root} から {len(copied)} 個コピーしました")
            return existing_files(d, sat)

    result = fetch_from_public(d, sat, extra_bases=extra_bases)
    for message in result.messages:
        print(message)
    return existing_files(d, sat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="UTC日付: YYYY-MM-DD")
    parser.add_argument("satellite", help="衛星: f16/f17/f18/f19")
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help=(
            "追加のSSUSI公開ミラーURL。{YYYY}/{MM}/{DD}/{DDD}/{date}/{sat}/{SAT} "
            "を含むテンプレートも指定できます。"
        ),
    )
    parser.add_argument(
        "--no-repo-cache",
        action="store_true",
        help="このリポジトリの data/raw/ssusi キャッシュを使わない。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = parse_date(args.date)
    sat = normalize_satellite(args.satellite)
    files = fetch_for_date(
        d,
        sat,
        include_repo_cache=not args.no_repo_cache,
        extra_bases=args.base_url,
    )
    if not files:
        print(
            f"{d.isoformat()} {sat.upper()} のSSUSI NetCDFがありません。\n"
            "NASA SPDFアーカイブに接続できない場合は、NetCDFを "
            f"{local_dir(d)} に手動で置くか、SSUSI_CACHE_ROOT を指定してください。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"{d.isoformat()} {sat.upper()}: {len(files)} files -> {[p.name for p in files]}")


if __name__ == "__main__":
    main()
