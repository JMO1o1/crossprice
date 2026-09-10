"""One-time SPX/CBOE snapshot acquisition; standard library only, never run by tests.

Pure parsing functions are tested offline. The CSV contains all quotes at six
selected AM-settled standard expiries, including quotes later rejected. Prices
are index points, CBOE IV is retained verbatim in annual decimal units, counts
are source counts. No last price is substituted for a bid/ask midpoint.
The same payload supplies spot and its last-update time. Valuation uses that
time, interpreted in America/New_York, not the HTTP retrieval clock. The raw
payload timestamp is preserved without silently assigning it a timezone.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    "?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)
EASTERN = ZoneInfo("America/New_York")
TARGET_MONTHS = (1, 2, 3, 6, 9, 12)
COLUMNS = (
    "underlying",
    "option_symbol",
    "expiry",
    "settlement",
    "expiry_timestamp",
    "kind",
    "strike",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "volume",
    "open_interest",
    "cboe_iv",
    "last_trade_price",
    "last_trade_time",
    "spot",
    "spot_timestamp",
    "valuation_timestamp",
    "snapshot_timestamp_raw",
    "retrieved_at",
    "source_url",
)


def parse_symbol(symbol: str) -> tuple[str, date, str, float]:
    """Decode the OCC-style root, YYMMDD, C/P and strike in thousandths."""
    match = re.fullmatch(r"([A-Z]+)(\d{6})([CP])(\d{8})", symbol)
    if match is None:
        raise ValueError(f"invalid option symbol: {symbol}")
    root, expiry, kind, strike = match.groups()
    return (
        root,
        datetime.strptime(expiry, "%y%m%d").date(),
        ("call" if kind == "C" else "put"),
        int(strike) / 1000,
    )


def select_expiries(expiries: set[date], valuation: date) -> list[date]:
    """Nearest distinct available dates to fixed month offsets; at least 7 days."""
    available = {expiry for expiry in expiries if (expiry - valuation).days >= 7}
    if len(available) < len(TARGET_MONTHS):
        raise ValueError("fewer than six eligible standard expiries")
    selected = []
    for months in TARGET_MONTHS:
        month_index = valuation.year * 12 + valuation.month - 1 + months
        year, month = divmod(month_index, 12)
        month += 1
        target = date(year, month, min(valuation.day, calendar.monthrange(year, month)[1]))
        expiry = min(available, key=lambda item: (abs((item - target).days), item))
        selected.append(expiry)
        available.remove(expiry)
    return sorted(selected)


def normalize_chain(payload: dict[str, Any], retrieved_at: datetime) -> tuple[list[dict], dict]:
    """Select SPX (not SPXW) and preserve raw quote fields and same-payload spot."""
    if payload.get("symbol") != "_SPX":
        raise ValueError("expected the CBOE _SPX payload")
    data = payload["data"]
    spot = float(data["current_price"])
    if not math.isfinite(spot) or spot <= 0:
        raise ValueError("snapshot spot must be finite and positive")
    valuation = datetime.fromisoformat(data["last_trade_time"])
    if valuation.tzinfo is None:
        valuation = valuation.replace(tzinfo=EASTERN)
    valuation = valuation.astimezone(EASTERN)
    if retrieved_at.tzinfo is None or valuation > retrieved_at:
        raise ValueError("retrieval must be timezone aware and no earlier than spot time")
    decoded = [(quote, parse_symbol(quote["option"])) for quote in data["options"]]
    selected = select_expiries(
        {expiry for _, (root, expiry, _, _) in decoded if root == "SPX"}, valuation.date()
    )
    rows = []
    for quote, (root, expiry, kind, strike) in decoded:
        if root != "SPX" or expiry not in selected:
            continue
        rows.append(
            {
                "underlying": "SPX",
                "option_symbol": quote["option"],
                "expiry": expiry.isoformat(),
                "settlement": "AM",
                "expiry_timestamp": datetime.combine(
                    expiry, time(9, 30), tzinfo=EASTERN
                ).isoformat(),
                "kind": kind,
                "strike": strike,
                **{
                    name: quote.get(name)
                    for name in (
                        "bid",
                        "ask",
                        "bid_size",
                        "ask_size",
                        "volume",
                        "open_interest",
                        "last_trade_price",
                        "last_trade_time",
                    )
                },
                "cboe_iv": quote.get("iv"),
                "spot": spot,
                "spot_timestamp": data["last_trade_time"],
                "valuation_timestamp": valuation.isoformat(),
                "snapshot_timestamp_raw": payload["timestamp"],
                "retrieved_at": retrieved_at.astimezone(UTC).isoformat(),
                "source_url": CBOE_URL,
            }
        )
    rows.sort(key=lambda row: (row["expiry"], row["kind"], row["strike"], row["option_symbol"]))
    metadata = {
        "schema_version": 1,
        "underlying": "SPX",
        "source_url": CBOE_URL,
        "retrieved_at": retrieved_at.astimezone(UTC).isoformat(),
        "snapshot_timestamp_raw": payload["timestamp"],
        "spot": spot,
        "spot_timestamp_raw": data["last_trade_time"],
        "valuation_timestamp": valuation.isoformat(),
        "spot_age_at_retrieval_hours": (retrieved_at - valuation).total_seconds() / 3600,
        "valuation_convention": (
            "Same-payload spot last_trade_time interpreted as America/New_York. "
            "Not a synchronized bid/ask timestamp; pre-open/stale quotes remain a limitation."
        ),
        "settlement_convention": (
            "Standard SPX root only, European cash exercise; 09:30 America/New_York on OCC expiry "
            "as an AM fixing-time proxy. Actual component opening times "
            "and cash-payment lag omitted."
        ),
        "settlement_source": (
            "https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications"
        ),
        "target_months": TARGET_MONTHS,
        "selected_expiries": [item.isoformat() for item in selected],
        "payload_quotes": len(decoded),
        "snapshot_quotes": len(rows),
        "columns": COLUMNS,
    }
    return rows, metadata


def parse_treasury(content: bytes, valuation: date) -> list[dict]:
    """Latest daily par-yield row at/before valuation; quoted percent, not zero rates."""
    root = ET.fromstring(content)
    observations = [
        {node.tag.rsplit("}", 1)[-1]: node.text for node in properties}
        for properties in root.findall(".//{*}properties")
    ]
    eligible = [
        row for row in observations if date.fromisoformat(row["NEW_DATE"][:10]) <= valuation
    ]
    if not eligible:
        raise ValueError("no Treasury observation at or before valuation date")
    chosen = max(eligible, key=lambda row: row["NEW_DATE"])
    tenors = {
        "BC_1MONTH": 1 / 12,
        "BC_2MONTH": 2 / 12,
        "BC_3MONTH": 0.25,
        "BC_4MONTH": 1 / 3,
        "BC_6MONTH": 0.5,
        "BC_1YEAR": 1.0,
        "BC_2YEAR": 2.0,
    }
    rows = [
        {"date": chosen["NEW_DATE"][:10], "tenor_years": tenor, "par_yield_pct": float(chosen[key])}
        for key, tenor in tenors.items()
        if chosen.get(key)
    ]
    if len(rows) < 2 or any(not math.isfinite(row["par_yield_pct"]) for row in rows):
        raise ValueError("insufficient or invalid Treasury curve")
    return rows


def fetch_bytes(url: str) -> bytes:
    """Bounded one-time GET; network errors propagate and no credentials are used."""
    with urlopen(
        Request(url, headers={"User-Agent": "crossprice research snapshot"}), timeout=45
    ) as response:
        return response.read()


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    """Write once, refusing to replace a previously captured snapshot."""
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Fetch quotes and rates once and save provenance for a manually sourced yield proxy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--external-dividend-yield", type=float, required=True)
    parser.add_argument("--external-dividend-date", type=date.fromisoformat, required=True)
    parser.add_argument("--external-dividend-url", required=True)
    parser.add_argument("--external-dividend-note", required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.external_dividend_yield) or args.external_dividend_yield < 0:
        parser.error("external dividend yield must be finite and nonnegative")
    raw_chain = fetch_bytes(CBOE_URL)
    retrieved = datetime.now(UTC)
    rows, metadata = normalize_chain(json.loads(raw_chain), retrieved)
    valuation = date.fromisoformat(metadata["valuation_timestamp"][:10])
    if args.external_dividend_date > valuation:
        parser.error("external dividend observation must not be after valuation")
    treasury_url = TREASURY_URL.format(year=valuation.year)
    raw_treasury = fetch_bytes(treasury_url)
    rates = parse_treasury(raw_treasury, valuation)
    basename = f"SPX_{retrieved.astimezone(EASTERN):%Y%m%d}"
    metadata.update(
        {
            "chain_payload_sha256": hashlib.sha256(raw_chain).hexdigest(),
            "treasury_payload_sha256": hashlib.sha256(raw_treasury).hexdigest(),
            "treasury_url": treasury_url,
            "treasury_date": rates[0]["date"],
            "treasury_csv": f"{basename}_treasury.csv",
            "rate_convention": (
                "Linear interpolation of par percent; r=2*log1p(y/2), a zero-rate proxy."
            ),
            "external_dividend_yield": args.external_dividend_yield,
            "external_dividend_date": args.external_dividend_date.isoformat(),
            "external_dividend_url": args.external_dividend_url,
            "external_dividend_note": args.external_dividend_note,
            "dividend_convention": (
                "External trailing cash yield used numerically as continuous q, sensitivity only."
            ),
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.out / f"{basename}{suffix}" for suffix in (".csv", ".json", "_treasury.csv")]
    if any(path.exists() for path in paths):
        parser.error("snapshot paths already exist; use a new output directory")
    write_csv(paths[0], rows, COLUMNS)
    write_csv(paths[2], rates, ("date", "tenor_years", "par_yield_pct"))
    with paths[1].open("x", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
