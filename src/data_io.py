"""Raw CSV validation, quarantine of malformed rows, and Parquet conversion.

The raw credit card panel in this copy of the competition data contains a small number
of structurally broken lines whose field count does not match the header. Reading with
pandas alone is unsafe: rows with too few fields are silently padded with nulls rather
than rejected. Every line is therefore checked against the header width before parsing,
and anything that does not match is written to a quarantine file with its original line
number so the rejection is auditable rather than invisible.
"""

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CARD_DTYPES = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "MONTHS_BALANCE": "int16",
    "SK_DPD": "int32",
    "SK_DPD_DEF": "int32",
    "CNT_DRAWINGS_CURRENT": "float32",
    "NAME_CONTRACT_STATUS": "category",
}


@dataclass
class ValidationReport:
    """Outcome of validating one raw CSV against its own header."""

    source: str
    header_fields: int
    total_lines: int
    data_rows_read: int
    rows_kept: int
    rows_quarantined: int
    quarantine_path: Optional[str]
    quarantined_line_numbers: List[int]

    def to_dict(self) -> Dict:
        return asdict(self)


def validate_and_split(
    source: Path, quarantine_path: Path, expect_quoted: bool = False
) -> ValidationReport:
    """Stream a CSV, keeping only rows whose field count matches the header.

    Files whose string columns legitimately contain commas must be read with the csv
    module rather than a naive split, which is what expect_quoted switches on.
    """
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    kept_path = quarantine_path.with_name(f"clean_{source.stem}.csv")

    bad_lines: List[int] = []
    kept = 0
    total = 0

    with open(source, newline="") as handle, \
            open(kept_path, "w", newline="") as clean_out, \
            open(quarantine_path, "w", newline="") as bad_out:
        reader = csv.reader(handle) if expect_quoted else handle
        writer = csv.writer(clean_out)
        bad_writer = csv.writer(bad_out)
        bad_writer.writerow(["source_line_number", "field_count", "raw_line"])

        if expect_quoted:
            header = next(reader)
            width = len(header)
            writer.writerow(header)
            total = 1
            for line_no, row in enumerate(reader, start=2):
                total += 1
                if len(row) == width:
                    writer.writerow(row)
                    kept += 1
                else:
                    bad_lines.append(line_no)
                    bad_writer.writerow([line_no, len(row), ",".join(row)])
        else:
            header_line = next(reader).rstrip("\n")
            width = header_line.count(",") + 1
            clean_out.write(header_line + "\n")
            total = 1
            for line_no, line in enumerate(reader, start=2):
                total += 1
                stripped = line.rstrip("\n")
                if stripped.count(",") + 1 == width:
                    clean_out.write(stripped + "\n")
                    kept += 1
                else:
                    bad_lines.append(line_no)
                    bad_writer.writerow([line_no, stripped.count(",") + 1, stripped])

    return ValidationReport(
        source=source.name,
        header_fields=width,
        total_lines=total,
        data_rows_read=total - 1,
        rows_kept=kept,
        rows_quarantined=len(bad_lines),
        quarantine_path=quarantine_path.name if bad_lines else None,
        quarantined_line_numbers=bad_lines[:100],
    )


def load_card_panel(clean_csv: Path) -> pd.DataFrame:
    """Read the validated card panel with memory efficient dtypes."""
    frame = pd.read_csv(clean_csv, dtype=CARD_DTYPES, engine="c")
    for column in frame.select_dtypes("float64").columns:
        frame[column] = frame[column].astype("float32")
    for column in frame.select_dtypes("int64").columns:
        frame[column] = pd.to_numeric(frame[column], downcast="integer")
    return frame


def profile(frame: pd.DataFrame, name: str) -> Dict:
    """Row counts, key cardinalities and null rates for the validation report."""
    nulls = (frame.isna().mean() * 100).round(3)
    summary = {
        "table": name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1e6, 1),
        "null_rate_pct": {k: float(v) for k, v in nulls[nulls > 0].items()},
    }
    for key in ("SK_ID_PREV", "SK_ID_CURR"):
        if key in frame.columns:
            summary[f"unique_{key}"] = int(frame[key].nunique())
    return summary


def write_report(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
