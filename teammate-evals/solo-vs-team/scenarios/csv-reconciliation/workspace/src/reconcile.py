from __future__ import annotations

import csv
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    bank_reference: str
    ledger_entry_id: str
    amount: str
    method: str


@dataclass(frozen=True)
class ReconciliationReport:
    matches: tuple[Match, ...]
    unmatched_bank: tuple[str, ...]
    unmatched_ledger: tuple[str, ...]
    errors: tuple[str, ...]


def reconcile(bank_csv: str, ledger_csv: str) -> ReconciliationReport:
    bank_rows = list(csv.DictReader(io.StringIO(bank_csv)))
    ledger_rows = list(csv.DictReader(io.StringIO(ledger_csv)))
    matches: list[Match] = []
    for bank, ledger in zip(bank_rows, ledger_rows):
        if float(bank["amount"]) == float(ledger["amount"]):
            matches.append(
                Match(bank["reference"], ledger["entry_id"], bank["amount"], "position")
            )
    matched = len(matches)
    return ReconciliationReport(
        tuple(matches),
        tuple(row["reference"] for row in bank_rows[matched:]),
        tuple(row["entry_id"] for row in ledger_rows[matched:]),
        (),
    )
