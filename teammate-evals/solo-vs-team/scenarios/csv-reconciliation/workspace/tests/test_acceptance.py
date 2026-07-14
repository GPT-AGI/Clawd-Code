from __future__ import annotations

import unittest

from src.reconcile import reconcile


class ReconciliationAcceptance(unittest.TestCase):
    def test_reference_matching_normalizes_and_is_not_positional(self) -> None:
        bank = "reference,date,amount\n ref-a ,2025-01-02,10.00\nREF-B,2025-01-03,20\n"
        ledger = (
            "entry_id,reference,date,amount\n"
            "L2,ref-b,2025-01-03,20.00\n"
            "L1,REF-A,2025-01-01,10\n"
        )
        report = reconcile(bank, ledger)
        self.assertEqual(
            [(match.bank_reference, match.ledger_entry_id, match.method) for match in report.matches],
            [("ref-a", "L1", "reference"), ("REF-B", "L2", "reference")],
        )

    def test_reference_candidate_tie_breaks_by_date_then_id(self) -> None:
        bank = "reference,date,amount\nX,2025-01-03,5\n"
        ledger = (
            "entry_id,reference,date,amount\n"
            "B,X,2025-01-02,5\n"
            "A,X,2025-01-02,5\n"
        )
        report = reconcile(bank, ledger)
        self.assertEqual(report.matches[0].ledger_entry_id, "A")
        self.assertEqual(report.unmatched_ledger, ("B",))

    def test_reference_dates_must_be_within_two_days(self) -> None:
        bank = "reference,date,amount\nX,2025-01-10,5\n"
        ledger = "entry_id,reference,date,amount\nL,X,2025-01-07,5\n"
        report = reconcile(bank, ledger)
        self.assertEqual(report.matches, ())
        self.assertEqual(report.unmatched_bank, ("X",))

    def test_unique_fallback_uses_amount_and_one_day_window(self) -> None:
        bank = "reference,date,amount\nBANK-X,2025-01-10,9.995\n"
        ledger = "entry_id,reference,date,amount\nL,OTHER,2025-01-11,10.00\n"
        report = reconcile(bank, ledger)
        self.assertEqual(report.matches[0].method, "amount_date")
        self.assertEqual(report.matches[0].amount, "10.00")

    def test_ambiguous_fallback_stays_unmatched(self) -> None:
        bank = "reference,date,amount\nBANK-X,2025-01-10,10\n"
        ledger = (
            "entry_id,reference,date,amount\n"
            "L1,A,2025-01-09,10\n"
            "L2,B,2025-01-11,10\n"
        )
        report = reconcile(bank, ledger)
        self.assertEqual(report.matches, ())
        self.assertEqual(report.unmatched_bank, ("BANK-X",))
        self.assertEqual(report.unmatched_ledger, ("L1", "L2"))

    def test_malformed_rows_are_reported_and_excluded(self) -> None:
        bank = (
            "reference,date,amount\n"
            ",2025-01-01,1\n"
            "B,not-a-date,2\n"
            "C,2025-01-01,nope\n"
            "OK,2025-01-01,3\n"
        )
        ledger = (
            "entry_id,reference,date,amount\n"
            ",OK,2025-01-01,3\n"
            "L,OK,2025-01-01,3\n"
        )
        report = reconcile(bank, ledger)
        self.assertEqual(len(report.errors), 4)
        self.assertTrue(report.errors[0].startswith("bank row 2:"))
        self.assertTrue(report.errors[-1].startswith("ledger row 2:"))
        self.assertEqual(report.matches[0].ledger_entry_id, "L")

    def test_negative_and_duplicate_rows_are_preserved(self) -> None:
        bank = "reference,date,amount\nR,2025-01-01,-2\nR,2025-01-01,-2\n"
        ledger = (
            "entry_id,reference,date,amount\n"
            "L1,R,2025-01-01,-2\n"
            "L2,R,2025-01-01,-2\n"
        )
        report = reconcile(bank, ledger)
        self.assertEqual([match.ledger_entry_id for match in report.matches], ["L1", "L2"])


if __name__ == "__main__":
    unittest.main()
