from __future__ import annotations

import unittest
from datetime import date, timedelta
from decimal import Decimal

from src.settlement import Invoice, aging_summary, allocate_payment


D = Decimal


class SettlementAcceptance(unittest.TestCase):
    def test_allocates_oldest_due_date_then_id(self) -> None:
        invoices = [
            Invoice("B", "c1", date(2025, 1, 1), D("20")),
            Invoice("A", "c1", date(2025, 1, 1), D("20")),
            Invoice("C", "c1", date(2025, 2, 1), D("20")),
        ]
        result = allocate_payment(invoices, "c1", D("25"))
        self.assertEqual(
            [(item.invoice_id, item.amount) for item in result.allocations],
            [("A", D("20.00")), ("B", D("5.00"))],
        )

    def test_filters_customer_and_non_open_invoices(self) -> None:
        invoices = [
            Invoice("other", "c2", date(2024, 1, 1), D("10")),
            Invoice("void", "c1", date(2024, 1, 2), D("10"), status="void"),
            Invoice("paid", "c1", date(2024, 1, 3), D("10"), paid=D("10")),
            Invoice("open", "c1", date(2024, 1, 4), D("10")),
        ]
        result = allocate_payment(invoices, "c1", D("10"))
        self.assertEqual([item.invoice_id for item in result.allocations], ["open"])

    def test_uses_outstanding_balance_and_preserves_excess(self) -> None:
        invoice = Invoice("part", "c1", date(2025, 1, 1), D("20"), paid=D("7.25"))
        result = allocate_payment([invoice], "c1", D("20"))
        self.assertEqual(result.allocations[0].amount, D("12.75"))
        self.assertEqual(result.unapplied, D("7.25"))

    def test_rounds_half_up_without_float_loss(self) -> None:
        invoice = Invoice("round", "c1", date(2025, 1, 1), D("1.005"))
        result = allocate_payment([invoice], "c1", D("1.005"))
        self.assertEqual(result.allocations[0].amount, D("1.01"))
        self.assertEqual(result.unapplied, D("0.00"))

    def test_rejects_invalid_payment_and_invoice_balances(self) -> None:
        with self.assertRaises(ValueError):
            allocate_payment([], "c1", D("0"))
        with self.assertRaises(ValueError):
            allocate_payment(
                [Invoice("bad", "c1", date.today(), D("5"), paid=D("6"))],
                "c1",
                D("1"),
            )

    def test_aging_excludes_closed_and_uses_outstanding(self) -> None:
        today = date(2025, 4, 1)
        invoices = [
            Invoice("open", "c1", today - timedelta(days=5), D("10"), paid=D("3")),
            Invoice("void", "c1", today - timedelta(days=5), D("50"), status="void"),
            Invoice("paid", "c1", today - timedelta(days=5), D("5"), paid=D("5")),
        ]
        self.assertEqual(aging_summary(invoices, today)["days_1_30"], D("7.00"))

    def test_aging_boundaries_are_stable(self) -> None:
        today = date(2025, 4, 1)
        invoices = [
            Invoice("future", "c", today + timedelta(days=1), D("1")),
            Invoice("d30", "c", today - timedelta(days=30), D("2")),
            Invoice("d60", "c", today - timedelta(days=60), D("3")),
            Invoice("d61", "c", today - timedelta(days=61), D("4")),
        ]
        self.assertEqual(
            aging_summary(invoices, today),
            {
                "current": D("1.00"),
                "days_1_30": D("2.00"),
                "days_31_60": D("3.00"),
                "days_over_60": D("4.00"),
            },
        )


if __name__ == "__main__":
    unittest.main()
