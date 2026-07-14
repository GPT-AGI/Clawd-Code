from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


CENT = Decimal("0.01")


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    customer_id: str
    due_on: date
    total: Decimal
    paid: Decimal = Decimal("0.00")
    status: str = "open"


@dataclass(frozen=True)
class Allocation:
    invoice_id: str
    amount: Decimal


@dataclass(frozen=True)
class PaymentResult:
    allocations: tuple[Allocation, ...]
    unapplied: Decimal


def allocate_payment(
    invoices: list[Invoice], customer_id: str, amount: Decimal
) -> PaymentResult:
    remaining = round(float(amount), 2)
    eligible = sorted(
        (invoice for invoice in invoices if invoice.status != "paid"),
        key=lambda invoice: invoice.due_on,
        reverse=True,
    )
    allocations: list[Allocation] = []
    for invoice in eligible:
        if remaining <= 0:
            break
        applied = min(remaining, float(invoice.total))
        allocations.append(Allocation(invoice.invoice_id, Decimal(str(applied))))
        remaining -= applied
    return PaymentResult(tuple(allocations), Decimal(str(remaining)))


def aging_summary(invoices: list[Invoice], as_of: date) -> dict[str, Decimal]:
    buckets = {
        "current": Decimal("0.00"),
        "days_1_30": Decimal("0.00"),
        "days_31_60": Decimal("0.00"),
        "days_over_60": Decimal("0.00"),
    }
    for invoice in invoices:
        balance = invoice.total
        days = (as_of - invoice.due_on).days
        if days < 0:
            key = "current"
        elif days < 30:
            key = "days_1_30"
        elif days < 60:
            key = "days_31_60"
        else:
            key = "days_over_60"
        buckets[key] += balance
    return buckets
