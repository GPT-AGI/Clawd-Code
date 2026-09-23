from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


CENT = Decimal("0.01")
MEMBER_RATE = Decimal("0.10")
FREE_SHIPPING_THRESHOLD = Decimal("100.00")


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class LineItem:
    unit_price: Decimal
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.unit_price < 0:
            raise ValueError("unit_price must be non-negative")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool) or self.quantity <= 0:
            raise ValueError("quantity must be a positive integer")

    @property
    def total(self) -> Decimal:
        return self.unit_price * self.quantity


@dataclass(frozen=True)
class Order:
    items: tuple[LineItem, ...]
    is_member: bool = False
    shipping_fee: Decimal = Decimal("10.00")
    store_credit: Decimal = Decimal("0.00")

    def __post_init__(self) -> None:
        if self.shipping_fee < 0:
            raise ValueError("shipping_fee must be non-negative")
        if self.store_credit < 0:
            raise ValueError("store_credit must be non-negative")

    @property
    def merchandise_subtotal(self) -> Decimal:
        return sum((item.total for item in self.items), start=Decimal("0.00"))

    def total(self) -> Decimal:
        subtotal = self.merchandise_subtotal
        shipping = Decimal("0.00") if subtotal >= FREE_SHIPPING_THRESHOLD else self.shipping_fee
        payable = subtotal + shipping
        if self.is_member:
            payable *= Decimal("1.00") - MEMBER_RATE
        payable = max(Decimal("0.00"), payable - self.store_credit)
        return _money(payable)
