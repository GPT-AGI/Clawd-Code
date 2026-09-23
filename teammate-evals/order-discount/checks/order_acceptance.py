from __future__ import annotations

import unittest
from decimal import Decimal

from src.order import LineItem, Order


D = Decimal


class OrderAcceptanceChecks(unittest.TestCase):
    def test_non_member_pays_merchandise_and_shipping(self) -> None:
        order = Order(items=(LineItem(D("40.00")),), shipping_fee=D("10.00"))
        self.assertEqual(order.total(), D("50.00"))

    def test_member_discount_does_not_reduce_shipping(self) -> None:
        order = Order(
            items=(LineItem(D("50.00")),),
            is_member=True,
            shipping_fee=D("10.00"),
        )
        self.assertEqual(order.total(), D("55.00"))

    def test_free_shipping_uses_pre_discount_subtotal(self) -> None:
        order = Order(
            items=(LineItem(D("50.00"), quantity=2),),
            is_member=True,
            shipping_fee=D("10.00"),
        )
        self.assertEqual(order.total(), D("90.00"))

    def test_store_credit_cannot_reduce_shipping(self) -> None:
        order = Order(
            items=(LineItem(D("20.00")),),
            is_member=True,
            shipping_fee=D("10.00"),
            store_credit=D("30.00"),
        )
        self.assertEqual(order.total(), D("10.00"))

    def test_rounding_uses_half_up(self) -> None:
        order = Order(
            items=(LineItem(D("10.05")),),
            is_member=True,
            shipping_fee=D("0.00"),
        )
        self.assertEqual(order.total(), D("9.05"))

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LineItem(D("-1.00"))
        with self.assertRaises(ValueError):
            LineItem(D("1.00"), quantity=0)
        with self.assertRaises(ValueError):
            Order(items=(), shipping_fee=D("-1.00"))
        with self.assertRaises(ValueError):
            Order(items=(), store_credit=D("-1.00"))


if __name__ == "__main__":
    unittest.main()
