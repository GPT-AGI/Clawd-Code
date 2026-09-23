from __future__ import annotations

import unittest

from src.client import build_request
from src.protocol import normalize_item


class AcceptanceTests(unittest.TestCase):
    def test_normalize_contract(self) -> None:
        self.assertEqual(
            normalize_item({"id": 7, "tags": [" alpha ", "", "beta"]}),
            {"id": "7", "tags": ("alpha", "beta")},
        )

    def test_client_uses_normalized_representation(self) -> None:
        self.assertEqual(
            build_request({"id": "x", "tags": ["one"]}),
            {"item": {"id": "x", "tags": ("one",)}},
        )


if __name__ == "__main__":
    unittest.main()
