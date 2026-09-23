from __future__ import annotations

import copy
import unittest

from src.config_migration import migrate_config, public_config


class ConfigMigrationAcceptance(unittest.TestCase):
    def test_migrates_v1_and_interpolates_multiple_variables(self) -> None:
        raw = {
            "endpoint": "https://${HOST}/${STAGE}",
            "token": "abc",
            "retries": 2,
            "features": {"label": "${STAGE}-worker"},
        }
        actual = migrate_config(raw, {"HOST": "api.example.com", "STAGE": "prod"})
        self.assertEqual(actual["service"]["url"], "https://api.example.com/prod")
        self.assertEqual(actual["features"]["label"], "prod-worker")
        self.assertEqual(actual["retry"]["max_attempts"], 2)

    def test_migrates_v2_nested_credentials(self) -> None:
        raw = {
            "version": 2,
            "service_url": "https://api.example.com",
            "credentials": {"api_token": "v2-secret"},
            "retry_count": 4,
            "features": {"fast": True},
        }
        actual = migrate_config(raw, {})
        self.assertEqual(actual["version"], 3)
        self.assertEqual(actual["service"]["auth"]["token"], "v2-secret")
        self.assertEqual(actual["retry"], {"max_attempts": 4})

    def test_v3_is_validated_and_deep_copied(self) -> None:
        raw = {
            "version": 3,
            "service": {"url": "https://api.example.com", "auth": {"token": "x"}},
            "retry": {"max_attempts": 0},
            "features": {"nested": {"enabled": True}},
        }
        actual = migrate_config(raw, {})
        actual["features"]["nested"]["enabled"] = False
        self.assertTrue(raw["features"]["nested"]["enabled"])

    def test_missing_environment_variable_is_explicit(self) -> None:
        raw = {"endpoint": "https://${MISSING}", "token": "x", "retries": 1}
        with self.assertRaisesRegex(ValueError, "MISSING"):
            migrate_config(raw, {})

    def test_rejects_invalid_url_token_retry_and_features(self) -> None:
        valid = {
            "version": 3,
            "service": {"url": "https://api.example.com", "auth": {"token": "x"}},
            "retry": {"max_attempts": 1},
            "features": {},
        }
        variants = []
        for path, value in (
            (("service", "url"), "http://api.example.com"),
            (("service", "auth", "token"), ""),
            (("retry", "max_attempts"), True),
            (("retry", "max_attempts"), 11),
            (("features",), []),
        ):
            item = copy.deepcopy(valid)
            target = item
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            variants.append(item)
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    migrate_config(variant, {})

    def test_rejects_unsupported_or_malformed_versions(self) -> None:
        with self.assertRaises(ValueError):
            migrate_config({"version": 99}, {})
        with self.assertRaises(ValueError):
            migrate_config({"version": 2, "credentials": "secret"}, {})

    def test_input_is_never_mutated(self) -> None:
        raw = {
            "version": 2,
            "service_url": "https://api.example.com",
            "credentials": {"api_token": "x"},
            "retry_count": 1,
            "features": {"items": ["${NAME}"]},
        }
        before = copy.deepcopy(raw)
        migrate_config(raw, {"NAME": "resolved"})
        self.assertEqual(raw, before)

    def test_public_config_redacts_recursively_and_is_detached(self) -> None:
        config = {
            "service": {"auth": {"token": "x", "password": "p"}},
            "items": [{"api_key": "k", "safe": {"secret": "s"}}],
        }
        public = public_config(config)
        self.assertEqual(public["service"]["auth"]["token"], "[REDACTED]")
        self.assertEqual(public["service"]["auth"]["password"], "[REDACTED]")
        self.assertEqual(public["items"][0]["api_key"], "[REDACTED]")
        self.assertEqual(public["items"][0]["safe"]["secret"], "[REDACTED]")
        public["items"][0]["safe"]["extra"] = True
        self.assertNotIn("extra", config["items"][0]["safe"])


if __name__ == "__main__":
    unittest.main()
