from __future__ import annotations

import unittest

from src.policy import PolicyEngine, Role, Rule


class PermissionPolicyAcceptance(unittest.TestCase):
    def test_exact_allow_and_default_deny(self) -> None:
        engine = PolicyEngine({"reader": Role("reader", [Rule("allow", "read", "docs/one")])})
        self.assertTrue(engine.is_allowed(["reader"], "read", "docs/one"))
        self.assertFalse(engine.is_allowed(["reader"], "write", "docs/one"))

    def test_shell_wildcards_are_case_sensitive(self) -> None:
        engine = PolicyEngine({"reader": Role("reader", [Rule("allow", "read*", "docs/*")])})
        self.assertTrue(engine.is_allowed(["reader"], "read:meta", "docs/one"))
        self.assertFalse(engine.is_allowed(["reader"], "Read:meta", "docs/one"))

    def test_transitive_inheritance(self) -> None:
        roles = {
            "base": Role("base", [Rule("allow", "read", "docs/*")]),
            "editor": Role("editor", [Rule("allow", "write", "docs/*")], ("base",)),
            "admin": Role("admin", [], ("editor",)),
        }
        engine = PolicyEngine(roles)
        self.assertTrue(engine.is_allowed(["admin"], "read", "docs/a"))
        self.assertTrue(engine.is_allowed(["admin"], "write", "docs/a"))

    def test_deny_overrides_allow_across_roles(self) -> None:
        roles = {
            "writer": Role("writer", [Rule("allow", "write", "docs/*")]),
            "suspended": Role("suspended", [Rule("deny", "*", "docs/locked")]),
        }
        engine = PolicyEngine(roles)
        self.assertFalse(engine.is_allowed(["writer", "suspended"], "write", "docs/locked"))
        self.assertTrue(engine.is_allowed(["writer", "suspended"], "write", "docs/open"))

    def test_tenant_placeholder_is_required_and_literal(self) -> None:
        engine = PolicyEngine(
            {"member": Role("member", [Rule("allow", "read", "tenant/{tenant}/*")])}
        )
        self.assertTrue(
            engine.is_allowed(["member"], "read", "tenant/acme/report", {"tenant": "acme"})
        )
        self.assertFalse(engine.is_allowed(["member"], "read", "tenant/acme/report", {}))
        self.assertFalse(
            engine.is_allowed(["member"], "read", "tenant/anything/report", {"tenant": "*"})
        )

    def test_unknown_parent_and_cycles_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolicyEngine({"child": Role("child", inherits=("missing",))})
        with self.assertRaises(ValueError):
            PolicyEngine(
                {
                    "a": Role("a", inherits=("b",)),
                    "b": Role("b", inherits=("a",)),
                }
            )

    def test_invalid_effect_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolicyEngine({"bad": Role("bad", [Rule("maybe", "read", "*")])})

    def test_engine_detaches_caller_owned_collections(self) -> None:
        rules = [Rule("allow", "read", "docs/*")]
        roles = {"reader": Role("reader", rules)}
        engine = PolicyEngine(roles)
        rules.clear()
        roles.clear()
        self.assertTrue(engine.is_allowed(["reader"], "read", "docs/a"))


if __name__ == "__main__":
    unittest.main()
