# Permission engine repair

Repair `src/policy.py`. The current authorization evaluator ignores role
inheritance, deny rules, wildcard semantics, and tenant placeholders. Preserve
the public dataclasses and `PolicyEngine.is_allowed` API while enforcing the
complete policy contract.
