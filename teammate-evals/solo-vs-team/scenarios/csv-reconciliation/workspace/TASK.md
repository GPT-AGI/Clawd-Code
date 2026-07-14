# CSV reconciliation repair

Repair `src/reconcile.py`. The existing implementation zips rows by position and
uses binary floating point, producing false matches and losing malformed-row
diagnostics. Preserve the public dataclasses and `reconcile` signature while
implementing the deterministic matching rules.
