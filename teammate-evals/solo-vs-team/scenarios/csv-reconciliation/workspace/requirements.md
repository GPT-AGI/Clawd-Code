# Reconciliation requirements

Inputs are CSV strings. Bank headers are `reference,date,amount`; ledger headers
are `entry_id,reference,date,amount`. Dates use ISO `YYYY-MM-DD` and amounts are
`Decimal` values rounded to cents using `ROUND_HALF_UP`.

- Trim fields and normalize references to uppercase for matching. Preserve the
  original bank reference and ledger entry ID in `Match`.
- Parse rows independently. A malformed date, amount, missing bank reference, or
  missing ledger entry ID is excluded and adds `bank row N: ...` or
  `ledger row N: ...` to `errors`, where the header is row 1.
- Match valid rows in two deterministic phases, consuming each row at most once.
- Phase 1: normalized reference, equal amount, and dates no more than two days
  apart. If several ledger rows qualify, choose the smallest `(date, entry_id)`.
- Phase 2 for remaining rows: equal amount and dates no more than one day apart,
  ignoring reference. Match only when exactly one remaining ledger candidate
  exists; ambiguous candidates stay unmatched.
- Process bank rows in input order. Return unmatched bank references and
  unmatched ledger entry IDs in input order.
- Do not silently drop duplicate rows, zero or negative amounts, or parse errors.
  Zero and negative amounts are valid when both sides agree.
