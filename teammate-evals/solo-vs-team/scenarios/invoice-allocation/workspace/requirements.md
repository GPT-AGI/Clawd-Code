# Settlement requirements

All monetary values use `Decimal` and are rounded to cents with `ROUND_HALF_UP`.

## Payment allocation

- `allocate_payment` rejects non-positive payment amounts.
- Only open invoices for the requested customer are eligible. Paid and void
  invoices, invoices belonging to other customers, and invoices with no
  outstanding balance are ignored.
- Outstanding balance is `total - paid`, rounded to cents. Invalid invoices
  where paid is negative or exceeds total must raise `ValueError`.
- Allocate oldest due date first, using invoice ID as the deterministic tie
  breaker. Partial allocation is allowed.
- Return one `Allocation` per affected invoice and preserve any excess as
  `unapplied`; never manufacture or lose a cent.

## Aging summary

- `aging_summary` includes only positive outstanding balances on open invoices.
- Buckets are `current` for invoices due on or after `as_of`, `days_1_30`,
  `days_31_60`, and `days_over_60`.
- Boundary days 30 and 60 belong to the earlier bucket.
- Every bucket is returned even when its value is zero.
