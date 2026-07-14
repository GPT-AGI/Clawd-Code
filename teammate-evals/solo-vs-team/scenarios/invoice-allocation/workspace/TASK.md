# Invoice settlement repair

Repair the invoice settlement module in `src/settlement.py`. Payment allocation
and aging reports currently disagree with the product rules and mishandle
several financial edge cases. Preserve the public dataclasses and function
signatures while making the complete acceptance suite pass.
