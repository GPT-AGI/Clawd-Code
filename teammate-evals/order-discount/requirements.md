# Order Pricing Requirements

The calculator works with monetary values represented by `Decimal`.

1. The merchandise subtotal is the sum of each unit price multiplied by its
   quantity.
2. Members receive a 10% discount on merchandise only.
3. Shipping is never discounted.
4. Orders with a merchandise subtotal of at least 100 receive free shipping.
   Eligibility is determined from the original subtotal before member discount
   or store credit.
5. Store credit is applied after the member discount and only to merchandise.
   Credit may reduce merchandise to zero but must never reduce shipping.
6. The final total is rounded to two decimal places using `ROUND_HALF_UP`.
7. Quantity must be a positive integer. Unit price, shipping fee, and store
   credit must be non-negative.
