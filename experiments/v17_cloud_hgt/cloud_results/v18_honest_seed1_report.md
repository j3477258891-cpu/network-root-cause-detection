# V18 Honest Rank-Only Result

V18 removed the count loss from the decision path and corrected the V17 early-stopping leakage. Each model used an inner fold to select its epoch count, then refit on all four non-outer folds before one-time evaluation on the untouched outer fold.

Seed `20260803` produced honest outer-fold TP deltas:

`[-2, 0, -16, -3, +3]`, total `-18 TP`.

The remaining two seeds were stopped because this seed cannot satisfy the required `+24 TP` minimum. No submission CSV was generated. The online champion remains unchanged at `0.906324`.
