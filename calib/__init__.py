"""Calibration package for the membership-space search-match-weight task.

Modules
-------
space       weight grid (each item 0..0.5, step 0.05, total == 1) as integer step units
ndcg        NDCG@10 with exponential gain (2**rel - 1) / log2(i + 2)
metrics     supply coverage / exposure gini / new-member exposure share / max item delta
constraints four-constraint gate plus rejection reasons
"""

__all__ = ["space", "ndcg", "metrics", "constraints"]
