"""Vectorised per-second aggregation (numpy, CPU).

Rules from the brief:
  * an object is present in a second if detected in at least half of that second's frames
  * people = median of per-frame person counts, rounded down when halfway between two numbers
  * one record for every second from 0 up to the second of the last frame (gaps -> empty)
"""

from __future__ import annotations

import numpy as np

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]


def aggregate(secs: np.ndarray, counts: np.ndarray):
    """secs: (n,) int seconds, counts: (n, 5) int per-frame counts.
    Returns (present: (S, 5) bool, people: (S,) int) for S = max(secs) + 1."""
    if secs.size == 0:
        return np.zeros((0, len(TARGETS)), bool), np.zeros(0, np.int64)
    # sort by second, then by person count, so each second's people values are sorted
    order = np.lexsort((counts[:, 0], secs))
    s = secs[order]
    c = counts[order]
    uniq, start, cnt = np.unique(s, return_index=True, return_counts=True)

    frames_with = np.add.reduceat((c > 0).astype(np.int64), start, axis=0)   # (G, 5)
    present_g = frames_with * 2 >= cnt[:, None]

    p = c[:, 0].astype(np.int64)
    lo = start + (cnt - 1) // 2
    hi = start + cnt // 2
    people_g = (p[lo] + p[hi]) // 2                                          # floor on halves

    n_sec = int(s[-1]) + 1
    present = np.zeros((n_sec, len(TARGETS)), bool)
    people = np.zeros(n_sec, np.int64)
    present[uniq] = present_g
    people[uniq] = people_g
    return present, people


def records(video_id: str, present: np.ndarray, people: np.ndarray):
    names = np.array(TARGETS, dtype=object)
    for s in range(len(people)):
        yield {"video": video_id, "second": s, "objects": names[present[s]].tolist(), "people": int(people[s])}
