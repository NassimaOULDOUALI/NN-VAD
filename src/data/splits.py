"""Speaker-disjoint train/val/test splits (no speaker leakage).

Whole speakers are assigned to a single split. Assignment is greedy on
utterance counts (speakers sorted by size, each placed in the split most under
its utterance quota) so the splits match the target ratios closely while
guaranteeing disjoint speakers. Deterministic given the seed.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import random
from collections import defaultdict
from .dataset import speaker_of


def speaker_disjoint_split(stems: List[str],
                           ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                           seed: int = 42) -> Dict[str, List[str]]:
    by_spk: Dict[str, List[str]] = defaultdict(list)
    for s in stems:
        by_spk[speaker_of(s)].append(s)

    total = len(stems)
    targets = {"train": ratios[0] * total,
               "val":   ratios[1] * total,
               "test":  ratios[2] * total}
    counts = {"train": 0, "val": 0, "test": 0}
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    speakers = sorted(by_spk, key=lambda k: (-len(by_spk[k]), k))  # big first
    rng = random.Random(seed)
    rng.shuffle(speakers)
    speakers.sort(key=lambda k: -len(by_spk[k]))                   # stable-ish

    for spk in speakers:
        # split with the largest remaining deficit (target - current)
        split = max(("train", "val", "test"),
                    key=lambda s: targets[s] - counts[s])
        out[split].extend(sorted(by_spk[spk]))
        counts[split] += len(by_spk[spk])

    for s in out:
        out[s].sort()
    return out


def split_summary(split: Dict[str, List[str]]) -> str:
    lines = []
    for name in ("train", "val", "test"):
        stems = split[name]
        spk = {speaker_of(s) for s in stems}
        lines.append(f"{name:5s}: {len(stems):4d} utt | {len(spk):2d} spk")
    return "\n".join(lines)