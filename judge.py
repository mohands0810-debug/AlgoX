"""Local judge replica: validates submission.json against the rules and prints the Colony Safety Score.
Usage: python judge.py data.csv submission.json
"""
import csv
import json
import sys

CREW_OF = {"STRUCTURAL": "E", "MEDICAL": "M", "LIFE_SUPPORT": "L"}


def judge(csv_path, json_path):
    inc = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            inc[int(r["id"])] = (int(r["arrival"]), r["category"], int(r["severity"]), int(r["duration"]))
    with open(json_path) as f:
        sched = json.load(f)["schedule"]

    seen = set()
    last_end = {"E": -1, "M": -1, "L": -1}
    last_start = -1
    total = 0.0
    for row in sched:
        i, crew, s, e = row["id"], row["crew"], row["start"], row["end"]
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (i, s, e)):
            return 0.0, "non-integer field"
        if i not in inc:
            return 0.0, f"unknown id {i}"
        if i in seen:
            return 0.0, f"id {i} repeated"
        seen.add(i)
        if crew not in ("E", "M", "L"):
            return 0.0, f"bad crew {crew}"
        arr, cat, sev, dur = inc[i]
        if s < arr:
            return 0.0, f"id {i} starts before arrival"
        need = dur if CREW_OF[cat] == crew else 2 * dur
        if e != s + need:
            return 0.0, f"id {i} wrong end (expected {s + need}, got {e})"
        if s < last_start:
            return 0.0, "not sorted by start"
        last_start = s
        if s < last_end[crew]:
            return 0.0, f"crew {crew} overlap at id {i}"
        last_end[crew] = e
        window = 60 - 4 * sev
        total += sev * max(0.0, 1 - (s - arr) / window)
    if len(seen) != len(inc):
        return 0.0, f"missing {len(inc) - len(seen)} incidents"
    return total, "valid"


if __name__ == "__main__":
    score, msg = judge(sys.argv[1], sys.argv[2])
    print(f"{msg}  score={score:.4f}")
