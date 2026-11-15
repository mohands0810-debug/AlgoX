"""AlgoX - Mars Colony Emergency Response Solver (Combined Best).

Architecture:
  * Active-queue + Drop-pool split: incidents are either in one of three crew
    queues (earning points) or in a drop pool (run at the end, earning ~0).
  * Multi-parameter advanced greedy dispatcher builds a strong starting solution
    via event simulation with dynamic urgency and automatic 0-point deferral.
  * Simulated annealing with EARLY-EXIT incremental evaluation:
      - Cached endT[] and cumulative score cum[] per queue.
      - After a move, recompute only until the end-time resyncs with the cache
        (unchanged suffix costs O(1) via cum[] lookup instead of O(k)).
  * 7 neighbourhood operators: swap, relocate, insert-from-drop, drop-active,
    cross-crew transfer, cross-crew swap, replace-active-with-dropped.
  * 4 parallel runs (diverse seeds + cooling), best result is kept.
  * Fully reproducible (fixed seed + fixed iters => same output every run).

Usage: python solve.py data.csv submission.json [iterations_per_run] [seed]
"""
import csv
import json
import math
import multiprocessing as mp
import random
import sys

CREWS = "EML"
CATN = {"STRUCTURAL": 0, "MEDICAL": 1, "LIFE_SUPPORT": 2}


class Problem:
    def __init__(self, path):
        rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(
                    (int(r["id"]), int(r["arrival"]), CATN[r["category"]],
                     int(r["severity"]), int(r["duration"]))
                )
        n = max(r[0] for r in rows)
        self.n = n
        self.arr = [0] * (n + 1)
        self.cat = [0] * (n + 1)
        self.sev = [0] * (n + 1)
        self.dur = [0] * (n + 1)
        self.win = [1] * (n + 1)
        for i, a, c, s, d in rows:
            self.arr[i] = a
            self.cat[i] = c
            self.sev[i] = s
            self.dur[i] = d
            self.win[i] = 60 - 4 * s
        self.ids = sorted(r[0] for r in rows)
        # d3[c][i] = effective duration of incident i handled by crew c
        self.d3 = [
            [(self.dur[i] if self.cat[i] == c else 2 * self.dur[i]) for i in range(n + 1)]
            for c in range(3)
        ]

    def pts(self, i, start):
        x = 1.0 - (start - self.arr[i]) / self.win[i]
        return self.sev[i] * x if x > 0 else 0.0


# ---------------------------------------------------------------------------
# Incremental evaluation with early-exit (the key performance driver)
# ---------------------------------------------------------------------------

def recompute(P, q, c, endT, cum, frm):
    """Rebuild cached end-times and cumulative scores from index frm onward."""
    arr, sev, win, dc = P.arr, P.sev, P.win, P.d3[c]
    del endT[frm:]
    del cum[frm:]
    t = endT[frm - 1] if frm else 0
    s = cum[frm - 1] if frm else 0.0
    for k in range(frm, len(q)):
        i = q[k]
        a = arr[i]
        st = a if a > t else t
        x = 1.0 - (st - a) / win[i]
        if x > 0:
            s += sev[i] * x
        t = st + dc[i]
        endT.append(t)
        cum.append(s)


def evaluate(P, q, c, endT, cum, frm, region, rest):
    """Score of queue q after replacing q[frm:rest] with region.
    Stops early when the finish time of a remaining element matches its cached
    value — the entire unchanged suffix is then added in O(1) via cum[].
    """
    arr, sev, win, dc = P.arr, P.sev, P.win, P.d3[c]
    t = endT[frm - 1] if frm else 0
    s = cum[frm - 1] if frm else 0.0
    for i in region:
        a = arr[i]
        st = a if a > t else t
        x = 1.0 - (st - a) / win[i]
        if x > 0:
            s += sev[i] * x
        t = st + dc[i]
    m = len(q)
    for j in range(rest, m):
        if endT[j] == t:                        # resync → suffix unchanged
            return s + (cum[m - 1] - cum[j])
        i = q[j]
        a = arr[i]
        st = a if a > t else t
        x = 1.0 - (st - a) / win[i]
        if x > 0:
            s += sev[i] * x
        t = st + dc[i]
    return s


# ---------------------------------------------------------------------------
# Advanced greedy dispatcher (builds initial active queues)
# ---------------------------------------------------------------------------

def advanced_greedy(P, alpha, beta, gamma, penalty):
    """Event-simulation dispatch with urgency decay and 0-point deferral."""
    n = len(P.ids)
    free = [0, 0, 0]
    pending = sorted(P.ids, key=lambda i: P.arr[i])
    k = 0
    waiting = []
    queues = [[], [], []]
    deferred = []
    t = 0
    allow_cross = penalty > 0

    while k < n or waiting:
        while k < n and P.arr[pending[k]] <= t:
            waiting.append(pending[k])
            k += 1

        # Defer expired incidents (points already 0) — keep crew capacity free
        still = []
        for i in waiting:
            if P.pts(i, t) <= 0:
                deferred.append(i)
            else:
                still.append(i)
        waiting = still

        progressed = False
        for c in sorted(range(3), key=lambda c: free[c]):
            if free[c] > t or not waiting:
                continue
            best, bv = None, -1e18
            for i in waiting:
                is_match = P.cat[i] == c
                if not is_match and not allow_cross:
                    continue
                dd = P.d3[c][i]
                p_now = P.pts(i, t)
                if p_now <= 0:
                    continue
                loss_rate = P.sev[i] / P.win[i]
                rem_win = max(0.1, P.win[i] - (t - P.arr[i]))
                v = (p_now + alpha * loss_rate * dd) / (dd ** beta)
                if gamma > 0:
                    v *= (1.0 / rem_win) ** gamma
                if not is_match:
                    v *= penalty
                if v > bv:
                    best, bv = i, v
            if best is None:
                continue
            waiting.remove(best)
            queues[c].append(best)
            free[c] = t + P.d3[c][best]
            progressed = True

        if not progressed:
            nxt = []
            if k < n:
                nxt.append(P.arr[pending[k]])
            nxt += [f for f in free if f > t]
            if nxt:
                t = min(nxt)
            else:
                t += 1 if waiting else 0
                if not waiting:
                    break
        else:
            if all(free[c] > t for c in range(3)) or not waiting:
                nxt = []
                if k < n:
                    nxt.append(P.arr[pending[k]])
                nxt += [f for f in free if f > t]
                if nxt:
                    t = min(nxt)

    for i in waiting + deferred:
        queues[P.cat[i]].append(i)
    return queues


def best_greedy(P):
    """Grid-search over greedy parameters, return the best initial queues."""
    best_q, best_s = None, -1.0
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        for beta in [0.8, 1.0, 1.2]:
            for gamma in [0.0, 0.1, 0.2]:
                for penalty in [0.3, 0.5, 0.6, 0.7, 0.0]:
                    q = advanced_greedy(P, alpha, beta, gamma, penalty)
                    # quick scoring via full scan (only done once per config)
                    s = 0.0
                    for c in range(3):
                        t = 0
                        for i in q[c]:
                            a = P.arr[i]
                            st = a if a > t else t
                            x = 1.0 - (st - a) / P.win[i]
                            if x > 0:
                                s += P.sev[i] * x
                            t = st + P.d3[c][i]
                    if s > best_s:
                        best_s, best_q = s, q
    return best_q, best_s


# ---------------------------------------------------------------------------
# Simulated annealing worker
# ---------------------------------------------------------------------------

def anneal_worker(args):
    """SA run: active queues + drop pool, 7 operators, early-exit evaluation."""
    csv_path, init_qs, init_dump, iters, seed, T0, T1 = args
    P = Problem(csv_path)

    rnd = random.Random(seed)
    rr = rnd.random
    ri = rnd.randrange
    exp = math.exp

    qs = [list(q) for q in init_qs]
    dump = list(init_dump)
    arr = P.arr

    endT = [[], [], []]
    cum = [[], [], []]
    for c in range(3):
        recompute(P, qs[c], c, endT[c], cum[c], 0)

    sc = [cum[c][-1] if cum[c] else 0.0 for c in range(3)]
    cur = sum(sc)
    best = cur
    bestq = ([list(q) for q in qs], list(dump))
    ratio = T1 / T0
    T = T0

    for it in range(iters):
        if it % 512 == 0:
            T = T0 * ratio ** (it / iters)
        r = rr()

        if r < 0.28:  # Operator 1: swap two incidents inside one queue
            c = ri(3)
            q = qs[c]
            m = len(q)
            if m < 2:
                continue
            a = ri(m)
            b = a + ri(25) - 12
            if b < 0:
                b = 0
            elif b >= m:
                b = m - 1
            if a == b:
                continue
            if a > b:
                a, b = b, a
            region = [q[b]] + q[a + 1:b] + [q[a]]
            ns = evaluate(P, q, c, endT[c], cum[c], a, region, b + 1)
            d = ns - sc[c]
            if d >= 0 or rr() < exp(d / T):
                q[a:b + 1] = region
                recompute(P, q, c, endT[c], cum[c], a)
                sc[c] = ns
                cur += d

        elif r < 0.42:  # Operator 2: relocate inside one queue
            c = ri(3)
            q = qs[c]
            m = len(q)
            if m < 2:
                continue
            a = ri(m)
            b = a + ri(61) - 30
            if b < 0:
                b = 0
            elif b >= m:
                b = m - 1
            if a == b:
                continue
            x = q[a]
            if a < b:
                frm, rest, region = a, b + 1, q[a + 1:b + 1] + [x]
            else:
                frm, rest, region = b, a + 1, [x] + q[b:a]
            ns = evaluate(P, q, c, endT[c], cum[c], frm, region, rest)
            d = ns - sc[c]
            if d >= 0 or rr() < exp(d / T):
                q[frm:rest] = region
                recompute(P, q, c, endT[c], cum[c], frm)
                sc[c] = ns
                cur += d

        elif r < 0.57:  # Operator 3: insert a dropped incident into active queue
            if not dump:
                continue
            di = ri(len(dump))
            x = dump[di]
            c = ri(3)
            q = qs[c]
            ax = arr[x]
            lo, hi = 0, len(q)
            while lo < hi:
                mid = (lo + hi) >> 1
                if arr[q[mid]] < ax:
                    lo = mid + 1
                else:
                    hi = mid
            b = lo + ri(13) - 6
            if b < 0:
                b = 0
            elif b > len(q):
                b = len(q)
            ns = evaluate(P, q, c, endT[c], cum[c], b, [x], b)
            d = ns - sc[c]
            if d >= 0 or rr() < exp(d / T):
                q.insert(b, x)
                recompute(P, q, c, endT[c], cum[c], b)
                sc[c] = ns
                cur += d
                dump[di] = dump[-1]
                dump.pop()

        elif r < 0.64:  # Operator 4: drop an active incident
            c = ri(3)
            q = qs[c]
            if not q:
                continue
            a = ri(len(q))
            ns = evaluate(P, q, c, endT[c], cum[c], a, [], a + 1)
            d = ns - sc[c]
            if d >= 0 or rr() < exp(d / T):
                dump.append(q.pop(a))
                recompute(P, q, c, endT[c], cum[c], a)
                sc[c] = ns
                cur += d

        elif r < 0.77:  # Operator 5: move incident to another crew
            c1 = ri(3)
            c2 = (c1 + 1 + ri(2)) % 3
            q1, q2 = qs[c1], qs[c2]
            if not q1:
                continue
            a = ri(len(q1))
            x = q1[a]
            ax = arr[x]
            lo, hi = 0, len(q2)
            while lo < hi:
                mid = (lo + hi) >> 1
                if arr[q2[mid]] < ax:
                    lo = mid + 1
                else:
                    hi = mid
            b = lo + ri(13) - 6
            if b < 0:
                b = 0
            elif b > len(q2):
                b = len(q2)
            n1 = evaluate(P, q1, c1, endT[c1], cum[c1], a, [], a + 1)
            n2 = evaluate(P, q2, c2, endT[c2], cum[c2], b, [x], b)
            d = n1 + n2 - sc[c1] - sc[c2]
            if d >= 0 or rr() < exp(d / T):
                q1.pop(a)
                q2.insert(b, x)
                recompute(P, q1, c1, endT[c1], cum[c1], a)
                recompute(P, q2, c2, endT[c2], cum[c2], b)
                sc[c1], sc[c2] = n1, n2
                cur += d

        elif r < 0.87:  # Operator 6: swap incidents between two crews
            c1 = ri(3)
            c2 = (c1 + 1 + ri(2)) % 3
            q1, q2 = qs[c1], qs[c2]
            if not q1 or not q2:
                continue
            a = ri(len(q1))
            x = q1[a]
            ax = arr[x]
            lo, hi = 0, len(q2)
            while lo < hi:
                mid = (lo + hi) >> 1
                if arr[q2[mid]] < ax:
                    lo = mid + 1
                else:
                    hi = mid
            b = lo + ri(13) - 6
            if b < 0:
                b = 0
            elif b >= len(q2):
                b = len(q2) - 1
            y = q2[b]
            n1 = evaluate(P, q1, c1, endT[c1], cum[c1], a, [y], a + 1)
            n2 = evaluate(P, q2, c2, endT[c2], cum[c2], b, [x], b + 1)
            d = n1 + n2 - sc[c1] - sc[c2]
            if d >= 0 or rr() < exp(d / T):
                q1[a] = y
                q2[b] = x
                recompute(P, q1, c1, endT[c1], cum[c1], a)
                recompute(P, q2, c2, endT[c2], cum[c2], b)
                sc[c1], sc[c2] = n1, n2
                cur += d

        else:  # Operator 7: replace active incident with nearest-arrival dropped one
            if not dump:
                continue
            c = ri(3)
            q = qs[c]
            if not q:
                continue
            a = ri(len(q))
            y = q[a]
            ay = arr[y]
            di = ri(len(dump))
            for _ in range(3):   # pick the dropped incident closest in arrival
                dj = ri(len(dump))
                if abs(arr[dump[dj]] - ay) < abs(arr[dump[di]] - ay):
                    di = dj
            x = dump[di]
            ns = evaluate(P, q, c, endT[c], cum[c], a, [x], a + 1)
            d = ns - sc[c]
            if d >= 0 or rr() < exp(d / T):
                q[a] = x
                recompute(P, q, c, endT[c], cum[c], a)
                sc[c] = ns
                cur += d
                dump[di] = y

        if cur > best + 1e-9:
            best = cur
            bestq = ([list(q) for q in qs], list(dump))

    return best, bestq


# ---------------------------------------------------------------------------
# Schedule output
# ---------------------------------------------------------------------------

def build_schedule(P, queues, dump):
    """Build the final JSON schedule: active queues first, then dropped incidents
    placed on the crew that can start them earliest (ties broken by shorter job)."""
    free = [0, 0, 0]
    rows = []
    for c in range(3):
        t = 0
        for i in queues[c]:
            s = max(P.arr[i], t)
            e = s + P.d3[c][i]
            rows.append({"id": i, "crew": CREWS[c], "start": s, "end": e})
            t = e
        free[c] = t
    for i in sorted(dump, key=lambda i: (P.arr[i], i)):
        c = min(range(3), key=lambda c: (max(P.arr[i], free[c]), P.d3[c][i]))
        s = max(P.arr[i], free[c])
        e = s + P.d3[c][i]
        rows.append({"id": i, "crew": CREWS[c], "start": s, "end": e})
        free[c] = e
    rows.sort(key=lambda r: (r["start"], r["id"]))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_path, out_path = sys.argv[1], sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2026

    P = Problem(csv_path)

    # Stage 1: Advanced multi-parameter greedy
    init_qs, init_score = best_greedy(P)
    print(f"greedy best: {init_score:.3f}", flush=True)

    # Stage 2: Partition into active queues and drop pool
    active = [[], [], []]
    drop = []
    for c in range(3):
        t = 0
        for i in init_qs[c]:
            s = P.arr[i] if P.arr[i] > t else t
            if s - P.arr[i] < P.win[i]:   # still earns positive points
                active[c].append(i)
                t = s + P.d3[c][i]
            else:
                drop.append(i)

    # Stage 3: 4 parallel SA runs with diverse cooling schedules
    configs = [
        (csv_path, active, drop, iters, seed,     2.0, 0.01),
        (csv_path, active, drop, iters, seed + 1, 1.0, 0.01),
        (csv_path, active, drop, iters, seed + 2, 4.0, 0.005),
        (csv_path, active, drop, iters, seed + 3, 2.0, 0.003),
    ]
    try:
        with mp.Pool(len(configs)) as pool:
            results = pool.map(anneal_worker, configs)
    except Exception:
        results = [anneal_worker(cfg) for cfg in configs]

    print("runs:", [round(r[0], 3) for r in results], flush=True)
    best_score, (best_qs, best_dump) = max(results, key=lambda r: r[0])
    print(f"annealed:    {best_score:.3f}", flush=True)

    # Stage 4: Write submission
    rows = build_schedule(P, best_qs, best_dump)
    with open(out_path, "w") as f:
        json.dump({"schedule": rows}, f)


if __name__ == "__main__":
    main()
