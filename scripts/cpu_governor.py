#!/usr/bin/env python3
"""Keep experiment CPU use bounded.

* kills leaked evaluation subprocesses (orphaned `tmp*.py` candidates whose
  evaluation already timed out but which keep spinning);
* keeps at most MAX_ACTIVE `skydiscover.cli` runs un-paused (SIGSTOP/SIGCONT),
  pausing a run only while it has no evaluation child, so no evaluation is
  cut short by the pause.

Usage: python scripts/cpu_governor.py [MAX_ACTIVE=6] [ORPHAN_MAX_AGE_S=480]
"""
import os, re, signal, subprocess, sys, time

MAX_ACTIVE = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ORPHAN_AGE = int(sys.argv[2]) if len(sys.argv) > 2 else 480
TMP = re.compile(r"/T/tmp\w+\.py")


def etime_s(s):
    days = 0
    if "-" in s:
        d, s = s.split("-"); days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def ps():
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,stat=,etime=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        p = line.split(None, 4)
        if len(p) == 5:
            rows.append((int(p[0]), int(p[1]), p[2], etime_s(p[3]), p[4]))
    return rows


def descendants(rows, pid):
    kids = [r[0] for r in rows if r[1] == pid]
    out = list(kids)
    for k in kids:
        out += descendants(rows, k)
    return out


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


paused = set()
while True:
    rows = ps()
    for pid, ppid, stat, age, cmd in rows:
        if ppid == 1 and TMP.search(cmd) and age > ORPHAN_AGE:
            try:
                os.kill(pid, signal.SIGKILL); log(f"killed orphan evaluator {pid} (age {age}s)")
            except ProcessLookupError:
                pass
    runs = sorted(r[0] for r in rows if "skydiscover.cli" in r[4] and "--config" in r[4])
    paused &= set(runs)
    if not runs:
        log("no runs left; exiting"); break
    active = [p for p in runs if p not in paused]
    # resume first (oldest paused first), then pause surplus
    while len(active) < MAX_ACTIVE and paused:
        p = sorted(paused)[0]; paused.discard(p); active.append(p)
        os.kill(p, signal.SIGCONT); log(f"resumed run {p}")
    surplus = len(active) - MAX_ACTIVE
    for p in reversed(active):
        if surplus <= 0:
            break
        cmds = {r[0]: r[4] for r in rows}
        busy = any("resource_tracker" not in cmds.get(d, "") for d in descendants(rows, p))
        if not busy:  # no evaluation child (waiting on the LLM): safe to pause
            os.kill(p, signal.SIGSTOP); paused.add(p); surplus -= 1; log(f"paused run {p}")
    time.sleep(20)
