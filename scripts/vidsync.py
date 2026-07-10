# Clock alignment between the three instruments. Correlating the signals
# blind turned out to be treacherous: the ride is periodic, so an
# unconstrained cross-correlation happily locks a rotation (or a whole spin
# cycle) off, and the two viewpoints see the chair bunching at different
# rotation phases, which biases even the "right" peak by several seconds.
# So the anchors are the recording clocks instead:
#   phone - each phyphox CSV states "Recording started at:" in its header
#   ryan  - his phone stamps the mp4 when the recording STOPS (mvhd, UTC);
#           start = stop - duration. Trial 1 agrees with the phone to 0.1 s.
#   alex  - his iPhone writes the recording start into the MOV as
#           com.apple.quicktime.creationdate (local time), but that clock ran
#           ~45 s fast on the day. The offset is the same for all four files,
#           so it is fitted from the footage and the stamps together.
# Correlation is only used inside a small window of what the stamps predict:
# the capped angle curves (ramps and stops count, plateau ripple doesn't) fit
# alex's clock offset and may refine a trial by a few seconds, never more.
# The resolved table is cached in output/report/camera_sync.json so figures
# can be rebuilt without the footage.
import datetime
import glob
import json
import os
import re
import struct

import numpy as np
import pandas as pd

import accel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "output", "report", "camera_sync.json")
PHONE = {1: "1st Trial.csv", 2: "2nd Trial.csv", 3: "3rd Trial.csv", 4: "4th Trial.csv"}


def xcorr_lag(t1, s1, t2, s2, grid=0.5, centre=None, span=None):
    # lag such that series2_time = series1_time + lag. Constrained mode scores
    # each candidate by the pearson r of the overlap, so a lag with more
    # overlap can't beat a lag with better agreement (matters for signals that
    # are mostly plateau); it also keeps a periodic signal from locking a
    # cycle off. Unconstrained mode is the plain full correlation.
    g1 = np.arange(t1.min(), t1.max(), grid)
    g2 = np.arange(t2.min(), t2.max(), grid)
    x1 = np.interp(g1, t1, s1)
    x2 = np.interp(g2, t2, s2)
    if centre is None:
        z1 = (x1 - x1.mean()) / (x1.std() + 1e-9)
        z2 = (x2 - x2.mean()) / (x2.std() + 1e-9)
        c = np.correlate(z2, z1, "full")
        return float((np.argmax(c) - (len(x1) - 1)) * grid + g2[0] - g1[0])
    best = (-2.0, centre)
    for lag in np.arange(centre - span, centre + span + grid / 2, grid):
        lo = max(g1[0] + lag, g2[0])
        hi = min(g1[-1] + lag, g2[-1])
        if hi - lo < 40:
            continue
        g = np.arange(lo, hi, grid)
        a = np.interp(g - lag, g1, x1)
        b = np.interp(g, g2, x2)
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        if d > 0 and (a * b).sum() / d > best[0]:
            best = (float((a * b).sum() / d), float(lag))
    return best[1]


def _atoms(f, start, end):
    pos = start
    while pos < end - 8:
        f.seek(pos)
        head = f.read(8)
        if len(head) < 8:
            return
        size, typ = struct.unpack(">I4s", head)
        header = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header = 16
        if size == 0:
            size = end - pos
        yield pos, size, header, typ
        pos += size


def recording_span(path):
    # what the container remembers about the recording, as
    # (kind, when, duration_s): kind "start" means `when` is the local wall
    # clock at record-start (iPhone creation date); "stop_utc" means `when`
    # is UTC at record-stop (androids stamp mvhd when finalising the file)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        moov = next(((p, s) for p, s, h, t in _atoms(f, 0, size) if t == b"moov"), None)
        if moov is None:
            return None
        f.seek(moov[0])
        blob = f.read(min(moov[1], 16 << 20))
    i = blob.find(b"mvhd")
    if i < 0:
        return None
    if blob[i + 4] == 1:
        ctime, _, scale, dur = struct.unpack(">QQIQ", blob[i + 8:i + 36])
    else:
        ctime, _, scale, dur = struct.unpack(">IIII", blob[i + 8:i + 24])
    dur_s = dur / scale
    m = re.search(rb"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?[+-]\d{4}", blob)
    if m:
        start = datetime.datetime.strptime(m.group(1).decode(), "%Y-%m-%dT%H:%M:%S")
        return "start", start, dur_s
    utc = datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=ctime)
    return "stop_utc", utc, dur_s


def phone_recording(n):
    # (wall-clock start, loaded-run span in CSV seconds) for a trial's log
    path = os.path.join(ROOT, "data", PHONE[n])
    start = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for _ in range(12):
            m = re.search(r"Recording started at:\s*([\d-]+ [\d:]+(?:\.\d+)?)", f.readline())
            if m:
                s = m.group(1)
                fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S"
                start = datetime.datetime.strptime(s, fmt)
                break
    a = accel.load(path)
    t = pd.to_numeric(a[accel.find_time(a)], errors="coerce").to_numpy()
    at = pd.to_numeric(a[a.columns[4]], errors="coerce").to_numpy()
    on = accel.ema(np.nan_to_num(at), 2 / 301) > 2.5
    best, i = None, 0
    while i < len(t):
        if on[i]:
            j = i
            while j + 1 < len(t) and on[j + 1]:
                j += 1
            if best is None or t[j] - t[i] > best[1] - best[0]:
                best = (t[i], t[j])
            i = j + 1
        else:
            i += 1
    return start, best


def _video(cam, n):
    pat = {"alex": ("*lex*", "MOV"), "ryan": ("*yan*", "mp4")}[cam]
    hits = glob.glob(os.path.join(ROOT, "Videos", pat[0], f"*trial {n}.{pat[1]}"))
    return hits[0] if hits else None


def _theta_curve(cam, n):
    hits = glob.glob(os.path.join(ROOT, "output", "angles", f"*trial {n}_angle_{cam}.csv"))
    if not hits:
        return None
    v = pd.read_csv(hits[0])
    t = v["time"].to_numpy()
    th = np.nan_to_num(pd.to_numeric(v["theta_ema"], errors="coerce").to_numpy())
    return t, th


def resolve(force=False, quiet=False):
    # work the whole sync table out once and cache it
    if not force and os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)

    say = (lambda *_: None) if quiet else print
    vids = {(c, n): _video(c, n) for c in ("alex", "ryan") for n in (1, 2, 3, 4)}
    if any(v is None for v in vids.values()):
        say("vidsync: footage missing and no cached table; cannot resolve")
        return None
    spans = {k: recording_span(v) for k, v in vids.items()}
    phones = {n: phone_recording(n) for n in (1, 2, 3, 4)}

    # ryan's stamps are UTC; find the one half-hour timezone that puts every
    # trial's phone run inside his recording window
    tz = None
    for cand in range(-12 * 3600, 14 * 3600 + 1, 1800):
        ok = True
        for n in (1, 2, 3, 4):
            kind, stop, dur = spans[("ryan", n)]
            if kind != "stop_utc":
                ok = False
                break
            start = stop + datetime.timedelta(seconds=cand - dur)
            ps, run = phones[n]
            lo = (ps - start).total_seconds() + run[0]
            hi = (ps - start).total_seconds() + run[1]
            if not (-15 <= lo and hi <= dur + 15):
                ok = False
                break
        if ok:
            tz = cand
            break
    if tz is None:
        say("vidsync: ryan's recording stamps do not bracket the phone runs")
        return None

    ryan_start = {}
    for n in (1, 2, 3, 4):
        _, stop, dur = spans[("ryan", n)]
        ryan_start[n] = stop + datetime.timedelta(seconds=tz - dur)

    # alex's stamps are recording starts on a clock that runs a constant bit
    # fast; fit that constant from the capped angle envelopes (ramps and
    # stops count, plateau ripple doesn't), then refine each trial's offset
    # near the stamp prediction
    d0, env = {}, {}
    for n in (1, 2, 3, 4):
        kind, start, _ = spans[("alex", n)]
        if kind != "start":
            say(f"vidsync: no creation date in {vids[('alex', n)]}")
            return None
        d0[n] = (start - ryan_start[n]).total_seconds()
        ta, tha = _theta_curve("alex", n)
        tr, thr = _theta_curve("ryan", n)
        env[n] = xcorr_lag(ta, np.minimum(tha, 22), tr, np.minimum(thr, 22),
                           centre=d0[n] - 45, span=60)
    skew = float(np.median([d0[n] - env[n] for n in (1, 2, 3, 4)]))

    table = {"timezone_hours": tz / 3600.0, "alex_clock_fast_s": round(skew, 1),
             "trials": {}}
    for n in (1, 2, 3, 4):
        # the envelope may fine-tune a trial by a few seconds; if it strays
        # further it lost its grip (junky readings), so the stamps stand
        pred = d0[n] - skew
        off = env[n] if abs(env[n] - pred) <= 5 else pred
        ps, run = phones[n]
        lag_r = (ryan_start[n] - ps).total_seconds()
        lag_a = lag_r + off
        say(f"trial {n}: ryan = alex {off:+.1f}s (stamps {pred:+.1f}, "
            f"envelope {env[n]:+.1f})   phone lag: alex {lag_a:+.1f}s ryan {lag_r:+.1f}s")
        table["trials"][str(n)] = {
            "alex": os.path.relpath(vids[("alex", n)], ROOT).replace("\\", "/"),
            "ryan": os.path.relpath(vids[("ryan", n)], ROOT).replace("\\", "/"),
            "off_alex_to_ryan": round(off, 2),
            "lag_alex": round(lag_a, 2),
            "lag_ryan": round(lag_r, 2),
        }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    say(f"cached: {CACHE}")
    return table


def trial_sync(n):
    # the resolved sync row for a trial, with absolute video paths; None if
    # the table can't be built (no cache and no footage)
    table = resolve(quiet=True)
    if not table or str(n) not in table["trials"]:
        return None
    row = dict(table["trials"][str(n)])
    row["alex"] = os.path.join(ROOT, row["alex"])
    row["ryan"] = os.path.join(ROOT, row["ryan"])
    return row


def video_lag(video):
    # phone_time = video_time + lag for one of the trial videos; None for
    # footage outside the sync table
    m = re.search(r"trial (\d)", os.path.basename(video).lower())
    if not m:
        return None
    row = trial_sync(int(m.group(1)))
    if row is None:
        return None
    name = os.path.basename(video)
    for cam in ("alex", "ryan"):
        if name == os.path.basename(row[cam]):
            return row[f"lag_{cam}"]
    return None


if __name__ == "__main__":
    import sys
    resolve(force="--force" in sys.argv)
