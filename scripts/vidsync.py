# clock sync between the phone and the two cameras. coarse from the files'
# own recording timestamps (alex's iphone clock ran ~41s fast that day, fitted
# out), fine from correlating the two soundtracks. cached in
# output/report/camera_sync.json - trials 3+4 pinned after frame-checking.
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
    # lag such that series2_time = series1_time + lag. give centre+span to
    # search only near an expected value (uses pearson r so overlap length
    # doesn't win over actual agreement)
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
    # (kind, when, duration_s) from the mp4/mov metadata. iphones store the
    # local start time, androids stamp mvhd in UTC when recording stops
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
    # phyphox states its start time in the csv header
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


def _flux(video):
    # onset strength of the soundtrack at 100 Hz, cached next to the video
    cache = os.path.splitext(video)[0] + ".volare_flux.npz"
    if os.path.exists(cache):
        return np.load(cache)["f"]
    import subprocess
    import tempfile
    import imageio_ffmpeg
    from scipy.io import wavfile
    from scipy.signal import stft
    wav = os.path.join(tempfile.gettempdir(), os.path.basename(video) + ".wav")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i",
                    video, "-vn", "-ac", "1", "-ar", "8000", "-f", "wav", wav],
                   check=True)
    fs, x = wavfile.read(wav)
    os.remove(wav)
    f, t, S = stft(x.astype(np.float64), fs=8000, nperseg=512,
                   noverlap=512 - 80, padded=False)
    m = np.abs(S[(f >= 400) & (f <= 3000)])
    fl = np.log1p(np.maximum(np.diff(m, axis=1), 0).sum(axis=0))
    fl = fl - pd.Series(fl).rolling(300, center=True, min_periods=1).mean().to_numpy()
    np.savez_compressed(cache, f=fl)
    return fl


def audio_offset(va, vr, centre, span=16.0):
    # each audio chunk votes for its best lag - a real lock is a tight cluster.
    # returns (offset, cluster votes, runner-up votes) or None
    try:
        fa, fr = _flux(va), _flux(vr)
    except Exception:
        return None
    rate = 100.0
    la, lr = len(fa) / rate, len(fr) / rate
    votes = []
    for chunk, hop in ((10.0, 5.0), (20.0, 10.0)):
        lo = max(0.0, -centre - span)
        hi = min(la, lr - centre + span) - chunk
        for c0 in np.arange(lo, hi, hop):
            a = fa[int(c0 * rate):int((c0 + chunk) * rate)]
            if a.std() < 1e-9:
                continue
            best = (-2.0, None)
            for lag in np.arange(centre - span, centre + span, 1.0 / rate):
                i0 = int(round((c0 + lag) * rate))
                b = fr[max(0, i0):max(0, i0 + len(a))]
                if len(b) < len(a):
                    continue
                aa = a - a.mean()
                bb = b - b.mean()
                d = np.sqrt((aa * aa).sum() * (bb * bb).sum())
                if d > 0:
                    r = float((aa * bb).sum() / d)
                    if r > best[0]:
                        best = (r, float(lag))
            if best[1] is not None and best[0] > 0.10:
                votes.append(best[1])
    if not votes:
        return None
    lags = np.array(votes)
    top = (0, None)
    for x in lags:
        k = int(np.sum(np.abs(lags - x) <= 0.75))
        if k > top[0]:
            top = (k, x)
    cl = lags[np.abs(lags - top[1]) <= 0.75]
    rest = lags[np.abs(lags - top[1]) > 0.75]
    runner = 0
    for x in rest:
        runner = max(runner, int(np.sum(np.abs(rest - x) <= 0.75)))
    return float(cl.mean()), int(len(cl)), runner


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

    # ryan's stamps are UTC - find the timezone that fits all four trials
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

    # alex's clock skew is constant, so fit it across the trials - the angle
    # envelopes give a rough offset first, audio then pins what it can
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

    audio, locked = {}, {}
    for n in (1, 2, 3, 4):
        audio[n] = audio_offset(vids[("alex", n)], vids[("ryan", n)], env[n])
        if audio[n] and audio[n][1] >= 8 and audio[n][1] >= 1.7 * audio[n][2]:
            locked[n] = audio[n][0]
    pool = locked or env
    skew = float(np.median([d0[n] - pool[n] for n in pool]))

    table = {"timezone_hours": tz / 3600.0, "alex_clock_fast_s": round(skew, 1),
             "trials": {}}
    for n in (1, 2, 3, 4):
        pred = d0[n] - skew
        if n in locked:
            off, src = locked[n], "audio"
        else:
            off, src = pred, "stamps"
            hint = (f" (best cluster {audio[n][0]:+.1f}, {audio[n][1]} v "
                    f"{audio[n][2]} votes)") if audio[n] else ""
            say(f"trial {n}: audio inconclusive{hint}; falling back to the "
                f"stamps - worth checking against the footage")
        ps, run = phones[n]
        lag_r = (ryan_start[n] - ps).total_seconds()
        lag_a = lag_r + off
        say(f"trial {n}: ryan = alex {off:+.1f}s [{src}] (stamps {pred:+.1f}, "
            f"envelope {env[n]:+.1f})   phone lag: alex {lag_a:+.1f}s ryan {lag_r:+.1f}s")
        table["trials"][str(n)] = {
            "alex": os.path.relpath(vids[("alex", n)], ROOT).replace("\\", "/"),
            "ryan": os.path.relpath(vids[("ryan", n)], ROOT).replace("\\", "/"),
            "off_alex_to_ryan": round(off, 2),
            "lag_alex": round(lag_a, 2),
            "lag_ryan": round(lag_r, 2),
            "source": src,
        }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    say(f"cached: {CACHE}")
    return table


def trial_sync(n):
    table = resolve(quiet=True)
    if not table or str(n) not in table["trials"]:
        return None
    row = dict(table["trials"][str(n)])
    row["alex"] = os.path.join(ROOT, row["alex"])
    row["ryan"] = os.path.join(ROOT, row["ryan"])
    return row


def video_lag(video):
    # phone_time = video_time + lag, or None for unknown footage
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
