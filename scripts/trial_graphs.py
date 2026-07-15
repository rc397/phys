# per-trial report figure: angle on top, acceleration underneath, one clock.
#   python trial_graphs.py
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import accel
import vidsync

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ACCEL = {1: "1st Trial.csv", 2: "2nd Trial.csv", 3: "3rd Trial.csv", 4: "4th Trial.csv"}


def angle_csv(trial):
    # prefer Alex's camera (its elevation estimate is the trusted one)
    for pat in (f"output/angles/*trial {trial}_angle_alex.csv", f"output/angles/*trial {trial}_angle_ryan.csv"):
        hits = glob.glob(os.path.join(ROOT, pat))
        if hits:
            return hits[0]
    return None


def ride_window(t, y, thr=5.0, gap=15.0, minlen=30.0):
    # longest sustained stretch of real acceleration = the loaded run
    on = np.asarray(y) > thr
    segs, i, n = [], 0, len(t)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            if not segs or t[i] - segs[-1][1] > gap:
                segs.append([t[i], t[j]])
            else:
                segs[-1][1] = t[j]
            i = j + 1
        else:
            i += 1
    segs = [s for s in segs if s[1] - s[0] > minlen]
    return max(segs, key=lambda s: s[1] - s[0]) if segs else None


for trial in (1, 2, 3, 4):
    alex_hits = glob.glob(os.path.join(ROOT, f"output/angles/*trial {trial}_angle_alex.csv"))
    ryan_hits = glob.glob(os.path.join(ROOT, f"output/angles/*trial {trial}_angle_ryan.csv"))
    acsv = alex_hits[0] if alex_hits else (ryan_hits[0] if ryan_hits else None)
    other_csv = ryan_hits[0] if (alex_hits and ryan_hits) else None
    pcsv = os.path.join(ROOT, "data", ACCEL[trial])
    if not acsv or not os.path.exists(pcsv):
        print(f"trial {trial}: missing data, skipped")
        continue
    v = pd.read_csv(acsv)
    a = accel.load(pcsv)
    ta = pd.to_numeric(a[accel.find_time(a)], errors="coerce").to_numpy()
    at = pd.to_numeric(a[a.columns[4]], errors="coerce").to_numpy()
    at_s = accel.smooth(at, 2 / 151)                 # zero-phase, ~1.5 s
    # average over the ~5-6 s wave for the comparison line
    at_w = pd.Series(at_s).rolling(801, center=True, min_periods=100).mean().to_numpy()

    # put the phone on the video clock (same resolved sync as everywhere)
    cam = "alex" if acsv.endswith("_alex.csv") else "ryan"
    sync = vidsync.trial_sync(trial)
    if sync:
        lag = sync[f"lag_{cam}"]
    else:
        th_phone = np.degrees(np.arctan(np.nan_to_num(at_s) / G))
        lag = vidsync.xcorr_lag(v["time"].to_numpy(),
                                np.nan_to_num(v["theta_ema"].to_numpy()), ta, th_phone)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   constrained_layout=True)
    tp = ta - lag                                    # phone time on the video clock
    keep = (tp >= 0) & (tp <= v["time"].max())

    # zero the clock at ride start - the phone recorded for ages in the queue
    ride = ride_window(tp, np.nan_to_num(at_s))
    t0, t_end = ride if ride else (0.0, v["time"].max())
    vt = v["time"].to_numpy() - t0                    # video time, ride-relative
    pt = tp - t0                                      # phone time, ride-relative
    xlo, xhi = -40.0, (t_end - t0) + 40.0

    # the cameras bracket the phone: alex reads low, ryan high (viewpoints)
    ax1.plot(vt, v["theta_ema"], color=accel.C_EMA, lw=2, label=f"video {cam} (reads low)")
    if other_csv:
        ov = pd.read_csv(other_csv)
        off = sync["off_alex_to_ryan"] if sync else 0.0
        ot = ov["time"].to_numpy() - off - t0                # ryan on the alex clock
        ax1.plot(ot, ov["theta_ema"], color="#2e8b57", lw=1.8,
                 label="video ryan (reads high)")
    th_p = np.degrees(np.arctan(at_s / G))
    th_w = np.degrees(np.arctan(at_w / G))
    ax1.plot(pt[keep], th_w[keep], color="#123f8f", lw=2.6,
             label="phone (trusted absolute)")
    ax1.plot(pt[keep], th_p[keep], color=accel.C_FIT, lw=0.9, alpha=0.4,
             label="phone (with the wave)")
    # shade where the ride spins without the phone rider aboard
    ph_on_grid = np.interp(vt, pt[keep], th_p[keep], left=0, right=0)
    solo = (v["theta_ema"].to_numpy() > 15) & (ph_on_grid < 5)
    if solo.any():
        start = None
        first = True
        for i, s in enumerate(solo):
            if s and start is None:
                start = vt[i]
            if (not s or i == len(solo) - 1) and start is not None:
                ax1.axvspan(start, vt[i], color="#bbbbbb", alpha=0.18,
                            label="ride spinning, phone not on it" if first else None)
                first = False
                start = None
    ax1.set_ylabel("fly-out angle  (deg)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    accel.style_axis(ax1)

    ax2.plot(pt[keep], at[keep], color=accel.C_RAW, lw=0.6, alpha=0.5, label="raw")
    ax2.plot(pt[keep], at_s[keep], color=accel.C_FIT, lw=1.2, alpha=0.7,
             label="smoothed (the wave)")
    ax2.plot(pt[keep], at_w[keep], color="#123f8f", lw=2.2, label="wave-averaged")
    ax2.set_ylabel("acceleration aT  (m/s$^2$)")
    ax2.set_xlabel("time from ride start (s)")
    ax2.set_xlim(xlo, xhi)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    accel.style_axis(ax2)

    fig.suptitle(f"Trial {trial}: fly-out angle (video) and rider acceleration (phone)",
                 fontsize=13, fontweight="bold")
    out = os.path.join(ROOT, "output", "visuals", f"trial {trial}_theta_accel.png")
    accel.save(fig, out)
    plt.close(fig)
    print(f"trial {trial}: {out}   (phone shifted {lag:+.1f}s onto the video clock)")
