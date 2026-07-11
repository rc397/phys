# Report figures: one per trial, angle over time (video) above acceleration over
# time (phone), on a shared clock. The phone trace is shifted onto the video
# clock using the recording stamps resolved in vidsync, same as everywhere.
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


for trial in (1, 2, 3, 4):
    acsv = angle_csv(trial)
    pcsv = os.path.join(ROOT, "data", ACCEL[trial])
    if not acsv or not os.path.exists(pcsv):
        print(f"trial {trial}: missing data, skipped")
        continue
    v = pd.read_csv(acsv)
    a = accel.load(pcsv)
    ta = pd.to_numeric(a[accel.find_time(a)], errors="coerce").to_numpy()
    at = pd.to_numeric(a[a.columns[4]], errors="coerce").to_numpy()
    at_s = accel.ema(at, 2 / 151)                    # ~1.5 s smoothing at 100 Hz
    # the rider feels the rotor-tilt wave (a stable ~5-6 s oscillation); average
    # over it for the line the video should be compared against
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

    # top: both angle estimates together - if the physics holds they line up
    ax1.plot(v["time"], v["theta"], ".", color=accel.C_RAW, ms=3.5, label="video, per window")
    ax1.plot(v["time"], v["theta_ema"], color=accel.C_EMA, lw=2, label="video, smoothed")
    th_p = np.degrees(np.arctan(at_s / G))
    th_w = np.degrees(np.arctan(at_w / G))
    ax1.plot(tp[keep], th_p[keep], color=accel.C_FIT, lw=1.0, alpha=0.55,
             label="phone  arctan(aT/g)")
    ax1.plot(tp[keep], th_w[keep], color="#123f8f", lw=2.2,
             label="phone, wave-averaged")
    # the ride also does empty warm-up / re-spins; the phone is in the queue
    # then, so the camera sees an angle while the phone reads flat. shade those
    # so the split doesn't read as a measurement error
    ph_on_grid = np.interp(v["time"], tp[keep], th_p[keep], left=0, right=0)
    solo = (v["theta_ema"].to_numpy() > 15) & (ph_on_grid < 5)
    if solo.any():
        tarr = v["time"].to_numpy()
        start = None
        first = True
        for i, s in enumerate(solo):
            if s and start is None:
                start = tarr[i]
            if (not s or i == len(solo) - 1) and start is not None:
                ax1.axvspan(start, tarr[i], color="#bbbbbb", alpha=0.18,
                            label="ride spinning, phone not on it" if first else None)
                first = False
                start = None
    ax1.set_ylabel("fly-out angle  (deg)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    accel.style_axis(ax1)

    ax2.plot(tp[keep], at[keep], color=accel.C_RAW, lw=0.6, alpha=0.5, label="raw")
    ax2.plot(tp[keep], at_s[keep], color=accel.C_FIT, lw=1.2, alpha=0.7,
             label="smoothed (the wave)")
    ax2.plot(tp[keep], at_w[keep], color="#123f8f", lw=2.2, label="wave-averaged")
    ax2.set_ylabel("acceleration aT  (m/s$^2$)")
    ax2.set_xlabel("time (s)")
    ax2.set_xlim(0, v["time"].max())
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    accel.style_axis(ax2)

    fig.suptitle(f"Trial {trial}: fly-out angle (video) and rider acceleration (phone)",
                 fontsize=13, fontweight="bold")
    out = os.path.join(ROOT, "output", "report", f"trial {trial}_theta_accel.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"trial {trial}: {out}   (phone shifted {lag:+.1f}s onto the video clock)")
