# Report figures: one per trial, angle over time (video) above acceleration over
# time (phone), on a shared clock. The phone trace is shifted onto the video clock
# by cross-correlating the two angle profiles, same as volare_angle does.
#   python trial_graphs.py
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import accel

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
ACCEL = {1: "1st Trial.csv", 2: "2nd Trial.csv", 3: "3rd Trial.csv", 4: "4th Trial.csv"}


def angle_csv(trial):
    # prefer Alex's camera (its elevation estimate is the trusted one)
    for pat in (f"output/*trial {trial}_angle_alex.csv", f"output/*trial {trial}_angle_ryan.csv"):
        hits = glob.glob(os.path.join(HERE, pat))
        if hits:
            return hits[0]
    return None


for trial in (1, 2, 3, 4):
    acsv = angle_csv(trial)
    pcsv = os.path.join(HERE, "Accelerometer data", ACCEL[trial])
    if not acsv or not os.path.exists(pcsv):
        print(f"trial {trial}: missing data, skipped")
        continue
    v = pd.read_csv(acsv)
    a = accel.load(pcsv)
    ta = pd.to_numeric(a[accel.find_time(a)], errors="coerce").to_numpy()
    at = pd.to_numeric(a[a.columns[4]], errors="coerce").to_numpy()
    at_s = accel.ema(at, 2 / 151)                    # ~1.5 s smoothing at 100 Hz

    # put the phone on the video clock via the two angle profiles
    grid = 0.5
    gv = np.arange(0, v["time"].max() + grid, grid)
    sv = np.interp(gv, v["time"], np.nan_to_num(v["theta_ema"]), left=0, right=0)
    ga = np.arange(ta.min(), ta.max(), grid)
    sa = np.interp(ga, ta, np.degrees(np.arctan(np.nan_to_num(at_s) / G)))
    corr = np.correlate((sa - sa.mean()) / (sa.std() + 1e-9),
                        (sv - sv.mean()) / (sv.std() + 1e-9), mode="full")
    lag = (np.argmax(corr) - (len(sv) - 1)) * grid + ga[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   constrained_layout=True)
    tp = ta - lag                                    # phone time on the video clock
    keep = (tp >= 0) & (tp <= v["time"].max())

    # top: both angle estimates together - if the physics holds they line up
    ax1.plot(v["time"], v["theta"], ".", color=accel.C_RAW, ms=3.5, label="video, per window")
    ax1.plot(v["time"], v["theta_ema"], color=accel.C_EMA, lw=2, label="video, smoothed")
    th_p = np.degrees(np.arctan(at_s / G))
    ax1.plot(tp[keep], th_p[keep], color=accel.C_FIT, lw=1.6,
             label="phone  arctan(aT/g)")
    ax1.set_ylabel("fly-out angle  (deg)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    accel.style_axis(ax1)

    ax2.plot(tp[keep], at[keep], color=accel.C_RAW, lw=0.6, alpha=0.5, label="raw")
    ax2.plot(tp[keep], at_s[keep], color=accel.C_FIT, lw=2, label="smoothed")
    ax2.set_ylabel("acceleration aT  (m/s$^2$)")
    ax2.set_xlabel("time (s)")
    ax2.set_xlim(0, v["time"].max())
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    accel.style_axis(ax2)

    fig.suptitle(f"Trial {trial}: fly-out angle (video) and rider acceleration (phone)",
                 fontsize=13, fontweight="bold")
    out = os.path.join(HERE, "output", f"trial {trial}_theta_accel.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"trial {trial}: {out}   (phone shifted {lag:+.1f}s onto the video clock)")
