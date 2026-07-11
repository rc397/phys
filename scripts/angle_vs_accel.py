# The correlation plot: video tan(theta) against the phone's centripetal
# acceleration a. For a conical pendulum tan(theta) = a/g, and since a = v^2/r
# this is the angle-vs-velocity-squared test with the radius factored out -
# no ride geometry needed, everything on the plot is measured. Time pairing
# comes from the synced per-trial sheets. Rows where the phone reads flat
# (ride idle / rider not aboard) are dropped: there the video only sees junk.
#   python scripts/angle_vs_accel.py
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import accel

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MARKS = {"1": "o", "2": "s", "3": "^", "4": "D"}

fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
pts = {"alex": [[], []], "ryan": [[], []]}
for n in ("1", "2", "3", "4"):
    d = pd.read_csv(os.path.join(ROOT, "output", "report", f"trial {n}_synced_data.csv"))
    a = d["phone_aT_smooth_ms2"].to_numpy()
    # ramp-up and plateau only: during the spin-down the chairs (and the
    # detector) lag the falling acceleration, so those points don't test the
    # steady-state law. cut at the last moment a is still near its plateau.
    hold = np.where(a >= 0.85 * np.nanmax(a))[0][-1]
    ok0 = (np.arange(len(a)) <= hold) & (a > 1.5)
    for cam, col in (("alex", "alex_video_angle_deg"), ("ryan", "ryan_video_angle_deg")):
        th = d[col].to_numpy()
        ok = ok0 & np.isfinite(th) & (th > 2)
        x, y = a[ok], np.tan(np.radians(th[ok]))
        pts[cam][0].append(x)
        pts[cam][1].append(y)
        color = accel.C_EMA if cam == "alex" else "#2e8b57"
        ax.plot(x[::4], y[::4], MARKS[n], color=color, ms=3, alpha=0.25)

xs = np.linspace(0, 17, 50)
ax.plot(xs, xs / G, color="#123f8f", lw=2.4, label="conical pendulum:  tan θ = a / g")
for cam, color in (("alex", accel.C_EMA), ("ryan", "#2e8b57")):
    x = np.concatenate(pts[cam][0])
    y = np.concatenate(pts[cam][1])
    slope = float((x * y).sum() / (x * x).sum())
    ax.plot(xs, xs * slope, color=color, lw=1.6, ls="--",
            label=f"video {cam} fit:  slope {slope:.4f}  (1/g = {1 / G:.4f})")
    ax.plot([], [], "o", color=color, ms=5, alpha=0.6, label=f"video {cam}, all trials")

ax.set_xlabel("phone centripetal acceleration  a = v$^2$/r  (m/s$^2$)")
ax.set_ylabel("tan θ   (video fly-out angle)")
ax.set_xlim(0, 17)
ax.set_ylim(0, 2.4)
ax.legend(fontsize=9, loc="upper left")
accel.style_axis(ax)
ax.set_title("Fly-out angle vs acceleration (∝ v²): the conical-pendulum test",
             fontsize=12, fontweight="bold")
out = os.path.join(ROOT, "output", "report", "tan_theta_vs_accel.png")
fig.savefig(out, dpi=150)
print(out)
