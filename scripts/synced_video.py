# Builds a synced side-by-side video per trial: Alex's camera and Ryan's camera
# playing together on one clock, with the rider's accelerometer trace scrolling
# underneath and a cursor marking "now". The clocks are aligned the same way as
# the analysis: cross-correlating each video's angle curve against the phone's
# arctan(aT/g), so all three views show the same physical moment.
#   python scripts/synced_video.py            all four trials
#   python scripts/synced_video.py 2          just trial 2
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accel
from vidsync import video_offset

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ACCEL = {1: "1st Trial.csv", 2: "2nd Trial.csv", 3: "3rd Trial.csv", 4: "4th Trial.csv"}
OUT_FPS = 15
PANE_W, PANE_H = 640, 360
STRIP_H = 230
HEAD_H = 34


def find_video(trial, cam):
    folder = "Alex's persepctive" if cam == "alex" else "Physics video, Ryan Perspective"
    ext = "MOV" if cam == "alex" else "mp4"
    hits = glob.glob(os.path.join(ROOT, "Videos", folder, f"*trial {trial}.{ext}"))
    return hits[0] if hits else None


def lag_for(trial, cam):
    # phone time = video time + lag, from the same correlation the analysis uses
    csvs = glob.glob(os.path.join(ROOT, "output", "angles", f"*trial {trial}_angle_{cam}.csv"))
    if not csvs:
        return None
    v = pd.read_csv(csvs[0])
    a = accel.load(os.path.join(ROOT, "data", ACCEL[trial]))
    ta = pd.to_numeric(a[accel.find_time(a)], errors="coerce").to_numpy()
    at = pd.to_numeric(a[a.columns[4]], errors="coerce").to_numpy()
    th_a = np.degrees(np.arctan(accel.ema(at, 2 / 301) / G))
    grid = 0.5
    gv = np.arange(0, v["time"].max() + grid, grid)
    sv = np.interp(gv, v["time"], np.nan_to_num(v["theta_ema"]), left=0, right=0)
    ga = np.arange(ta.min(), ta.max(), grid)
    sa = np.interp(ga, ta, np.nan_to_num(th_a))
    sv_n = (sv - sv.mean()) / (sv.std() + 1e-9)
    sa_n = (sa - sa.mean()) / (sa.std() + 1e-9)
    corr = np.correlate(sa_n, sv_n, mode="full")
    lag = (np.argmax(corr) - (len(sv_n) - 1)) * grid + ga[0]
    theta = (v["time"].to_numpy(), np.nan_to_num(v["theta_ema"].to_numpy()))
    return lag, theta


class Player:
    # sequential reader that hands out the frame nearest a requested time
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.idx = -1
        self.frame = None

    def at(self, t):
        want = int(round(t * self.fps))
        while self.idx < want:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx += 1
            self.frame = f
        return self.frame

    def close(self):
        self.cap.release()


def accel_strip(ta, at_s, at_w, t0, t1, width):
    # pre-rendered acceleration trace; the per-frame cursor is drawn on a copy
    img = np.full((STRIP_H, width, 3), 250, np.uint8)
    x0, x1, y0, y1 = 46, width - 12, STRIP_H - 26, 10
    top = np.nanmax(at_s) * 1.08

    def X(t):
        return int(x0 + (t - t0) / (t1 - t0) * (x1 - x0))

    def Y(v):
        return int(y0 - v / top * (y0 - y1))

    for v in range(0, int(top) + 1, 5):
        cv2.line(img, (x0, Y(v)), (x1, Y(v)), (235, 235, 235), 1)
        cv2.putText(img, str(v), (8, Y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1, cv2.LINE_AA)
    sel = (ta >= t0) & (ta <= t1)
    for series, col, th in ((at_s, (229, 145, 74), 1), (at_w, (143, 63, 18), 2)):
        pts = np.column_stack([[X(t) for t in ta[sel][::20]],
                               [Y(v) for v in series[sel][::20]]]).astype(np.int32)
        cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
    cv2.putText(img, "phone aT (m/s^2): light = with the wave, dark = wave-averaged",
                (x0, STRIP_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1,
                cv2.LINE_AA)
    return img, X, Y


def build(trial):
    va = find_video(trial, "alex")
    vr = find_video(trial, "ryan")
    la = lag_for(trial, "alex")
    lr = lag_for(trial, "ryan")
    if not (va and vr and la and lr):
        print(f"trial {trial}: missing pieces, skipped")
        return
    lag_a, th_a = la
    _, th_r = lr
    # sync the cameras to each other on the ride's motion, then anchor the pair
    # to the phone through Alex's verified lag
    off = video_offset(va, vr)
    lag_r = lag_a - off

    a = accel.load(os.path.join(ROOT, "data", ACCEL[trial]))
    ta = pd.to_numeric(a[accel.find_time(a)], errors="coerce").to_numpy()
    at = pd.to_numeric(a[a.columns[4]], errors="coerce").to_numpy()
    at_s = accel.ema(at, 2 / 151)
    at_w = pd.Series(at_s).rolling(801, center=True, min_periods=100).mean().to_numpy()

    pa, pr = Player(va), Player(vr)
    dur_a = pa.cap.get(cv2.CAP_PROP_FRAME_COUNT) / pa.fps
    dur_r = pr.cap.get(cv2.CAP_PROP_FRAME_COUNT) / pr.fps
    # master clock is the phone's; play only where all three overlap
    t0 = max(lag_a, lag_r, ta.min())
    t1 = min(lag_a + dur_a, lag_r + dur_r, ta.max())
    if t1 - t0 < 20:
        print(f"trial {trial}: overlap too short, skipped")
        return

    W = 2 * PANE_W
    H = HEAD_H + PANE_H + STRIP_H
    base_strip, X, Y = accel_strip(ta, at_s, at_w, t0, t1, W)
    out = os.path.join(ROOT, "output", "report", f"trial {trial}_synced.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (W, H))

    for T in np.arange(t0, t1, 1.0 / OUT_FPS):
        canvas = np.full((H, W, 3), 16, np.uint8)
        for px, (pl, lag, th, name) in enumerate(
                ((pa, lag_a, th_a, "alex"), (pr, lag_r, th_r, "ryan"))):
            f = pl.at(T - lag)
            if f is not None:
                canvas[HEAD_H:HEAD_H + PANE_H, px * PANE_W:(px + 1) * PANE_W] = \
                    cv2.resize(f, (PANE_W, PANE_H))
            ang = float(np.interp(T - lag, th[0], th[1]))
            cv2.putText(canvas, f"{name}   video theta {ang:4.1f} deg",
                        (px * PANE_W + 10, HEAD_H + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
        at_now = float(np.interp(T, ta, np.nan_to_num(at_s)))
        cv2.putText(canvas,
                    f"trial {trial}   t = {T - t0:6.1f} s   phone aT {at_now:4.1f} m/s^2   "
                    f"arctan(aT/g) {np.degrees(np.arctan(at_now / G)):4.1f} deg",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        strip = base_strip.copy()
        cv2.line(strip, (X(T), 8), (X(T), STRIP_H - 24), (60, 60, 220), 2)
        canvas[HEAD_H + PANE_H:] = strip
        vw.write(canvas)
    vw.release()
    pa.close()
    pr.close()
    print(f"trial {trial}: {out}   ({t1 - t0:.0f}s synced)")


trials = [int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2, 3, 4]
for n in trials:
    build(n)
