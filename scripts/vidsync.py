# Clock alignment between the three instruments. One rule everywhere:
# the two cameras are synced to EACH OTHER on the ride's motion fingerprint
# (both watched the same ride, so how-much-moves-over-time is a near-identical
# curve in both, including the empty spins), and the pair is anchored to the
# phone through the alex camera's angle curve. Correlating each camera against
# the phone separately is unreliable, because the phone only shares the loaded
# run - it sits in the queue during empty spins.
import glob
import os
import re

import cv2
import numpy as np


def xcorr_lag(t1, s1, t2, s2, grid=0.5):
    # lag such that series2_time = series1_time + lag, at the best overlap
    g1 = np.arange(t1.min(), t1.max(), grid)
    g2 = np.arange(t2.min(), t2.max(), grid)
    x1 = np.interp(g1, t1, s1)
    x2 = np.interp(g2, t2, s2)
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-9)
    x2 = (x2 - x2.mean()) / (x2.std() + 1e-9)
    c = np.correlate(x2, x1, "full")
    return float((np.argmax(c) - (len(x1) - 1)) * grid + g2[0] - g1[0])


def motion_series(path):
    # how much of the scene moves, over time; cached because it needs a decode
    cache = os.path.splitext(path)[0] + ".volare_motion.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return z["t"], z["m"]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / 8)))
    ts, ms, prev, fi = [], [], None, -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fi += 1
        if fi % step:
            continue
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (480, 270)),
                                          cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev is not None:
            ts.append(fi / fps)
            ms.append(int(np.count_nonzero(cv2.absdiff(g, prev) > 16)))
        prev = g
    cap.release()
    t, m = np.array(ts), np.array(ms, float)
    np.savez_compressed(cache, t=t, m=m)
    return t, m


def video_offset(path_a, path_b):
    # lag such that b_time = a_time + offset, from the motion fingerprints
    ta, ma = motion_series(path_a)
    tb, mb = motion_series(path_b)
    return xcorr_lag(ta, ma, tb, mb)


def lag_via_alex(video, ta, th_phone, root):
    # phone_time = video_time + lag for a non-alex camera, anchored through the
    # alex camera of the same trial; None when the sibling pieces are missing
    m = re.search(r"trial \d", os.path.basename(video).lower())
    if not m:
        return None
    key = m.group(0)
    avid = glob.glob(os.path.join(root, "Videos", "*lex*", f"*{key}.MOV"))
    acsv = glob.glob(os.path.join(root, "output", "angles", f"*{key}_angle_alex.csv"))
    if not avid or not acsv:
        return None
    import pandas as pd
    v = pd.read_csv(acsv[0])
    lag_a = xcorr_lag(v["time"].to_numpy(), np.nan_to_num(v["theta_ema"].to_numpy()),
                      ta, th_phone)
    return lag_a - video_offset(avid[0], video)
