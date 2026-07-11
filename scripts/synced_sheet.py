# Build one synced spreadsheet per trial: the phone acceleration and both
# cameras' fly-out angles resampled onto a single shared time column, using
# the offsets in output/report/camera_sync.json. Every row is the same real
# instant across all three instruments, so it charts straight in Excel with
# no manual lining-up. Missing cells (a camera wasn't rolling yet) are blank.
#   python scripts/synced_sheet.py
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accel

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHONE = {"1": "1st Trial.csv", "2": "2nd Trial.csv", "3": "3rd Trial.csv", "4": "4th Trial.csv"}
STEP = 0.1  # shared clock resolution (s)


def angle_csv(trial, cam):
    hits = glob.glob(os.path.join(ROOT, "output", "angles", f"*trial {trial}_angle_{cam}.csv"))
    return hits[0] if hits else None


def ride_start(tp, aT):
    # first moment of the longest sustained flying stretch (aT high), so t_ride
    # is measured from when the actual ride spins up, not a stray early spike
    th = np.degrees(np.arctan(accel.ema(aT, 2 / 301) / G))
    g = np.arange(tp.min(), tp.max(), 0.1)
    on = np.interp(g, tp, np.nan_to_num(th)) > 15
    segs, i = [], 0
    while i < len(g):
        if on[i]:
            j = i
            while j + 1 < len(g) and on[j + 1]:
                j += 1
            if not segs or g[i] - segs[-1][1] > 20:
                segs.append([g[i], g[j]])
            else:
                segs[-1][1] = g[j]
            i = j + 1
        else:
            i += 1
    return max(segs, key=lambda s: s[1] - s[0])[0] if segs else tp.min()


with open(os.path.join(ROOT, "output", "report", "camera_sync.json")) as f:
    table = json.load(f)
outdir = os.path.join(ROOT, "output", "report")

for n in ("1", "2", "3", "4"):
    row = table["trials"][n]
    lag_a, lag_r = row["lag_alex"], row["lag_ryan"]

    a = accel.load(os.path.join(ROOT, "data", PHONE[n]))
    tp = pd.to_numeric(a[accel.find_time(a)], errors="coerce").to_numpy()
    at_col = next(c for c in a.columns if c.strip().lower().startswith("at"))
    aT = pd.to_numeric(a[at_col], errors="coerce").to_numpy()
    aT_s = accel.ema(aT, 2 / 301)

    va = pd.read_csv(angle_csv(n, "alex"))
    vr = pd.read_csv(angle_csv(n, "ryan"))
    # each video, placed on the phone clock: phone_time = video_time + lag
    a_span = (va["time"].min() + lag_a, va["time"].max() + lag_a)
    r_span = (vr["time"].min() + lag_r, vr["time"].max() + lag_r)

    # shared clock covers where the phone overlaps either camera (the ride)
    t0 = max(tp.min(), min(a_span[0], r_span[0]))
    t1 = min(tp.max(), max(a_span[1], r_span[1]))
    grid = np.round(np.arange(t0, t1 + STEP, STEP), 3)
    t_ride = ride_start(tp, aT)

    def video_on_grid(v, lag, col):
        # video column sampled at grid; blank where the camera had no footage
        x = np.interp(grid, v["time"].to_numpy() + lag, v[col].to_numpy(),
                      left=np.nan, right=np.nan)
        return x

    out = pd.DataFrame({
        "time_s": grid,                                   # phone clock
        "t_from_ride_start_s": np.round(grid - t_ride, 3),
        "phone_aT_ms2": np.round(np.interp(grid, tp, np.nan_to_num(aT_s)), 4),
        "phone_angle_deg": np.round(np.degrees(np.arctan(
            np.interp(grid, tp, np.nan_to_num(aT_s)) / G)), 3),
        "alex_video_angle_deg": np.round(video_on_grid(va, lag_a, "theta_ema"), 3),
        "ryan_video_angle_deg": np.round(video_on_grid(vr, lag_r, "theta_ema"), 3),
    })
    out_path = os.path.join(outdir, f"trial {n}_synced_data.csv")
    out.to_csv(out_path, index=False)
    cov_a = int(np.isfinite(out["alex_video_angle_deg"]).sum())
    cov_r = int(np.isfinite(out["ryan_video_angle_deg"]).sum())
    print(f"trial {n}: {out_path}")
    print(f"   {len(out)} rows  {grid[0]:.1f}-{grid[1 if len(grid) < 2 else -1]:.1f}s "
          f"phone clock   ride start {t_ride:.1f}s   "
          f"alex {cov_a} rows, ryan {cov_r} rows with angle")
