"""EMA of the horizontal magnitude sqrt(ax^2 + ay^2), with z excluded.
Usage: python ema_noz.py <file.csv> [-n SPAN | -a ALPHA] [--start T] [--end T] [--show]
Outputs land in an 'output' folder next to this script.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

SMOOTHING = 40  # higher = smoother (alpha = 2/(SMOOTHING+1))


def load(path):
    df = pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_time(df):
    for c in df.columns:
        if c.lower().startswith("time") or c.lower() in ("t", "t (s)", "t(s)"):
            return c
    first = pd.to_numeric(df[df.columns[0]], errors="coerce")
    if first.notna().all() and first.is_monotonic_increasing:
        return df.columns[0]
    return None


def pick_axis(cols, axis):
    for c in cols:
        if c.lower().replace(" ", "").startswith("a" + axis):   # ax, ay, az
            return c
    for c in cols:                                              # fallback: stray axis token, never the total
        n = c.lower()
        if "total" not in n and re.search(rf"(^|[^a-z]){axis}([^a-z]|$)", n):
            return c
    return None


def ema(x, alpha):
    # y_t = a*x_t + (1-a)*y_(t-1), seeded with the first sample
    y = np.empty(len(x))
    prev = None
    for i, v in enumerate(x):
        if np.isnan(v):
            y[i] = prev if prev is not None else np.nan
        else:
            y[i] = v if prev is None else alpha * v + (1 - alpha) * prev
        prev = y[i]
    return y


def active_window(t, mag, pad=3.0, level=0.015, smooth_s=2.0, gap_s=10.0):
    # envelope of |signal|; keep the active span, bridging brief quiet dips
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    w = max(1, int(round(smooth_s / dt)))
    env = pd.Series(np.abs(np.nan_to_num(mag))).rolling(w, center=True, min_periods=1).mean().to_numpy()
    base, peak = np.nanpercentile(env, 10), np.nanmax(env)
    active = env > base + level * (peak - base)
    gap = max(1, int(round(gap_s / dt)))     # only a sustained quiet stretch ends the ride
    pk = int(np.nanargmax(env))

    def edge(step):
        i, last, quiet = pk, pk, 0
        while 0 <= i + step < len(env):
            i += step
            if active[i]:
                last, quiet = i, 0
            else:
                quiet += 1
                if quiet >= gap:
                    break
        return last

    lo_i, hi_i = edge(-1), edge(1)
    return max(t[0], t[lo_i] - pad), min(t[-1], t[hi_i] + pad)


def main():
    ap = argparse.ArgumentParser(description="EMA of sqrt(ax^2 + ay^2), z excluded.")
    ap.add_argument("file", nargs="?", default="acceleration_data.csv")
    ap.add_argument("-n", "--span", type=int, default=SMOOTHING, help="EMA span; alpha = 2/(N+1)")
    ap.add_argument("-a", "--alpha", type=float, help="alpha directly (0-1), overrides --span")
    ap.add_argument("--start", type=float, help="drop data before this time")
    ap.add_argument("--end", type=float, help="drop data after this time")
    ap.add_argument("--auto", action="store_true", help="auto-trim to just the active ride")
    ap.add_argument("--log", action="store_true", help="log y-axis (exponential ramps look straight)")
    ap.add_argument("-o", "--out", help="exact PNG path")
    ap.add_argument("--outdir", help="output folder (default: ./output beside this script)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    if args.alpha is not None:
        if not 0 < args.alpha <= 1:
            sys.exit("alpha must be between 0 and 1")
        alpha, tag = args.alpha, f"alpha={args.alpha:.3f}"
    else:
        if args.span < 1:
            sys.exit("span must be >= 1")
        alpha = 2 / (args.span + 1)
        tag = f"N={args.span}, alpha={alpha:.3f}"

    df = load(args.file)
    time_col = find_time(df)
    cols = [c for c in df.columns if c != time_col]
    xcol, ycol = pick_axis(cols, "x"), pick_axis(cols, "y")
    if not xcol or not ycol:
        sys.exit(f"Couldn't find x and y columns in: {', '.join(cols)}")

    if time_col:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        xlabel = time_col
    else:
        t = np.arange(len(df), dtype=float)
        xlabel = "sample"

    ax_raw = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    ay_raw = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    axy = np.hypot(ax_raw, ay_raw)

    lo, hi = args.start, args.end
    if args.auto:
        a, b = active_window(t, axy)
        lo = a if lo is None else max(lo, a)
        hi = b if hi is None else min(hi, b)

    if lo is not None or hi is not None:
        keep = np.ones(len(t), dtype=bool)
        if lo is not None:
            keep &= t >= lo
        if hi is not None:
            keep &= t <= hi
        t, axy = t[keep], axy[keep]
        if len(t) == 0:
            sys.exit("No data left after trimming")

    axy_ema = ema(axy, alpha)

    print(f"File:    {args.file}")
    print(f"Rows:    {len(t)}")
    print(f"x, y:    {xcol}  |  {ycol}   (z excluded)")
    print(f"EMA:     {tag}")
    if args.auto:
        print(f"Auto:    kept {t[0]:.1f}-{t[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(t, axy, color="#9aa0a6", lw=1.0, alpha=0.6, label="raw")
    ax.plot(t, axy_ema, color="#c0392b", lw=2.0, label="EMA")
    ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    if args.log:
        ax.set_yscale("log")
        pos = axy[np.isfinite(axy) & (axy > 0)]
        ax.set_ylim(max(1e-2, float(np.nanpercentile(pos, 1))), float(np.nanmax(axy)) * 1.3)
    else:
        ax.set_ylim(0, float(np.nanmax(axy)) * 1.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\sqrt{a_x^2 + a_y^2}$   (m/s$^2$)")
    ax.grid(True, color="#ededed", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle(f"Horizontal acceleration magnitude, z excluded  ({tag})",
                 fontsize=13, fontweight="bold")

    if args.out:
        png = args.out
        csv = os.path.splitext(args.out)[0] + ".csv"
    else:
        d = args.outdir or os.path.join(HERE, "output")
        base = os.path.splitext(os.path.basename(args.file))[0]
        png = os.path.join(d, base + "_xy_ema.png")
        csv = os.path.join(d, base + "_xy_ema.csv")
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)

    fig.savefig(png, dpi=150)
    pd.DataFrame({xlabel: t, "a_xy": axy, "EMA a_xy": axy_ema}).to_csv(csv, index=False)
    print(f"Graph:   {png}")
    print(f"Data:    {csv}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
