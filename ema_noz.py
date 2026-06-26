"""EMA of the horizontal magnitude sqrt(ax^2 + ay^2), with z excluded.
Usage: python ema_noz.py <file.csv> [-n SPAN | -a ALPHA] [--start T] [--end T] [--show]
Outputs land in an 'output' folder next to this script.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import accel

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="EMA of sqrt(ax^2 + ay^2), z excluded.")
    accel.add_common_args(ap)
    ap.add_argument("--auto", action="store_true", help="auto-trim to just the active ride")
    ap.add_argument("--log", action="store_true", help="log y-axis (exponential ramps look straight)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    alpha, tag = accel.resolve_alpha(args)

    df = accel.load(args.file)
    time_col = accel.find_time(df)
    cols = [c for c in df.columns if c != time_col]
    xcol, ycol = accel.pick_axis(cols, "x"), accel.pick_axis(cols, "y")
    if not xcol or not ycol:
        sys.exit(f"Couldn't find x and y columns in: {', '.join(cols)}")

    t, xlabel = accel.time_axis(df, time_col)

    ax_raw = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    ay_raw = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    axy = np.hypot(ax_raw, ay_raw)

    lo, hi = args.start, args.end
    if args.auto:
        a, b = accel.active_window(t, axy)
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

    axy_ema = accel.ema(axy, alpha)

    print(f"File:    {args.file}")
    print(f"Rows:    {len(t)}")
    print(f"x, y:    {xcol}  |  {ycol}   (z excluded)")
    print(f"EMA:     {tag}")
    if args.auto:
        print(f"Auto:    kept {t[0]:.1f}-{t[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(t, axy, color=accel.C_RAW, lw=1.0, alpha=0.6, label="raw")
    ax.plot(t, axy_ema, color=accel.C_EMA, lw=2.0, label="EMA")
    ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    if args.log:
        ax.set_yscale("log")
        pos = axy[np.isfinite(axy) & (axy > 0)]
        ax.set_ylim(max(1e-2, float(np.nanpercentile(pos, 1))), float(np.nanmax(axy)) * 1.3)
    else:
        ax.set_ylim(0, float(np.nanmax(axy)) * 1.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\sqrt{a_x^2 + a_y^2}$   (m/s$^2$)")
    accel.style_axis(ax)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle(f"Horizontal acceleration magnitude, z excluded  ({tag})",
                 fontsize=13, fontweight="bold")

    png, csv = accel.out_paths(args, HERE, "_xy_ema")
    fig.savefig(png, dpi=150)
    pd.DataFrame({xlabel: t, "a_xy": axy, "EMA a_xy": axy_ema}).to_csv(csv, index=False)
    print(f"Graph:   {png}")
    print(f"Data:    {csv}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
