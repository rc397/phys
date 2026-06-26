"""EMA smoothing for accelerometer logs (phyphox CSV, etc.).
Usage: python ema.py <file.csv> [-n SPAN | -a ALPHA] [--show]
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
    ap = argparse.ArgumentParser(description="Compute and plot an EMA of accelerometer data.")
    accel.add_common_args(ap)
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    alpha, tag = accel.resolve_alpha(args)

    df = accel.load(args.file)
    time_col = accel.find_time(df)
    signals = accel.find_signals(df, time_col)
    if not signals:
        sys.exit("No numeric data columns found")

    t, xlabel = accel.time_axis(df, time_col)

    if args.start is not None or args.end is not None:
        keep = np.ones(len(t), dtype=bool)
        if args.start is not None:
            keep &= t >= args.start
        if args.end is not None:
            keep &= t <= args.end
        df, t = df.loc[keep].reset_index(drop=True), t[keep]
        if len(t) == 0:
            sys.exit("No data left after trimming")

    print(f"File:    {args.file}")
    print(f"Rows:    {len(df)}")
    print(f"Time:    {time_col or '(sample index)'}")
    print(f"Signals: {', '.join(signals)}")
    print(f"EMA:     {tag}")

    out = {xlabel: t}
    smoothed = {}
    for c in signals:
        raw = pd.to_numeric(df[c], errors="coerce").to_numpy()
        smoothed[c] = accel.ema(raw, alpha)
        out[c] = raw
        out[f"EMA {c}"] = smoothed[c]

    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 2.6 * len(signals)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, signals):
        raw = out[c]
        ax.axhline(0, color=accel.C_ZERO, lw=0.8)
        ax.plot(t, raw, color=accel.C_RAW, lw=1.0, alpha=0.6, label="raw")
        ax.plot(t, smoothed[c], color=accel.C_EMA, lw=2.0, label="EMA")
        ax.set_ylabel(c)
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
        lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
        if lo >= 0:                       # magnitude channel sits on its baseline
            ax.set_ylim(0, hi * 1.08 if hi > 0 else 1)
        else:                             # signed channel keeps 0 in the middle
            m = max(abs(lo), abs(hi)) * 1.08
            ax.set_ylim(-m, m)
        accel.style_axis(ax)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"Acceleration: raw vs EMA  ({tag})", fontsize=13, fontweight="bold")

    png, csv = accel.out_paths(args, HERE, "_ema")
    fig.savefig(png, dpi=150)
    pd.DataFrame(out).to_csv(csv, index=False)
    print(f"Graph:   {png}")
    print(f"Data:    {csv}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
