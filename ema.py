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


def find_signals(df, time_col):
    return [c for c in df.columns
            if c != time_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]


def ema(x, alpha):
    # y_t = a*x_t + (1-a)*y_(t-1),  seeded with the first sample
    y = np.empty(len(x))
    prev = None
    for i, v in enumerate(x):
        if np.isnan(v):
            y[i] = prev if prev is not None else np.nan
        else:
            y[i] = v if prev is None else alpha * v + (1 - alpha) * prev
        prev = y[i]
    return y


def main():
    ap = argparse.ArgumentParser(description="Compute and plot an EMA of accelerometer data.")
    ap.add_argument("file", nargs="?", default="acceleration_data.csv")
    ap.add_argument("-n", "--span", type=int, default=SMOOTHING, help="EMA span; alpha = 2/(N+1)")
    ap.add_argument("-a", "--alpha", type=float, help="alpha directly (0-1), overrides --span")
    ap.add_argument("--start", type=float, help="drop data before this time")
    ap.add_argument("--end", type=float, help="drop data after this time")
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
    signals = find_signals(df, time_col)
    if not signals:
        sys.exit("No numeric data columns found")

    if time_col:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        xlabel = time_col
    else:
        t = np.arange(len(df), dtype=float)
        xlabel = "sample"

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
        smoothed[c] = ema(raw, alpha)
        out[c] = raw
        out[f"EMA {c}"] = smoothed[c]

    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 2.6 * len(signals)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, signals):
        raw = out[c]
        ax.axhline(0, color="#b0b0b0", lw=0.8)
        ax.plot(t, raw, color="#9aa0a6", lw=1.0, alpha=0.6, label="raw")
        ax.plot(t, smoothed[c], color="#c0392b", lw=2.0, label="EMA")
        ax.set_ylabel(c)
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
        lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
        if lo >= 0:                       # magnitude channel sits on its baseline
            ax.set_ylim(0, hi * 1.08 if hi > 0 else 1)
        else:                             # signed channel keeps 0 in the middle
            m = max(abs(lo), abs(hi)) * 1.08
            ax.set_ylim(-m, m)
        ax.grid(True, color="#ededed", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"Acceleration: raw vs EMA  ({tag})", fontsize=13, fontweight="bold")

    if args.out:
        png = args.out
        csv = os.path.splitext(args.out)[0] + ".csv"
    else:
        d = args.outdir or os.path.join(HERE, "output")
        base = os.path.splitext(os.path.basename(args.file))[0]
        png = os.path.join(d, base + "_ema.png")
        csv = os.path.join(d, base + "_ema.csv")
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)

    fig.savefig(png, dpi=150)
    pd.DataFrame(out).to_csv(csv, index=False)
    print(f"Graph:   {png}")
    print(f"Data:    {csv}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
