# shared helpers for the accelerometer scripts

import argparse
import os

import numpy as np
import pandas as pd

SMOOTHING = 40

C_RAW = "#9aa0a6"
C_EMA = "#c0392b"
C_FIT = "#1f6fd6"
C_GRID = "#ededed"
C_ZERO = "#b0b0b0"


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


def pick_axis(cols, axis):
    import re
    for c in cols:
        if c.lower().replace(" ", "").startswith("a" + axis):
            return c
    for c in cols:
        n = c.lower()
        if "total" not in n and re.search(rf"(^|[^a-z]){axis}([^a-z]|$)", n):
            return c
    return None


def time_axis(df, time_col):
    if time_col:
        return pd.to_numeric(df[time_col], errors="coerce").to_numpy(), time_col
    return np.arange(len(df), dtype=float), "sample"


def ema(x, alpha):
    # NaNs hold the previous value instead of poisoning the rest
    y = np.empty(len(x))
    prev = None
    for i, v in enumerate(x):
        if np.isnan(v):
            y[i] = prev if prev is not None else np.nan
        else:
            y[i] = v if prev is None else alpha * v + (1 - alpha) * prev
        prev = y[i]
    return y


def smooth(x, alpha):
    # zero-phase: run the EMA forward then backward so the lag cancels
    f = ema(np.asarray(x, float), alpha)
    return ema(f[::-1], alpha)[::-1]


def active_window(t, mag, pad=3.0, level=0.015, smooth_s=2.0, gap_s=10.0):
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    w = max(1, int(round(smooth_s / dt)))
    env = pd.Series(np.abs(np.nan_to_num(mag))).rolling(w, center=True, min_periods=1).mean().to_numpy()
    base, peak = np.nanpercentile(env, 10), np.nanmax(env)
    active = env > base + level * (peak - base)
    gap = max(1, int(round(gap_s / dt)))
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


def add_common_args(ap):
    ap.add_argument("file", nargs="?", default="acceleration_data.csv")
    ap.add_argument("-n", "--span", type=int, default=SMOOTHING, help="EMA span; alpha = 2/(N+1)")
    ap.add_argument("-a", "--alpha", type=float, help="alpha directly (0-1), overrides --span")
    ap.add_argument("--start", type=float, help="drop data before this time")
    ap.add_argument("--end", type=float, help="drop data after this time")
    ap.add_argument("-o", "--out", help="exact PNG path")
    ap.add_argument("--outdir", help="output folder (default: ./output beside this script)")
    ap.add_argument("--show", action="store_true")
    return ap


def resolve_alpha(args):
    if args.alpha is not None:
        if not 0 < args.alpha <= 1:
            raise SystemExit("alpha must be between 0 and 1")
        return args.alpha, f"alpha={args.alpha:.3f}"
    if args.span < 1:
        raise SystemExit("span must be >= 1")
    alpha = 2 / (args.span + 1)
    return alpha, f"N={args.span}, alpha={alpha:.3f}"


def out_paths_for(out, outdir, here, infile, suffix):
    # plots go to output/visuals, the data csv stays in its own folder
    if out:
        png = out
        csv = os.path.splitext(out)[0] + ".csv"
    else:
        d = outdir or os.path.join(here, "output")
        base = os.path.splitext(os.path.basename(infile))[0]
        png = os.path.join(here, "output", "visuals", base + suffix + ".png")
        csv = os.path.join(d, base + suffix + ".csv")
    for p in (png, csv):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    return png, csv


def out_paths(args, here, suffix):
    return out_paths_for(args.out, args.outdir, here, args.file, suffix)


def style_axis(ax):
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, png_path, dpi=150):
    # png for quick viewing, svg alongside it for print quality
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(os.path.splitext(png_path)[0] + ".svg")
