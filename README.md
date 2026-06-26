# phys - swing-ride motion analysis

Analysis of a swing/pendulum fairground ride (the *Volare*) from two independent
sources:

1. **Phone accelerometer** logs (phyphox, 100 Hz) - `ax, ay, az, aT` in m/s².
2. **Video** of the ride - the swing angle θ(t) measured frame by frame.

The point where they meet is the physics: the tangential acceleration a swing
should feel is **a = g·sin θ**. The video gives θ independently, so you can test
that hypothesis against the accelerometer instead of assuming it.

## Layout

```
accel.py        shared helpers: CSV loading, column sniffing, EMA, plot styling, output paths
ema.py          EMA-smooth every channel (ax/ay/az/aT), one panel each
ema_noz.py      EMA of the horizontal magnitude sqrt(ax^2+ay^2) (z excluded), with --auto / --log
swing_angle.py  measure swing angle theta(t) from a video, frame by frame
output/         generated PNGs + CSVs
*.csv           raw accelerometer recordings
```

`ema.py`, `ema_noz.py` and `swing_angle.py` all import the common code in `accel.py`,
so loading/smoothing/styling lives in exactly one place.

## Setup

```
pip install numpy pandas matplotlib opencv-python scipy
```

## Accelerometer analysis

```bash
# every channel, raw vs EMA
python ema.py 1st_Trial.csv

# horizontal magnitude, auto-trimmed to just the ride, log y to see the exponential spin-up
python ema_noz.py 1st_Trial.csv --auto --log -o output/1st_auto_log.png
```

Common flags: `-n SPAN` (EMA span, default 40; alpha = 2/(N+1)) or `-a ALPHA`
directly, `--start/--end T` to trim by time, `--auto` (ema_noz only) to keep just
the active ride, `--log` (ema_noz only) for a log y-axis. Every run writes a PNG
and a matching CSV to `output/`.

What the data shows: a queue/quiet period, an **exponential spin-up** of the swing
amplitude, a sustained plateau at ~15–20 m/s² (≈1.5–2 g), then spin-down.

## Video angle analysis

`swing_angle.py` measures the angle θ between the **straight part** (the vertical
rest line through the pivot) and the **swing arm**, by tracking the seat and
measuring its bearing from the pivot. It also writes **g·sin θ**, the tangential
acceleration the swing model predicts, ready to overlay on the accelerometer trace.

It's built for accuracy. By default it:
- tracks the seat with **rotation-invariant ORB matching** to a reference frame, so
  it keeps lock even as the seat spins, through motion blur, noise and compression;
- recovers a **sub-pixel** seat position each frame from a RANSAC similarity fit;
- **fits the pivot** from the whole swept arc by robust least-squares circle fit -
  typically to a fraction of a pixel, so **you only mark the seat, not the pivot**;
- drops low-confidence frames and prints a quality summary.

Easiest - drag a box round the seat on the first frame:

```bash
python swing_angle.py volare.mp4 --pick --annot
```

Headless - if you know the seat's pixel position on the first analysed frame:

```bash
python swing_angle.py volare.mp4 --point 640,560 --bbox 70 --annot
```

`--annot` writes `<video>_angle_annot.mp4` with the fitted pivot, the live arm and the
angle drawn on - **watch it first** to confirm the tracking followed the seat. A
`<video>.swing.json` sidecar remembers what you clicked, so a later rerun is just
`python swing_angle.py volare.mp4`.

### Options

| flag | meaning |
|------|---------|
| `--point X,Y` | seat position on the first analysed frame (**original-video** pixels) |
| `--pick` | drag a box round the seat interactively instead |
| `--bbox N` | seat box side in px (the patch ORB learns / the template) |
| `--track rigid` | ORB feature match **(default, most accurate; needs some texture)** |
| `--track template` | sub-pixel luma cross-correlation (good fallback for low-texture seats) |
| `--track color/flow/mil/manual` | HSV blob / optical flow / OpenCV box tracker / click each frame |
| `--pivot X,Y` `--no-fit-pivot` | give the pivot instead of fitting it (needed if the swing barely moves) |
| `--ref vertical \| X,Y` | "straight" reference: vertical (default) or a point down the mast |
| `--center` | measure from the swing's **mean (equilibrium)** position - cancels camera roll and a sloppy seed |
| `--min-conf F` | drop frames below F× the median tracking confidence (default 0.25) |
| `--rotate 90\|180\|270`, `--resize W` | rotate a sideways clip / downscale for speed (coords mapped through both) |
| `--start/--end T`, `--step K` | analyse a time range / every K-th frame |
| `--flip`, `--smooth N` | flip the angle sign / EMA span for the smoothed line (default 11) |

Numeric `--point/--pivot/--ref` are given in **original-video pixels** - the script
maps them through the same rotate+resize the frames go through. (`--pick` coords are
taken straight from the frames you click.)

Output: `output/<video>_angle.png` (angle and g·sin θ vs time) and a CSV with
`time, x, y, confidence, valid, angle_deg, angle_deg_ema, g_sin_theta`.

### Tips for a real ride video

- **Mark a textured part of the seat/gondola** (structure, railings, riders) - ORB
  needs texture. If the seat is a plain blob, use `--track template` instead.
- For **full accuracy** keep full resolution and `--step 1` (the default); use
  `--resize`/`--step` only to preview quickly.
- Use **`--center`** if the camera is slightly rolled or you're unsure your seed sits
  exactly on the seat centre - it measures from the swing's own equilibrium.
- Always **sanity-check the `--annot` video** before trusting the numbers.

### Validated accuracy

Against a synthetic HD swing with a *rotating textured seat*, sensor noise, motion
blur and known ground-truth angle:

| scenario | mean error | max error |
|----------|-----------:|----------:|
| fixed/stabilised camera, full res (default) | **0.02°** | 0.05° |
| lossy (mp4) vs lossless - no difference | 0.02° | 0.05° |
| `--resize 640` (speed mode) | 0.15° | - |

All driftless (correlation > 0.9999) and unaffected by `--rotate`/`--resize`.
