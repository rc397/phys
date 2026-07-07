# phys - swing-ride motion analysis

Analysis of a rotating chair-swing fairground ride (the *Volare*, a wave swinger)
from two independent sources:

1. **Phone accelerometer** logs (phyphox, 100 Hz, ridden) - `ax, ay, az, aT` in m/s².
2. **Video** of the ride (two cameras) - the chains' fly-out angle θ(t), measured
   automatically frame by frame.

The physics linking them: the ride is a conical pendulum, so at fly-out angle θ
from vertical, **tan θ = a_horizontal / g**. The videos give θ independently of the
phone, so the model is tested rather than assumed. (The phone logs are linear
acceleration, so on the rider **θ_accel = arctan(aT / g)** at steady state.)

## Layout

```
accel.py         shared helpers: CSV loading, column sniffing, EMA, plot styling, output paths
ema.py           EMA-smooth every channel (ax/ay/az/aT), one panel each
ema_noz.py       EMA of the horizontal magnitude sqrt(ax^2+ay^2) (z excluded), with --auto / --log
volare_angle.py  fly-out angle theta(t) from a ride video, fully automatic
swing_angle.py   older tool for a planar (back-and-forth) pendulum ride
output/          generated PNGs + CSVs
*.csv            raw accelerometer recordings
Videos/          raw footage (not in git - too large)
```

The analysis scripts import the common code in `accel.py`, so loading/smoothing/
styling lives in exactly one place.

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

## Fly-out angle from video (volare_angle.py) - fully automatic

The ride is a rotating chair swing, i.e. a conical pendulum: at the fly-out angle
θ from vertical, tan θ = a_horizontal / g, so the video angle ties directly to the
accelerometer. `volare_angle.py` measures θ(t) from a video with **no clicking**:

```bash
python volare_angle.py "Videos/Alex's persepctive/probably trial 1.MOV" --annot
python volare_angle.py --all          # every video under Videos/, plus a summary table
```

How it works, per video:
1. **Auto-calibration**: the ride region from fast motion (blurred pair-differences,
   so camera micro-wobble on static edges is ignored), and the true vertical from
   long building edges in the background (handles camera roll).
2. **Measurement** (every frame): moving chairs are isolated by background
   subtraction; on each side the extreme blob is the side-on chair - a rope at
   azimuth φ shows tan θ' = tan θ·|cos φ|, so the extreme chair shows the TRUE
   side-on angle. From its seat, rays are scanned up-and-inward for the chain in
   the raw pixels: the winning ray must be **continuously covered** (crossing
   ropes fail), **thin** (the canopy fails), and agree with a robust line refit.
3. **Aggregation**: per time window the upper quartile per side, best side kept
   (occlusion only ever lowers a reading), giving θ'(t) with a p10-p90 band -
   the band also shows the rotor-tilt "wave".
4. **Elevation correction**: the swept chair ring's lower boundary is an ellipse
   arc whose axis ratio is sin(elevation); tan θ = tan θ'·cos(elevation). If the
   ring bottom is hidden by scenery the tool says so (confidence flag) - trust
   the apparent angle more than the corrected one in that case, or pass `--eps`.
5. **Accelerometer tie-in**: for trials with a phone log it overlays
   θ_accel = arctan(aT/g) on the video curve (aligned by cross-correlation) and
   compares the steady states.

Output per video: `output/<video>_angle.png` (θ(t) + band + steady state and
g·tan θ), a CSV, with `--annot` a check video with the protractor gauge drawn on
every measured chain, and `output/<video>_vs_accel.png` where phone data exists.

| flag | meaning |
|------|---------|
| `--all` | run every video under `Videos/`, write `output/flyout_summary.csv` |
| `--annot` | write a full-length check video with the angle gauge drawn on (large file, kept local) |
| `--debug` | dump calibration overlays + gate rejection counts |
| `--eps E` | override the camera elevation (deg) |
| `--win S` / `--step K` | aggregation window (default 3 s) / frame stride |
| `--start/--end T` | limit the analysed time range |
| `--recal` | ignore the cached auto-calibration sidecar |

### Validated accuracy

Against a synthetic wave-swinger (16 chairs, spinning patterned canopy, cluttered
background with vertical building edges and horizontal rails, camera elevation 15°,
camera roll 2°, sensor noise) with known ground truth, zero manual input:

| quantity | result | truth |
|----------|--------|-------|
| steady fly-out | 39.6 ± 0.3° | 40.00° |
| θ(t) RMS over ramp/plateau/spin-down | 1.7° | - |
| camera roll detected | +1.97° | +2.0° |
| camera elevation from the chair ring | 15.9° | 15° |

Uncertainty on real footage: the statistical precision is ~±1° (per-trial sd),
and the dominant systematic is the camera-elevation estimate, worth roughly ±3°
on the absolute angle (the tool prints a confidence flag; pass `--eps` if you
know the camera's elevation). On trial 1 the video gives 47.9° (Alex's camera,
elevation trusted) vs the phone's arctan(aT/g) = 50.9° - independent instruments
agreeing within that systematic.

## Planar swing tool (swing_angle.py)

An earlier tool for a back-and-forth (planar) pendulum ride: tracks the seat with
ORB feature matching (~0.02° on synthetic), fits the pivot from the swept arc, and
writes the swing angle + g·sin θ. Kept for reference; the Volare is a rotating
ride, so `volare_angle.py` is the tool for this project's videos.
