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
scripts/
  volare_angle.py   fly-out angle theta(t) from a ride video, fully automatic
  trial_graphs.py   per-trial report figures: angle + acceleration on a shared clock
  synced_video.py   both cameras + the accelerometer trace playing on one clock
  vidsync.py        that clock: sync anchored on the recordings' own timestamps
  ema.py            EMA-smooth every accelerometer channel, one panel each
  ema_noz.py        EMA of the horizontal magnitude sqrt(ax^2+ay^2)
  accel.py          shared helpers: CSV loading, EMA, plot styling, output paths
data/               the four phone recordings, one per trial
output/
  angles/           per-video theta(t) plots + CSVs (+ check videos, local only)
  accel/            video-vs-phone overlay per video
  report/           trial figures + flyout_summary.csv
Videos/             raw footage (not in git - too large)
```

## Setup

```
pip install numpy pandas matplotlib opencv-python scipy
```

## Fly-out angle from video - fully automatic

`scripts/volare_angle.py` measures θ(t) from a video with no clicking:

```bash
python scripts/volare_angle.py "Videos/Alex's persepctive/probably trial 1.MOV" --annot
python scripts/volare_angle.py --all      # every video under Videos/, plus a summary table
```

How it works, per video:
1. **Auto-calibration**: the ride region from fast motion (blurred pair-differences,
   so camera micro-wobble on static edges is ignored), and the true vertical from
   long building edges in the background (handles camera roll).
2. **Measurement** (every frame): moving chairs are isolated by background
   subtraction; on each side the outermost blob is the side-on chair, and rays are
   scanned above its seat for the chain - the winning ray must be continuously
   covered (crossing ropes fail), thin (the canopy fails), and agree with a robust
   line refit. A per-side lock re-fits the same chain on later frames so the check
   video reads continuously.
3. **Ride-state detection**: measurements only count while the ride actually
   rotates. Tent-seam flutter on the parked canopy, boarding crowds and passing
   clouds all produce motion, so rotation requires ride-wide fast motion that
   persists and sweeps out the chair ring; parked windows report as rest.
4. **Aggregation**: per time window the upper quantile per side, best side kept
   (occlusion only ever lowers a reading), giving θ'(t) with a p10-p90 band -
   the band also shows the rotor-tilt "wave".
5. **Elevation correction**: the swept chair ring's lower boundary is an ellipse
   arc whose axis ratio is sin(elevation); the steepest-apparent-chain formula
   then inverts to the true angle. If the ring bottom is hidden by scenery the
   tool says so (confidence flag) - pass `--eps` if you know the elevation.
6. **Accelerometer tie-in**: for trials with a phone log it overlays
   θ_accel = arctan(aT/g) on the video curve (clocks aligned from the
   recording stamps - see the sync note under the synced video below).

Outputs land in `output/angles/` and `output/accel/`; `--all` also writes
`output/report/flyout_summary.csv`.

| flag | meaning |
|------|---------|
| `--all` | run every video under `Videos/`, write the summary table |
| `--annot` | write a full-length check video with the angle gauge drawn on (large file, kept local) |
| `--debug` | dump calibration overlays + gate rejection counts |
| `--eps E` | override the camera elevation (deg) |
| `--win S` / `--step K` | aggregation window (default 3 s) / frame stride |
| `--start/--end T` | limit the analysed time range |
| `--recal` | ignore the cached auto-calibration sidecar |

## Report figures

```bash
python scripts/trial_graphs.py
```

writes `output/report/trial N_theta_accel.png` for each trial: fly-out angle over
time (video and phone side by side) above the rider's acceleration over time, on
a shared clock. Spans where the ride spins without the phone rider aboard (empty
warm-up and loading spins) are shaded - the camera sees the ride, the phone in
the queue does not, so disagreement there is expected rather than error.

## Synced three-view video

```bash
python scripts/synced_video.py        # or:  python scripts/synced_video.py 2
```

writes `output/report/trial N_synced.mp4`: both cameras playing side by side on
one clock with the rider's accelerometer trace scrolling underneath and a cursor
marking the current moment. Large files, kept local.

**How the clocks line up** (`scripts/vidsync.py`): correlating the video signals
blind is unreliable here - the ride is periodic, so a cross-correlation happily
locks a whole rotation (or a whole spin cycle) off, which is exactly what an
earlier version did. The sync is layered instead. Coarse: the recordings' own
clocks - each phyphox log states when it started, Ryan's phone stamps every mp4
when recording stops, and Alex's iPhone writes the recording start into each MOV
(that clock ran a constant ~41 s fast on the day; fitted out). Fine: audio -
both cameras hear the same PA and crowd transients, so the soundtracks' onset
patterns are correlated in short chunks and a dominant vote cluster pins the
offset to ~0.1 s with no viewpoint effects (re-deriving this needs
`imageio-ffmpeg` for the audio decode). Trials whose audio vote was ambiguous
were confirmed frame-by-frame against the rotor's spin-up lift before being
pinned. The resolved table lives in `output/report/camera_sync.json`, marked
per trial with which source decided it.

## Accelerometer-only plots

```bash
python scripts/ema.py "data/1st Trial.csv"
python scripts/ema_noz.py "data/1st Trial.csv" --auto --log
```

## Validated accuracy

Against a synthetic wave-swinger (16 chairs, spinning patterned canopy, cluttered
background with vertical building edges and horizontal rails, camera elevation 15°,
camera roll 2°, sensor noise) with known ground truth, zero manual input:

| quantity | result | truth |
|----------|--------|-------|
| steady fly-out | 39.8 ± 0.5° | 40.00° |
| θ(t) RMS over ramp/plateau/spin-down | 1.7° | - |
| camera roll detected | +1.97° | +2.0° |
| camera elevation from the chair ring | 15.9° | 15° |

On the real footage: statistical precision is ~±1-2° per trial; the dominant
systematic is each camera's elevation estimate, worth several degrees on the
absolute angle (Alex's camera reads low, Ryan's high, the phone in between).
Across four trials and three instruments the steady fly-out is **θ ≈ 50 ± 4°**,
giving a horizontal acceleration of **g·tan θ ≈ 11-13 m/s²**, consistent with
the conical-pendulum model. Brief wind gusts can still leave an isolated spurious
window or two during idle stretches; the phone trace beside the curve makes them
obvious.
