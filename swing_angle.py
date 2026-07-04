# Swing angle vs time from a video of a pendulum/swing ride.
# Tracks the seat (ORB matching, sub-pixel), fits the pivot from the swept arc,
# and outputs the angle from vertical + g*sin(theta) for the accelerometer comparison.
#   python swing_angle.py video.mp4 --pick --annot
#   python swing_angle.py video.mp4 --point 640,560 --bbox 70 --annot

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

import accel

G = 9.81  # m/s^2, for the g*sin(theta) tangential-acceleration prediction

HERE = os.path.dirname(os.path.abspath(__file__))


def signed_angle(ref, vec):
    """Signed angle (radians) from ref to vec, both 2-vectors in image pixels.
    Positive = vec is counter-clockwise from ref in screen coords."""
    ax, ay = ref
    bx, by = vec
    return np.arctan2(ax * by - ay * bx, ax * bx + ay * by)


def reference_vector(ref_arg, pivot):
    """The 'straight' direction. 'vertical' -> straight down in the image
    (y grows downward), otherwise a second point defining the mast direction."""
    if ref_arg is None or ref_arg == "vertical":
        return np.array([0.0, 1.0])
    bottom = parse_xy(ref_arg)
    v = np.array(bottom, float) - np.array(pivot, float)
    n = np.hypot(*v)
    if n == 0:
        sys.exit("--ref point coincides with the pivot")
    return v / n


def parse_xy(s):
    try:
        x, y = (float(v) for v in str(s).split(","))
        return (x, y)
    except Exception:
        raise SystemExit(f"expected X,Y but got: {s!r}")


def map_point(x, y, rotate, scale, w0, h0):
    """Map a point given in original-video pixels into the processed frame
    (same rotate-then-resize the frames go through). Verified against cv2.rotate."""
    if rotate == 90:                      # ROTATE_90_CLOCKWISE
        x, y = h0 - 1 - y, x
    elif rotate == 180:
        x, y = w0 - 1 - x, h0 - 1 - y
    elif rotate == 270:                   # ROTATE_90_COUNTERCLOCKWISE
        x, y = y, w0 - 1 - x
    return x * scale, y * scale


def fit_circle(x, y):
    """Robust least-squares circle fit. Returns (cx, cy, R, residual_px).

    Kasa algebraic fit for the initial guess, then a geometric refinement with a
    soft-L1 loss so a few stray tracked points don't drag the centre."""
    from scipy.optimize import least_squares
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.c_[x, y, np.ones_like(x)]
    a0, b0, c0 = np.linalg.lstsq(A, x**2 + y**2, rcond=None)[0]
    cx0, cy0 = a0 / 2, b0 / 2
    r0 = np.sqrt(max(c0 + cx0**2 + cy0**2, 1.0))

    def res(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    sol = least_squares(res, [cx0, cy0, r0], loss="soft_l1", f_scale=2.0)
    cx, cy, r = sol.x
    return cx, cy, r, float(np.std(res(sol.x)))


# Every tracker returns (pts[N,2] float, conf[N] float). Higher conf = better.
# track_rigid additionally returns a per-frame rotation (deg) as a third array;
# the simple trackers return None for it.

def track_rigid(frames, point, bbox, ratio=0.75, nfeat=1500):
    """Rotation-invariant ORB matching of the seat to the first frame.

    Detects ORB features in a box round `point` on frame 0, then every frame finds
    them again (within a search window) and fits a RANSAC similarity transform.
    The transform maps the seed point to a sub-pixel seat location and yields the
    seat's rotation. Robust to the seat spinning, to blur, noise and compression."""
    import cv2
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    H, W = gray[0].shape
    orb = cv2.ORB_create(nfeatures=nfeat, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    cx, cy = point
    b = bbox
    m0 = np.zeros((H, W), np.uint8)
    m0[max(0, int(cy - b)):int(cy + b), max(0, int(cx - b)):int(cx + b)] = 255
    kp_ref, des_ref = orb.detectAndCompute(gray[0], m0)
    if des_ref is None or len(kp_ref) < 8:
        sys.exit("rigid track: too few features on the seat; pick a more textured "
                 "patch / bigger --bbox, or use --track template")
    ref_pts = np.float32([k.pt for k in kp_ref])
    anchor = np.array([cx, cy], np.float32)

    pts, conf, rot = [], [], []
    last = anchor.copy()
    sw = b * 3
    for g in gray:
        x0, y0 = max(0, int(last[0] - sw)), max(0, int(last[1] - sw))
        x1, y1 = min(W, int(last[0] + sw)), min(H, int(last[1] + sw))
        mask = np.zeros((H, W), np.uint8)
        mask[y0:y1, x0:x1] = 255
        kp, des = orb.detectAndCompute(g, mask)
        ok = False
        if des is not None and len(kp) >= 6:
            pairs = bf.knnMatch(des_ref, des, k=2)
            good = [a for mm in pairs if len(mm) == 2
                    for a, bb in [mm] if a.distance < ratio * bb.distance]
            if len(good) >= 6:
                src = ref_pts[[gm.queryIdx for gm in good]]
                dst = np.float32([kp[gm.trainIdx].pt for gm in good])
                M, inl = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    a = M @ np.array([anchor[0], anchor[1], 1.0])
                    pts.append(a.astype(float))
                    rot.append(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                    conf.append(float(inl.sum()) if inl is not None else float(len(good)))
                    last = a.astype(np.float32)
                    ok = True
        if not ok:
            pts.append([float(last[0]), float(last[1])])
            rot.append(np.nan)
            conf.append(0.0)
    return np.array(pts), np.array(conf), np.array(rot)


def _subpix(res, mx):
    """Parabolic sub-pixel refinement of a matchTemplate peak."""
    x, y = mx
    h, w = res.shape
    dx = dy = 0.0
    if 0 < x < w - 1:
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
        d = a - 2 * b + c
        if d != 0:
            dx = 0.5 * (a - c) / d
    if 0 < y < h - 1:
        a, b, c = res[y - 1, x], res[y, x], res[y + 1, x]
        d = a - 2 * b + c
        if d != 0:
            dy = 0.5 * (a - c) / d
    return dx, dy


def track_template(frames, point, bbox, search=2.5):
    """Sub-pixel normalised cross-correlation template matching on luma."""
    import cv2
    half = bbox // 2
    x, y = int(point[0]), int(point[1])
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    H, W = g0.shape
    tmpl = g0[max(0, y - half):y + half, max(0, x - half):x + half]
    pts, conf = [[float(x), float(y)]], [1.0]
    cx, cy = x, y
    sr = int(bbox * search)
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        x0, y0 = max(0, cx - sr), max(0, cy - sr)
        x1, y1 = min(W, cx + sr), min(H, cy + sr)
        win = g[y0:y1, x0:x1]
        if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
            pts.append([float(cx), float(cy)])
            conf.append(0.0)
            continue
        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, mx = cv2.minMaxLoc(res)
        dx, dy = _subpix(res, mx)
        fx = x0 + mx[0] + dx + tmpl.shape[1] / 2.0
        fy = y0 + mx[1] + dy + tmpl.shape[0] / 2.0
        cx, cy = int(round(fx)), int(round(fy))
        pts.append([fx, fy])
        conf.append(float(peak))
    return np.array(pts), np.array(conf), None


def track_flow(frames, point):
    """Lucas-Kanade optical flow on the single seed point (sub-pixel)."""
    import cv2
    lk = dict(winSize=(31, 31), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    p = np.array([[point]], np.float32)
    pts, conf = [[float(point[0]), float(point[1])]], [1.0]
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        nxt, st, err = cv2.calcOpticalFlowPyrLK(prev, g, p, None, **lk)
        if st[0][0]:
            p = nxt
            pts.append([float(p[0][0][0]), float(p[0][0][1])])
            conf.append(1.0)
        else:
            pts.append(pts[-1])
            conf.append(0.0)
        prev = g
    return np.array(pts), np.array(conf), None


def track_color(frames, hsv_lo, hsv_hi):
    """Centroid of the largest blob inside an HSV colour range, per frame."""
    import cv2
    pts, conf = [], []
    last = None
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(mask, binaryImage=True)
            if M["m00"] > 0:
                last = [M["m10"] / M["m00"], M["m01"] / M["m00"]]
                pts.append(last)
                conf.append(float(cv2.contourArea(c)))
                continue
        pts.append(last if last is not None else [np.nan, np.nan])
        conf.append(0.0)
    return np.array(pts, float), np.array(conf), None


def make_tracker(kind):
    import cv2
    pref = {"mil": "TrackerMIL_create", "csrt": "TrackerCSRT_create",
            "kcf": "TrackerKCF_create"}[kind]
    for ctor in (pref, "TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        if hasattr(cv2, ctor):
            return getattr(cv2, ctor)()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, ctor):
            return getattr(cv2.legacy, ctor)()
    sys.exit(f"No box tracker ('{kind}') in this OpenCV build; use --track rigid")


def track_box(frames, point, bbox, kind):
    """Follow `point` with an OpenCV box tracker (mil/csrt/kcf)."""
    half = bbox // 2
    x, y = point
    tr = make_tracker(kind)
    tr.init(frames[0], (int(x - half), int(y - half), bbox, bbox))
    pts, conf = [[float(x), float(y)]], [1.0]
    for f in frames[1:]:
        ok, bb = tr.update(f)
        pts.append([bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0] if ok else pts[-1])
        conf.append(1.0 if ok else 0.0)
    return np.array(pts), np.array(conf), None


def track_manual(frames):
    """Click the seat on every analysed frame. ESC/q stops early."""
    import cv2
    pts = []
    win = "click the seat  (ESC to stop)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    click = {}
    cv2.setMouseCallback(win, lambda e, x, y, *_: click.update(p=(float(x), float(y)))
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    last = None
    for i, f in enumerate(frames):
        click.pop("p", None)
        disp = f.copy()
        cv2.putText(disp, f"frame {i+1}/{len(frames)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        if (cv2.waitKey(0) & 0xFF) in (27, ord("q")):
            break
        last = click.get("p", last)
        pts.append(last if last else [np.nan, np.nan])
    cv2.destroyAllWindows()
    while len(pts) < len(frames):
        pts.append(pts[-1] if pts else [np.nan, np.nan])
    return np.array(pts, float), np.ones(len(pts)), None


def pick_seat(frame):
    """Drag a box round the seat to track; returns (cx, cy, half_side)."""
    import cv2
    r = cv2.selectROI("drag a box round the SEAT, then ENTER", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = r
    if w == 0 or h == 0:
        sys.exit("no ROI selected")
    return x + w / 2.0, y + h / 2.0, int(max(w, h) / 2)


def pick_clicks(frame, labels):
    """Click a sequence of labelled points; returns list of (x, y)."""
    import cv2
    picks = []
    win = "click: " + " -> ".join(l.split()[0] for l in labels)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, lambda e, x, y, *_: picks.append((float(x), float(y)))
                         if e == cv2.EVENT_LBUTTONDOWN and len(picks) < len(labels) else None)
    while True:
        disp = frame.copy()
        for j, p in enumerate(picks):
            cv2.circle(disp, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)
        idx = len(picks)
        msg = labels[idx] if idx < len(labels) else "done - press ENTER"
        cv2.putText(disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and idx == len(labels):
            break
        if k == 27:
            sys.exit("cancelled")
    cv2.destroyAllWindows()
    return picks


def read_frames(path, rotate, resize_w, step, t0, t1):
    """Load (and pre-process) the frames we'll analyse plus their timestamps."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    frames, times = [], []
    w0 = h0 = None
    scale = 1.0
    i = -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if w0 is None:
            h0, w0 = f.shape[:2]
        t = i / fps if fps > 0 else float(i)
        if t0 is not None and t < t0:
            continue
        if t1 is not None and t > t1:
            break
        if i % step != 0:
            continue
        if rotate in rot_map:
            f = cv2.rotate(f, rot_map[rotate])
        if resize_w and f.shape[1] > resize_w:
            scale = resize_w / f.shape[1]
            f = cv2.resize(f, (resize_w, int(round(f.shape[0] * scale))))
        frames.append(f)
        times.append(t)
    cap.release()
    return frames, np.array(times, float), fps, scale, (w0, h0)


def write_annotated(frames, pivot, ref, pts, ang_deg, valid, out_mp4):
    """Draw the fitted pivot, the straight reference, the live arm and angle."""
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    px, py = int(round(pivot[0])), int(round(pivot[1]))
    reflen = 0.5 * h
    for f, p, a, v in zip(frames, pts, ang_deg, valid):
        d = f.copy()
        rb = (int(px + ref[0] * reflen), int(py + ref[1] * reflen))
        cv2.line(d, (px, py), rb, (170, 170, 170), 2, cv2.LINE_AA)
        if v and np.all(np.isfinite(p)):
            cv2.line(d, (px, py), (int(p[0]), int(p[1])), (40, 60, 200), 3, cv2.LINE_AA)
            cv2.circle(d, (int(p[0]), int(p[1])), 7, (40, 60, 200), -1, cv2.LINE_AA)
        cv2.circle(d, (px, py), 6, (0, 220, 0), -1, cv2.LINE_AA)
        if np.isfinite(a):
            cv2.putText(d, f"{a:+.2f} deg", (px + 12, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(d)
    vw.release()


def sidecar_path(video):
    return os.path.splitext(video)[0] + ".swing.json"


def load_labels(video):
    p = sidecar_path(video)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def save_labels(video, data):
    with open(sidecar_path(video), "w") as fh:
        json.dump(data, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Measure swing angle vs time from a ride video.")
    ap.add_argument("video")
    ap.add_argument("--point", help="seat pixel X,Y on the first analysed frame (original-video px)")
    ap.add_argument("--pivot", help="pivot X,Y; omit to FIT it from the swept arc (recommended)")
    ap.add_argument("--ref", default="vertical",
                    help="'vertical' (default) or X,Y of a point down the mast from the pivot")
    ap.add_argument("--track", default="rigid",
                    choices=["rigid", "template", "mil", "csrt", "kcf", "color", "flow", "manual"],
                    help="how to follow the seat (default: rigid = ORB, most accurate)")
    ap.add_argument("--hsv", help="colour range 'h,s,v:h,s,v' for --track color")
    ap.add_argument("--bbox", type=int, default=70, help="seat box side in px (seed/template)")
    ap.add_argument("--pick", action="store_true", help="drag a box round the seat on frame 1")
    ap.add_argument("--no-fit-pivot", action="store_true", help="use --pivot as-is, don't fit")
    ap.add_argument("--center", action="store_true",
                    help="measure angle from the swing's mean (equilibrium) position")
    ap.add_argument("--min-conf", type=float, default=0.25,
                    help="drop frames below this fraction of the median confidence")
    ap.add_argument("--start", type=float, help="analyse from this time (s)")
    ap.add_argument("--end", type=float, help="analyse up to this time (s)")
    ap.add_argument("--step", type=int, default=1, help="analyse every K-th frame")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--resize", type=int, help="downscale to this width (lower accuracy; default full res)")
    ap.add_argument("--fps", type=float, help="override frame rate")
    ap.add_argument("--flip", action="store_true", help="flip the sign of the angle")
    ap.add_argument("--smooth", type=int, default=11, help="EMA span for the smoothed angle")
    ap.add_argument("--annot", action="store_true", help="also write an annotated mp4")
    ap.add_argument("-o", "--out", help="exact PNG path")
    ap.add_argument("--outdir", help="output folder (default ./output)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    frames, times, fps_read, scale, (w0, h0) = read_frames(
        args.video, args.rotate, args.resize, max(1, args.step), args.start, args.end)
    if len(frames) < 2:
        sys.exit("Need at least 2 frames in the selected range")
    fps = args.fps or fps_read or 25.0

    def to_proc(p):
        return map_point(p[0], p[1], args.rotate, scale, w0, h0)

    # --- resolve the seat point, pivot and reference (sidecar > flags > pick) -----
    labels = load_labels(args.video)
    need_ref_point = args.ref not in (None, "vertical")
    pivot_xy = None
    if args.pick:
        cx, cy, half = pick_seat(frames[0])
        point = (cx, cy)
        args.bbox = max(args.bbox, 2 * half)
        extra = []
        if not need_ref_point and args.no_fit_pivot:
            extra = ["PIVOT (rotation centre)"]
        elif need_ref_point:
            extra = ["PIVOT (rotation centre)", "MAST point (straight down from pivot)"]
        if extra:
            clk = pick_clicks(frames[0], extra)
            pivot_xy = clk[0]
            if need_ref_point:
                args.ref = f"{clk[1][0]},{clk[1][1]}"
                need_ref_point = True
        save_labels(args.video, {"point": list(point), "bbox": args.bbox,
                                 "pivot": list(pivot_xy) if pivot_xy else None, "ref": args.ref})
    elif args.point:
        point = to_proc(parse_xy(args.point))
        if args.pivot:
            pivot_xy = to_proc(parse_xy(args.pivot))
    elif labels:
        point = to_proc(tuple(labels["point"]))
        args.bbox = labels.get("bbox", args.bbox)
        if labels.get("pivot"):
            pivot_xy = to_proc(tuple(labels["pivot"]))
        args.ref = labels.get("ref", args.ref)
        need_ref_point = args.ref not in (None, "vertical")
        print(f"(loaded labels from {os.path.basename(sidecar_path(args.video))})")
    else:
        sys.exit("Give --point X,Y (seat), or --pick to draw a box, "
                 "or create a sidecar with a previous --pick run")

    # --- track the seat ---------------------------------------------------------
    if args.track == "rigid":
        pts, conf, rot = track_rigid(frames, point, args.bbox)
    elif args.track == "template":
        pts, conf, rot = track_template(frames, point, args.bbox)
    elif args.track == "flow":
        pts, conf, rot = track_flow(frames, point)
    elif args.track == "color":
        if not args.hsv:
            sys.exit("--track color needs --hsv 'h,s,v:h,s,v'")
        lo, hi = (tuple(int(v) for v in part.split(",")) for part in args.hsv.split(":"))
        pts, conf, rot = track_color(frames, lo, hi)
    elif args.track == "manual":
        pts, conf, rot = track_manual(frames)
    else:
        pts, conf, rot = track_box(frames, point, args.bbox, args.track)

    # --- confidence -> which frames to trust ------------------------------------
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    med = np.median(conf[conf > 0]) if np.any(conf > 0) else 0.0
    valid = finite & (conf >= args.min_conf * med if med > 0 else finite)
    if valid.sum() < 3:
        sys.exit("Tracking failed on almost every frame; try --track template, a "
                 "bigger --bbox, or trim to a cleaner stretch with --start/--end")

    # --- pivot: fit from the arc (default) or take the given one ----------------
    fit_resid = spread = None
    if pivot_xy is not None and args.no_fit_pivot:
        pivot = np.array(pivot_xy, float)
        pivot_src = "given"
    else:
        cx, cy, R, fit_resid = fit_circle(pts[valid, 0], pts[valid, 1])
        bear = np.degrees(np.arctan2(pts[valid, 0] - cx, pts[valid, 1] - cy))
        spread = float(bear.max() - bear.min())
        if spread < 8.0 and pivot_xy is not None:    # arc too short to trust the fit
            pivot = np.array(pivot_xy, float)
            pivot_src = "given (arc too short to fit)"
        else:
            pivot = np.array([cx, cy])
            pivot_src = "fitted"
            if spread < 8.0:
                print("WARNING: small angular sweep -- fitted pivot may be imprecise; "
                      "pass --pivot X,Y --no-fit-pivot if you have a better one")

    ref = reference_vector(args.ref, pivot)

    # --- angle per frame --------------------------------------------------------
    arm = pts - pivot
    ang = np.array([signed_angle(ref, v) for v in arm])
    ang_deg = np.degrees(ang)
    ang_deg[~valid] = np.nan
    if args.flip:
        ang_deg = -ang_deg

    centre_off = 0.0
    if args.center:
        centre_off = np.nanmean(ang_deg)
        ang_deg = ang_deg - centre_off

    # interpolate across dropped frames for a continuous smoothed line
    filled = ang_deg.copy()
    if np.any(~valid):
        good = np.isfinite(filled)
        filled[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), filled[good])
    ang_ema = accel.ema(filled, 2 / (args.smooth + 1))
    a_tan = G * np.sin(np.radians(ang_deg))

    # --- report -----------------------------------------------------------------
    amp = (np.nanmax(ang_deg) - np.nanmin(ang_deg)) / 2 if valid.any() else float("nan")
    print(f"Video:   {args.video}")
    print(f"Frames:  {len(frames)} @ {fps:.2f} fps (step {args.step})   "
          f"trusted {valid.sum()}/{len(frames)} ({100*valid.mean():.0f}%)")
    print(f"Track:   {args.track}   median conf {med:.0f}")
    print(f"Pivot:   {pivot_src}  ({pivot[0]:.1f}, {pivot[1]:.1f})"
          + (f"   fit residual {fit_resid:.3f}px, arc {spread:.0f} deg" if fit_resid is not None else ""))
    print(f"Angle:   {np.nanmin(ang_deg):+.2f} .. {np.nanmax(ang_deg):+.2f} deg   amplitude ~{amp:.2f} deg"
          + (f"   (centred, was offset {centre_off:+.2f})" if args.center else ""))

    # --- plot -------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    a1.axhline(0, color=accel.C_ZERO, lw=0.8)
    a1.plot(times, ang_deg, color=accel.C_RAW, lw=1.0, alpha=0.7, label="raw")
    a1.plot(times, ang_ema, color=accel.C_EMA, lw=2.0, label=f"EMA (N={args.smooth})")
    if np.any(~valid):
        a1.plot(times[~valid], np.full((~valid).sum(), np.nanmin(ang_deg)),
                "x", color="#888", ms=4, label="dropped")
    a1.set_ylabel("swing angle  (deg)")
    a1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    accel.style_axis(a1)
    a2.axhline(0, color=accel.C_ZERO, lw=0.8)
    a2.plot(times, a_tan, color=accel.C_FIT, lw=1.5)
    a2.set_ylabel(r"$g\,\sin\theta$  (m/s$^2$)")
    a2.set_xlabel("time (s)")
    accel.style_axis(a2)
    a1.set_xlim(float(times[0]), float(times[-1]))
    ttl = "Swing angle from video  (" + ("centred on equilibrium" if args.center
                                         else "straight = rest vertical") + ")"
    fig.suptitle(ttl, fontsize=13, fontweight="bold")

    png, csv = accel.out_paths_for(args.out, args.outdir, HERE, args.video, "_angle")
    fig.savefig(png, dpi=150)
    pd.DataFrame({"time": times, "x": pts[:, 0], "y": pts[:, 1], "confidence": conf,
                  "valid": valid.astype(int), "angle_deg": ang_deg,
                  "angle_deg_ema": ang_ema, "g_sin_theta": a_tan}).to_csv(csv, index=False)
    print(f"Graph:   {png}")
    print(f"Data:    {csv}")

    if args.annot:
        mp4 = os.path.splitext(png)[0] + "_annot.mp4"
        write_annotated(frames, pivot, ref, pts, ang_deg, valid, mp4)
        print(f"Video:   {mp4}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
