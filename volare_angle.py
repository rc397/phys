# Fly-out angle of the chair-swing (Volare) from video.
# Angle between the ride's vertical (marked with 2 dots on the central support, since
# the rotor tilts / the camera may be rolled) and the rope holding a chair as it swings.
# Conical pendulum: tan(theta) = a_horizontal / g, which ties to the accelerometer.
#   python volare_angle.py "trial 1.mp4" --pick --watch    # auto-detect at a checkpoint box
#   python volare_angle.py "trial 1.mp4" --pick --mark      # click the rope each pass

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

import accel

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))


def sidecar(video):
    return os.path.splitext(video)[0] + ".volare.json"


def down(v):
    """Orient a 2-vector to point downward in the image (positive y)."""
    v = np.asarray(v, float)
    return v if v[1] >= 0 else -v


def angle_from_vertical(d, vref):
    """Unsigned angle (deg) between rope direction d and the vertical reference vref.
    Both are oriented downward, so 0 = along the support, larger = flung out."""
    d, vref = down(d), down(vref)
    cross = abs(d[0] * vref[1] - d[1] * vref[0])
    dot = d[0] * vref[0] + d[1] * vref[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def rope_dir(roi_bgr, fill_lo=0.03, fill_hi=0.75):
    """Direction of the dark rope+chair inside a sky box, by PCA. Returns
    (dir_vector, fill_fraction, dark_mask, centroid) or None when empty/blocked."""
    import cv2
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    frac = float((dark > 0).mean())
    if not (fill_lo < frac < fill_hi):
        return None
    ys, xs = np.where(dark > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)                       # 2x2 -> principal axis (O(N), fast)
    evals, evecs = np.linalg.eigh(cov)
    d = down(evecs[:, int(np.argmax(evals))])
    proj = (pts - mean) @ d                            # apex = top end of the rope along d
    apex = mean + d * proj.min()
    return d, frac, dark, mean, apex


def draw_gauge(img, apex, vref, d, ang, off=(0, 0)):
    """Protractor-style angle read-out: vertical ray + rope ray + filled wedge + value."""
    import cv2
    ap = (int(apex[0] + off[0]), int(apex[1] + off[1]))
    Lv, Lr, R = 150, 130, 78
    av = np.degrees(np.arctan2(vref[1], vref[0]))
    ad = np.degrees(np.arctan2(d[1], d[0]))
    a0, a1 = (av, ad) if (ad - av) % 360 <= 180 else (ad, av)
    ov = img.copy()
    cv2.ellipse(ov, ap, (R, R), 0, a0, a1, (60, 200, 255), -1)        # filled wedge
    cv2.addWeighted(ov, 0.40, img, 0.60, 0, img)
    cv2.ellipse(img, ap, (R, R), 0, a0, a1, (60, 200, 255), 2, cv2.LINE_AA)
    ve = (int(ap[0] + vref[0] * Lv), int(ap[1] + vref[1] * Lv))
    re = (int(ap[0] + d[0] * Lr), int(ap[1] + d[1] * Lr))
    cv2.line(img, ap, ve, (255, 255, 255), 2, cv2.LINE_AA)            # vertical (white)
    cv2.line(img, ap, re, (40, 150, 255), 4, cv2.LINE_AA)            # rope (orange)
    cv2.circle(img, ap, 6, (40, 150, 255), -1, cv2.LINE_AA)
    mid = (down(vref) + down(d)); n = np.hypot(*mid)
    mid = mid / n if n else np.array([0, 1.0])
    lp = (int(ap[0] + mid[0] * (R + 18)), int(ap[1] + mid[1] * (R + 18)) + 8)
    cv2.putText(img, f"{ang:.1f}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, f"{ang:.1f}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 200, 255), 2, cv2.LINE_AA)


def deproject(theta_deg, eps_deg):
    return float(np.degrees(np.arctan(np.tan(np.radians(theta_deg)) * np.cos(np.radians(eps_deg)))))


# --------------------------------------------------------------------- calibrate

def banner(img, lines):
    """Big readable instruction text on a dark strip at the top."""
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 38 + 34 * len(lines)), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (16, 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255) if i == 0 else (235, 235, 235), 2, cv2.LINE_AA)


def calibrate(cap, video, eps, disp_w):
    """One big window, 4 labelled clicks: 2 for the vertical support, 2 for the box."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
    ok, f = cap.read()
    if not ok:
        sys.exit("could not read a frame to calibrate on")
    H, W = f.shape[:2]
    ds = disp_w / W                                    # display scale; clicks map back by /ds
    base = cv2.resize(f, (disp_w, int(H * ds)))
    labels = ["1/4  click the TOP of the vertical support pole",
              "2/4  click the BOTTOM of the vertical support pole",
              "3/4  click ONE corner of the checkpoint box (sky where the chair swings)",
              "4/4  click the OPPOSITE corner of the box"]
    clicks = []                                        # stored in ORIGINAL pixels
    win = "calibrate  (u = undo,  Enter = confirm,  Esc = cancel)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, lambda e, x, y, *_: clicks.append((x / ds, y / ds))
                         if e == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4 else None)

    def sp(p):
        return int(p[0] * ds), int(p[1] * ds)
    while True:
        d = base.copy()
        for p in clicks:
            cv2.circle(d, sp(p), 6, (0, 0, 255), -1, cv2.LINE_AA)
        if len(clicks) >= 2:
            cv2.line(d, sp(clicks[0]), sp(clicks[1]), (0, 0, 255), 2, cv2.LINE_AA)
        if len(clicks) >= 4:
            cv2.rectangle(d, sp(clicks[2]), sp(clicks[3]), (0, 230, 0), 2)
        i = len(clicks)
        banner(d, [labels[i] if i < 4 else "Done - press ENTER to save  (u = undo)"])
        cv2.imshow(win, d)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and clicks:
            clicks.pop()
        elif k in (13, 10) and len(clicks) == 4:
            break
        elif k == 27:
            sys.exit("cancelled")
    cv2.destroyAllWindows()
    (cx0, cy0), (cx1, cy1) = clicks[2], clicks[3]
    box = [int(min(cx0, cx1)), int(min(cy0, cy1)), int(max(cx0, cx1)), int(max(cy0, cy1))]
    cal = {"vref": [list(clicks[0]), list(clicks[1])], "roi": box, "eps_deg": eps}
    json.dump(cal, open(sidecar(video), "w"), indent=2)
    print(f"saved calibration -> {os.path.basename(sidecar(video))}  box={box}")
    return cal


# ------------------------------------------------------------------- manual mark

def run_mark(cap, fps, box, vref, eps, f_lo, f_hi, step, disp_w):
    """Click the rope (top, then chair) on each frame you want; angle vs vref."""
    import cv2
    x0, y0, x1, y1 = box
    win = "MARK: click rope TOP then CHAIR (2 clicks) | n=skip  u=undo  q=done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    clicks = []
    cv2.setMouseCallback(win, lambda e, x, y, *_: clicks.append((float(x), float(y)))
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    times, angles = [], []
    fi = f_lo
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
    while fi < f_hi:
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - f_lo) % step:
            fi += 1
            continue
        clicks.clear()
        while True:
            d = frame.copy()
            cv2.line(d, tuple(map(int, vref_pts[0])), tuple(map(int, vref_pts[1])), (0, 0, 255), 2)
            cv2.rectangle(d, (x0, y0), (x1, y1), (0, 200, 0), 1)
            for cpt in clicks:
                cv2.circle(d, (int(cpt[0]), int(cpt[1])), 5, (0, 255, 255), -1, cv2.LINE_AA)
            if len(clicks) >= 2:
                dvec = down(np.array(clicks[1]) - np.array(clicks[0]))
                ang = angle_from_vertical(dvec, vref)
                ang = deproject(ang, eps) if eps else ang
                draw_gauge(d, clicks[0], vref, dvec, ang)
            banner(d, [f"t={fi/fps:5.1f}s   click rope TOP then CHAIR",
                       "Enter = keep   n = skip   u = undo   q = done"])
            sc = disp_w / d.shape[1]
            cv2.imshow(win, cv2.resize(d, None, fx=sc, fy=sc))
            k = cv2.waitKey(20) & 0xFF
            if k == ord("u"):
                clicks.clear()
            elif k == ord("n"):
                break
            elif k == ord("q"):
                cv2.destroyAllWindows()
                return np.array(times), np.array(angles)
            elif k in (13, 10) and len(clicks) >= 2:
                ang = angle_from_vertical(np.array(clicks[1]) - np.array(clicks[0]), vref)
                times.append(fi / fps)
                angles.append(deproject(ang, eps) if eps else ang)
                break
        fi += 1
    cv2.destroyAllWindows()
    return np.array(times), np.array(angles)


vref_pts = None    # set in main so run_mark can draw the actual clicked endpoints


# --------------------------------------------------------------------------- main

def main():
    global vref_pts
    ap = argparse.ArgumentParser(description="Volare fly-out angle (rope vs the ride's vertical).")
    ap.add_argument("video")
    ap.add_argument("--pick", action="store_true", help="mark the vertical support + checkpoint box")
    ap.add_argument("--mark", action="store_true", help="manually click the rope each pass")
    ap.add_argument("--watch", action="store_true", help="auto mode: show measurement live")
    ap.add_argument("--vref", help="vertical support as x1,y1,x2,y2 (instead of --pick)")
    ap.add_argument("--roi", help="checkpoint box x0,y0,x1,y1")
    ap.add_argument("--eps", type=float, default=0.0, help="camera elevation (deg) de-projection")
    ap.add_argument("--start", type=float), ap.add_argument("--end", type=float)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--speed", type=int, default=25, help="--watch playback ms/frame")
    ap.add_argument("--display", type=int, default=1600, help="window width in px (bump up if too small)")
    ap.add_argument("-o", "--out"), ap.add_argument("--outdir")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.pick:
        cal = calibrate(cap, args.video, args.eps, args.display)
    elif args.vref and args.roi:
        v = [float(x) for x in args.vref.split(",")]
        cal = {"vref": [[v[0], v[1]], [v[2], v[3]]],
               "roi": [int(x) for x in args.roi.split(",")], "eps_deg": args.eps}
    elif os.path.exists(sidecar(args.video)):
        cal = json.load(open(sidecar(args.video)))
        if not args.eps:
            args.eps = cal.get("eps_deg", 0.0)
        print(f"(loaded calibration {os.path.basename(sidecar(args.video))})")
    else:
        sys.exit("No calibration. Run once with --pick, or pass --vref and --roi.")

    vref_pts = [np.array(cal["vref"][0]), np.array(cal["vref"][1])]
    vref = down(vref_pts[1] - vref_pts[0])
    box = cal["roi"]; x0, y0, x1, y1 = box
    f_lo = int((args.start or 0) * fps)
    f_hi = int(args.end * fps) if args.end else total
    roll = np.degrees(np.arctan2(vref[0], vref[1]))
    print(f"Video:   {args.video}  {total} frames @ {fps:.1f}fps")
    print(f"Vertical: support tilted {roll:+.1f}deg from image-vertical   box={box}  eps={args.eps:.0f}")

    if args.mark:
        t, a = run_mark(cap, fps, box, vref, args.eps, f_lo, f_hi, max(1, args.step), args.display)
        cap.release()
        if len(t) == 0:
            sys.exit("no measurements taken")
        pt, pa = t, a                               # each click is already one measurement
    else:
        if f_lo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
        times, angles = [], []
        fi = f_lo - 1
        quit_early = False
        while not quit_early:
            ok, frame = cap.read()
            if not ok or fi >= f_hi:
                break
            fi += 1
            if fi % args.step:
                continue
            res = rope_dir(frame[y0:y1, x0:x1])
            ang = None
            if res is not None:
                d, frac, dark, c, apex = res
                ang = angle_from_vertical(d, vref)
                ang = deproject(ang, args.eps) if args.eps else ang
                times.append(fi / fps); angles.append(ang)
            if args.watch:
                disp = frame.copy()
                cv2.line(disp, tuple(map(int, vref_pts[0])), tuple(map(int, vref_pts[1])),
                         (0, 0, 180), 1, cv2.LINE_AA)               # marked vertical (faint)
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 200, 0), 1)
                if ang is not None:
                    draw_gauge(disp, (x0 + apex[0], y0 + apex[1]), vref, d, ang)
                banner(disp, [f"t={fi/fps:5.1f}s    angle {ang:.1f} deg" if ang is not None
                              else f"t={fi/fps:5.1f}s    (no chair in box)", "q = quit"])
                sc = args.display / disp.shape[1]
                cv2.imshow("checkpoint", cv2.resize(disp, None, fx=sc, fy=sc))
                if (cv2.waitKey(max(1, args.speed)) & 0xFF) == ord("q"):
                    quit_early = True
        cap.release()
        if args.watch:
            cv2.destroyAllWindows()
        if not angles:
            sys.exit("No rope detected in the box -- reposition it over sky where the chair swings.")
        t = np.array(times); a = np.array(angles)
        # per-pass peaks: a run of detections = one crossing; its max = side-on theta
        pt, pa, i = [], [], 0
        while i < len(t):
            j = i
            while j + 1 < len(t) and t[j + 1] - t[j] < 0.4:
                j += 1
            k = i + int(np.argmax(a[i:j + 1]))
            pt.append(t[k]); pa.append(a[k]); i = j + 1
        pt, pa = np.array(pt), np.array(pa)

    pa_ema = accel.ema(pa, 2 / (args.smooth + 1)) if len(pa) > 1 else pa
    hi = pa_ema >= np.nanpercentile(pa_ema, 60) if len(pa) > 2 else np.ones(len(pa), bool)
    ss, sd = float(np.mean(pa[hi])), float(np.std(pa[hi]))
    print(f"Measured: {len(pa)} angles   range {pa.min():.1f}..{pa.max():.1f} deg"
          + ("  (de-projected)" if args.eps else "  (apparent; --eps to correct)"))
    print(f"Steady:  fly-out ~ {ss:.1f} +/- {sd:.1f} deg  ->  g*tan = {G*np.tan(np.radians(ss)):.1f} m/s^2")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    a1.plot(pt, pa, "o-", color=accel.C_RAW, ms=4, lw=1.0, label="per pass")
    if len(pa) > 1:
        a1.plot(pt, pa_ema, color=accel.C_EMA, lw=2.0, label=f"EMA (N={args.smooth})")
    a1.axhline(ss, color=accel.C_FIT, lw=1.2, ls="--", label=f"steady ~{ss:.1f}°")
    a1.set_ylabel("fly-out angle θ  (deg)")
    a1.legend(loc="lower right", fontsize=9)
    accel.style_axis(a1)
    a2.plot(pt, G * np.tan(np.radians(pa)), "o-", color=accel.C_FIT, ms=4, lw=1.0)
    a2.set_ylabel(r"$g\,\tan\theta$  (m/s$^2$)")
    a2.set_xlabel("time (s)")
    accel.style_axis(a2)
    fig.suptitle("Volare fly-out angle (rope vs the ride's vertical)"
                 + (f"  eps={args.eps:.0f}°" if args.eps else "  (apparent)"),
                 fontsize=13, fontweight="bold")
    png, csv = accel.out_paths_for(args.out, args.outdir, HERE, args.video, "_angle")
    fig.savefig(png, dpi=150)
    pd.DataFrame({"pass_time": pt, "theta_deg": pa, "theta_deg_ema": pa_ema,
                  "g_tan_theta": G * np.tan(np.radians(pa))}).to_csv(csv, index=False)
    print(f"Graph:   {png}")
    print(f"Data:    {csv}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
