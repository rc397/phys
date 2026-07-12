# fly-out angle of the volare from video, no clicking. background-subtract
# the moving chairs, take the outermost blob each side (side-on chair ->
# steepest apparent chain), find the chain above the seat, correct for the
# camera looking up.
#   python volare_angle.py "trial 1.MOV" --annot
#   python volare_angle.py --all

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

import accel

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # the repo root, one up from scripts/
PROC_W = 1280         # all mask work happens at this width (thin chains need it)


def camera_tag(video):
    # both cameras use the same clip names, so outputs carry the camera's name
    parent = os.path.basename(os.path.dirname(os.path.abspath(video))).lower()
    for name in ("alex", "ryan"):
        if name in parent:
            return name
    return "".join(c for c in parent if c.isalnum())[:10] or "cam"


def sidecar(video):
    return os.path.splitext(video)[0] + ".volare.json"


def masks_path(video):
    return os.path.splitext(video)[0] + ".volare_masks.npz"


def down(v):
    v = np.asarray(v, float)
    return v if v[1] >= 0 else -v


def angle_from_vertical(d, vref):
    # unsigned angle (deg) between rope direction d and the vertical reference
    d, vref = down(d), down(vref)
    cross = abs(d[0] * vref[1] - d[1] * vref[0])
    dot = d[0] * vref[0] + d[1] * vref[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def deproject(theta_deg, eps_deg):
    # steepest apparent chain -> true angle:
    # tan(theta) = T cos(eps)/sqrt(1 + T^2 sin^2(eps)), T = tan(theta'_max)
    T = np.tan(np.radians(theta_deg))
    e = np.radians(eps_deg)
    return float(np.degrees(np.arctan(T * np.cos(e) / np.sqrt(1 + T * T * np.sin(e) ** 2))))


def banner(img, lines):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 + 26 * len(lines)), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (12, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255) if i == 0 else (230, 230, 230), 2, cv2.LINE_AA)


def draw_gauge(img, apex, vref, d, ang, dim=False):
    # gauge overlay - dim = held reading between verified frames
    import cv2
    wedge = (140, 160, 170) if dim else (60, 200, 255)
    rope = (120, 130, 140) if dim else (40, 150, 255)
    ap = (int(apex[0]), int(apex[1]))
    Lv, Lr, R = 110, 95, 55
    av = np.degrees(np.arctan2(vref[1], vref[0]))
    ad = np.degrees(np.arctan2(d[1], d[0]))
    a0, a1 = (av, ad) if (ad - av) % 360 <= 180 else (ad, av)
    ov = img.copy()
    cv2.ellipse(ov, ap, (R, R), 0, a0, a1, wedge, -1)
    cv2.addWeighted(ov, 0.25 if dim else 0.40, img, 0.75 if dim else 0.60, 0, img)
    cv2.ellipse(img, ap, (R, R), 0, a0, a1, wedge, 2, cv2.LINE_AA)
    ve = (int(ap[0] + vref[0] * Lv), int(ap[1] + vref[1] * Lv))
    re = (int(ap[0] + d[0] * Lr), int(ap[1] + d[1] * Lr))
    cv2.line(img, ap, ve, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, ap, re, rope, 3, cv2.LINE_AA)
    cv2.circle(img, ap, 5, rope, -1, cv2.LINE_AA)
    mid = down(vref) + down(d)
    n = np.hypot(*mid)
    mid = mid / n if n else np.array([0, 1.0])
    lp = (int(ap[0] + mid[0] * (R + 14)), int(ap[1] + mid[1] * (R + 14)) + 6)
    cv2.putText(img, f"{ang:.1f}", lp, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, f"{ang:.1f}", lp, cv2.FONT_HERSHEY_SIMPLEX, 0.8, wedge, 2, cv2.LINE_AA)


def detect_vertical(bg_gray, exclude, min_len):
    # camera roll from long near-vertical background edges (building/pylon lines)
    import cv2
    edges = cv2.Canny(bg_gray, 50, 140)
    edges[exclude > 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                            minLineLength=int(min_len), maxLineGap=6)
    if lines is None:
        return 0.0, 0
    angs, wts = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if dy == 0:
            continue
        a = np.degrees(np.arctan2(dx, dy))          # signed, from image-vertical
        if a > 90:
            a -= 180
        if a < -90:
            a += 180
        if abs(a) <= 10:
            angs.append(a)
            wts.append(np.hypot(dx, dy))
    if not angs:
        return 0.0, 0
    order = np.argsort(angs)
    angs, wts = np.array(angs)[order], np.array(wts)[order]
    cum = np.cumsum(wts)
    roll = float(angs[np.searchsorted(cum, cum[-1] / 2)])   # weighted median
    return roll, len(angs)


def auto_calibrate(video, cap, fps, ntot, args):
    # find the ride region + true vertical once, cache next to the video
    import cv2
    if not args.recal and os.path.exists(sidecar(video)) and os.path.exists(masks_path(video)):
        with open(sidecar(video)) as fh:
            cal = json.load(fh)
        if cal.get("version") == 4:
            m = np.load(masks_path(video))
            cal["roi"] = m["roi"]
            print(f"(calibration loaded from {os.path.basename(sidecar(video))})")
            return cal

    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = PROC_W / W0
    Hs = int(round(H0 * sc))
    gap = max(2, int(round(0.2 * fps)))             # pair spacing: 0.2 s
    K = 90
    anchors = np.linspace(0.04 * ntot, 0.95 * ntot - gap, K).astype(int)

    fast_masks, grays = [], []
    for a in anchors:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a))
        ok, f0 = cap.read()
        for _ in range(gap - 1):
            cap.grab()
        ok2, f1 = cap.read()
        if not (ok and ok2):
            continue
        f0 = cv2.resize(f0, (PROC_W, Hs))
        f1 = cv2.resize(f1, (PROC_W, Hs))
        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        # blur so 1-px camera wobble doesn't count as motion
        g0b = cv2.GaussianBlur(g0, (5, 5), 0)
        g1b = cv2.GaussianBlur(g1, (5, 5), 0)
        fast_masks.append(cv2.absdiff(g0b, g1b) > 18)
        grays.append(g0)
    fast_masks = np.stack(fast_masks)

    # keep only samples where the ride is actually running
    activity = fast_masks.reshape(len(fast_masks), -1).sum(1)
    active = activity > 0.25 * activity.max()
    if active.sum() < 8:
        active = activity >= np.partition(activity, -8)[-8]
    mot = fast_masks[active].mean(0)

    # ride region: where fast motion happens at all often
    env = (mot > 0.10).astype(np.uint8) * 255
    env = cv2.morphologyEx(env, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    env = cv2.morphologyEx(env, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(env, 8)
    if nlab < 2:
        sys.exit("calibration failed: no moving ride found")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    roi = (lab == big).astype(np.uint8) * 255
    roi = cv2.dilate(roi, np.ones((int(0.02 * PROC_W),) * 2, np.uint8))
    rys, rxs = np.where(roi > 0)

    # true vertical from static background edges, away from the ride
    med_bg = np.median(np.stack(grays), 0).astype(np.uint8)
    roll, nlines = detect_vertical(med_bg, cv2.dilate(roi, np.ones((25, 25), np.uint8)),
                                   min_len=0.12 * Hs)
    vref = [float(np.sin(np.radians(roll))), float(np.cos(np.radians(roll)))]

    cal = {"version": 4, "scale": sc,
           "roll_deg": round(roll, 2), "n_vert_lines": int(nlines),
           "axis_x": float(rxs.mean()),
           "roi": roi}
    np.savez_compressed(masks_path(video), roi=roi)
    slim = {k: v for k, v in cal.items() if k != "roi"}
    with open(sidecar(video), "w") as fh:
        json.dump(slim, fh, indent=2)

    if args.debug:
        dbg = cv2.cvtColor(med_bg, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, cnts, -1, (0, 255, 0), 2)
        cv2.line(dbg, (int(cal["axis_x"]), 0), (int(cal["axis_x"]), Hs), (0, 255, 255), 1)
        x0 = int(PROC_W / 2)
        cv2.line(dbg, (x0, 0), (int(x0 + vref[0] * Hs), int(vref[1] * Hs)), (255, 255, 0), 1)
        d = args.outdir or os.path.join(ROOT, "output")
        os.makedirs(d, exist_ok=True)
        base = os.path.splitext(os.path.basename(video))[0]
        cv2.imwrite(os.path.join(d, base + "_debug_cal.png"), dbg)
        cv2.imwrite(os.path.join(d, base + "_debug_mot.png"),
                    cv2.applyColorMap((np.clip(mot, 0, 1) * 255).astype(np.uint8),
                                      cv2.COLORMAP_JET))
    return cal


def measure(cap, fps, ntot, cal, args, video, annot_path=None):
    # main pass: outermost moving blob per side, chain above its seat
    import cv2
    roi = cal["roi"]
    Hs = roi.shape[0]
    axis_x = cal["axis_x"]
    vref = np.array([np.sin(np.radians(cal["roll_deg"])), np.cos(np.radians(cal["roll_deg"]))])
    rys, rxs = np.where(roi > 0)
    roi_w = rxs.max() - rxs.min() if len(rxs) else PROC_W
    roi_h = rys.max() - rys.min() if len(rys) else Hs
    a_min = max(35, int(2.2e-4 * roi_w * roi_h))
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_join = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))

    step = args.step or max(1, int(round(fps / 30)))
    f_lo = int((args.start or 0) * fps)
    f_hi = int(args.end * fps) if args.end else ntot
    warm = int(3 * fps)
    mog = cv2.createBackgroundSubtractorMOG2(history=int(8 * fps / step),
                                             varThreshold=16, detectShadows=True)
    start = max(0, f_lo - warm)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)           # calibration moved the read head
    meas_from = start + warm                          # let MOG2 settle first

    meas = []                                        # (t, side, theta_apparent, agree)
    sweeps = {}                                      # per-window unions of the chair mask
    activity = []                                    # (t, moving px) for rest detection
    rej = {"few_px": 0, "still": 0, "cam": 0, "few_side": 0, "no_chain": 0,
           "refit": 0, "range": 0, "kept": 0, "held": 0}
    bg_area = max(1, int(np.count_nonzero(roi == 0)))
    # re-fit a found chain on later frames - a lock lives ~2 s
    lock = {"left": None, "right": None}
    max_hold = max(3, int(round(2.0 * fps / step)))
    rot_streak = 0                                   # rotation must persist to count
    streak_min = max(3, int(round(0.7 * fps / step)))
    # 0.2 s frame difference - riding survives it, cloud lighting doesn't
    from collections import deque
    gdeq = deque(maxlen=max(1, int(round(0.2 * fps / step))))

    # the check video covers every processed frame, so it plays like the real ride
    annot_vw = None
    annot_count = 0

    def annot_frame(img, got, t, note=""):
        nonlocal annot_vw, annot_count
        if not annot_path:
            return
        annot_count += 1
        if annot_count % 2:                          # every 2nd frame keeps the size sane
            return
        d = img.copy()
        for _side, top, dv, ang, stale in got:
            dvn = dv / (np.hypot(*dv) + 1e-9)
            draw_gauge(d, top, vref, dvn, ang, dim=stale)
        msg = f"t={t:6.1f}s"
        if got:
            parts = [f"{g[3]:.1f} deg" + (" (holding)" if g[4] else "") for g in got]
            msg += "   " + "  ".join(parts)
        elif note:
            msg += f"   {note}"
        banner(d, [msg])
        d = cv2.resize(d, (960, int(round(Hs * 960 / PROC_W))))
        if annot_vw is None:
            fps_out = max(5.0, fps / step / 2)
            annot_vw = cv2.VideoWriter(annot_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps_out, (d.shape[1], d.shape[0]))
        annot_vw.write(d)

    fi = start - 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi >= f_hi:
            break
        if fi % step:
            continue
        small = cv2.resize(frame, (PROC_W, Hs))
        fg = mog.apply(small)
        gb = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        fast_mass = 0
        cam_moving = False
        if len(gdeq) == gdeq.maxlen:
            diff = cv2.absdiff(gb, gdeq[0]) > 18
            fast_mass = int(np.count_nonzero(diff & (roi > 0)))
            # background moving = camera moving, drop those frames
            cam_moving = np.count_nonzero(diff & (roi == 0)) > 0.04 * bg_area
        gdeq.append(gb)
        if fi < f_lo or fi < meas_from:
            continue
        t = fi / fps
        if cam_moving:
            rej["cam"] += 1
            rot_streak = 0
            annot_frame(small, [], t, "camera moving")
            continue
        raw = ((fg >= 200) & (roi > 0)).astype(np.uint8) * 255
        chair = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel_open)
        chair = cv2.morphologyEx(chair, cv2.MORPH_CLOSE, kernel_join)
        n_ch = cv2.countNonZero(chair)
        if n_ch < a_min:
            rej["few_px"] += 1
            activity.append((t, fast_mass, 0))
            annot_frame(small, [], t, "at rest")
            continue
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(chair, 8)
        # need ride-wide motion persisting ~0.7 s, not a gust on the tent
        big = [i for i in range(1, nlab) if stats[i, cv2.CC_STAT_AREA] >= a_min]
        if big:
            xspread = (max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                           for i in big)
                       - min(stats[i, cv2.CC_STAT_LEFT] for i in big)) / max(roi_w, 1)
        else:
            xspread = 0.0
        rot_streak = rot_streak + 1 if (len(big) >= 4 and xspread >= 0.45
                                        and fast_mass >= 4 * a_min) else 0
        rotating = rot_streak >= streak_min
        activity.append((t, fast_mass, 1 if rotating else 0))
        if not rotating:
            rej["still"] += 1
            annot_frame(small, [], t, "at rest")
            continue
        t_w = int(t // args.win)
        if t_w not in sweeps:
            sweeps[t_w] = np.zeros((Hs, PROC_W), np.uint8)
        sweeps[t_w] |= chair
        # raw pixels for the chain search
        rys_, rxs_ = np.where(raw > 0)
        p_r = np.column_stack([rxs_, rys_]).astype(np.float32)

        def scan_blob(li, sgn):
            # scan for this blob's chain in the raw pixels above its seat
            bys, bxs = np.where(lab == li)
            by = bys.max()
            bot = np.array([float(np.mean(bxs[bys >= by - 3])), float(by)], np.float32)
            # median seat height so a blob merged with the canopy can't inflate it
            hs_all = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, nlab)
                      if stats[i, cv2.CC_STAT_AREA] >= a_min]
            h_seat = float(np.clip(np.median(hs_all), 8, 0.12 * roi_h))
            # rays start at the seat bottom
            L_skip = 0.9 * h_seat
            L_ray = L_skip + 3.5 * h_seat
            nb = ((np.abs(p_r[:, 0] - bot[0]) <= L_ray + 10)
                  & (p_r[:, 1] <= bot[1] + 2) & (p_r[:, 1] >= bot[1] - L_ray - 10))
            q = p_r[nb] - bot
            if len(q) < 12:
                return None
            n_bins = max(4, int((L_ray - L_skip) // 4))
            best = None
            for adeg in range(3, 66, 2):
                u = np.array([sgn * np.sin(np.radians(adeg)), -np.cos(np.radians(adeg))],
                             np.float32)
                proj = q @ u
                perp = np.abs(q @ np.array([-u[1], u[0]], np.float32))
                in_range = (proj > L_skip) & (proj < L_ray)
                on = in_range & (perp <= 3.5)
                if on.sum() < 10:
                    continue
                # a real chain fills the ray continuously, a crossing rope doesn't
                filled = len(np.unique(((proj[on] - L_skip) // 4).astype(int)))
                cover = filled / n_bins
                if cover < 0.55:
                    continue
                # and it's thin - a ray through the canopy fails this
                wide = in_range & (perp <= 10.0)
                if wide.sum() > 1.8 * on.sum():
                    continue
                if best is None or cover > best[0]:
                    best = (cover, adeg, on)
            if best is None:
                return None
            _, adeg, on = best
            pts = q[on] + bot
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            nline = np.array([-vy, vx])
            resid = float(np.mean(np.abs((pts - [x0, y0]) @ nline)))
            ang = angle_from_vertical(np.array([vx, vy]), vref)
            # scan and refit must agree, and the support must look like a line
            if resid > 2.5 or abs(ang - adeg) > 5.0 or not (3.0 <= ang <= 65.0):
                return None
            return ang, pts, np.array([vx, vy], np.float32), np.array([x0, y0], np.float32)

        def try_fresh(side):
            # try the few outermost chairs, keep the steepest that passes
            side_ok = [li for li in range(1, nlab)
                       if stats[li, cv2.CC_STAT_AREA] >= a_min
                       and (cent[li][0] < axis_x if side == "left" else cent[li][0] > axis_x)
                       and stats[li, cv2.CC_STAT_LEFT] > 2
                       and stats[li, cv2.CC_STAT_LEFT] + stats[li, cv2.CC_STAT_WIDTH] < PROC_W - 2]
            if not side_ok:
                rej["few_side"] += 1
                return None
            if side == "left":
                cands = sorted(side_ok, key=lambda i: stats[i, cv2.CC_STAT_LEFT])[:3]
                sgn = 1.0                             # chains rise toward the axis
            else:
                cands = sorted(side_ok, key=lambda i: -(stats[i, cv2.CC_STAT_LEFT]
                                                        + stats[i, cv2.CC_STAT_WIDTH]))[:3]
                sgn = -1.0
            results = [r for r in (scan_blob(li, sgn) for li in cands) if r]
            if not results:
                rej["no_chain"] += 1
                return None
            return max(results, key=lambda r: r[0])

        def try_relock(side, st):
            # re-fit the raw pixels near the last known line
            dvec, a0 = st["dvec"], st["pt"]
            ylo, yhi = st["ylo"] - 12, st["yhi"] + 12
            pts = None
            for tol in (6.0, 3.5):
                nvec = np.array([-dvec[1], dvec[0]], np.float32)
                dist = np.abs((p_r - a0) @ nvec)
                near = (dist <= tol) & (p_r[:, 1] >= ylo) & (p_r[:, 1] <= yhi)
                if near.sum() < 15:
                    return None
                pts = p_r[near]
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                dvec = np.array([vx, vy], np.float32)
                a0 = np.array([x0, y0], np.float32)
            nline = np.array([-dvec[1], dvec[0]])
            resid = float(np.mean(np.abs((pts - a0) @ nline)))
            ang = angle_from_vertical(dvec, vref)
            yspan = float(pts[:, 1].max() - pts[:, 1].min())
            if (resid > 2.5 or abs(ang - st["ang"]) > 8.0
                    or yspan < max(15.0, 0.4 * (st["yhi"] - st["ylo"]))
                    or not (3.0 <= ang <= 65.0)):
                return None
            # same gates as a fresh find
            in_y = (p_r[:, 1] >= ylo) & (p_r[:, 1] <= yhi)
            dist = np.abs((p_r - a0) @ nline)
            on = in_y & (dist <= 3.5)
            wide = in_y & (dist <= 10.0)
            if wide.sum() > 1.8 * on.sum():
                return None
            proj = (pts - a0) @ dvec
            filled = len(np.unique((proj // 4).astype(int)))
            if filled < 0.5 * max(4, int(yspan // 4)):
                return None
            return ang, pts, dvec, a0

        got = []
        for side in ("left", "right"):
            st = lock[side]
            if st is not None:
                st["age"] += 1
                if st["age"] > max_hold:
                    lock[side] = st = None
            m = try_fresh(side)
            held = False
            if m is None and st is not None:
                m = try_relock(side, st)
                held = m is not None
            if m is None:
                if st is not None:
                    # could not re-verify this frame - show the last reading dimmed
                    got.append((side, st["top"], down(st["dvec"]), st["ang"], True))
                continue
            ang, pts, dvec, a0 = m
            rej["held" if held else "kept"] += 1
            gauge_at = pts[int(np.argmin(pts[:, 1]))]
            lock[side] = {"pt": a0, "dvec": dvec, "ang": ang, "top": gauge_at,
                          "ylo": float(pts[:, 1].min()), "yhi": float(pts[:, 1].max()),
                          "age": st["age"] if held else 0}
            meas.append((t, side, ang, 0.7 if held else 1.0))
            got.append((side, gauge_at, down(dvec), ang, False))
        annot_frame(small, got, t, "" if got else "no lock")
    if annot_vw is not None:
        annot_vw.release()
    if args.debug:
        print("Gates:   " + "  ".join(f"{k}={v}" for k, v in rej.items()))
    sweep_area = {w: int(cv2.countNonZero(mk)) for w, mk in sweeps.items()}
    return meas, sweeps, sweep_area, activity, a_min


def aggregate(meas, sweep_area, activity, a_min, args):
    # window into theta(t) - idle windows become rest rows (theta 0)
    m = pd.DataFrame(meas, columns=["t", "side", "theta", "agree"])
    act = pd.DataFrame(activity, columns=["t", "px", "rotating"])
    win = args.win
    m["w"] = (m["t"] // win).astype(int)
    act["w"] = (act["t"] // win).astype(int)
    groups = dict(tuple(m.groupby("w")))
    # flutter/crowds move far fewer pixels than riding - gate per window
    # against this video's own levels
    med_px = act.groupby("w")["px"].median()
    px_floor = 0.10 * np.percentile(med_px, 95)
    areas = pd.Series(sweep_area)
    area_floor = 0.12 * np.percentile(areas, 95) if len(areas) else 0.0
    rows = []
    for w, agrp in act.groupby("w"):
        if float(med_px[w]) < px_floor or sweep_area.get(w, 0) < area_floor:
            rows.append({"time": (w + 0.5) * win, "theta_apparent": 0.0,
                         "lo": 0.0, "hi": 0.0, "n": 0, "sides": 0, "rest": 1})
        elif w in groups:
            grp = groups[w]
            # stats use fresh finds only - re-locks are just for the check video
            fresh = grp[grp["agree"] >= 0.99]
            if len(fresh) >= 5:
                grp = fresh
            # keep the upper edge - occlusion only drags readings down
            per_side = grp.groupby("side")["theta"].quantile(0.85)
            rows.append({"time": (w + 0.5) * win,
                         "theta_apparent": float(per_side.max()),
                         "lo": float(np.percentile(grp["theta"], 10)),
                         "hi": float(np.percentile(grp["theta"], 90)),
                         "n": len(grp),
                         "sides": len(per_side),
                         "rest": 0})
        elif float(agrp["rotating"].median()) < 0.5:
            # nothing moving, or only in-place flutter: the ride is parked
            rows.append({"time": (w + 0.5) * win, "theta_apparent": 0.0,
                         "lo": 0.0, "hi": 0.0, "n": 0, "sides": 0, "rest": 1})
        # rotating but nothing measurable passed the gates: leave an honest gap
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def fit_lower_arc(cols, bot, Hs):
    # lower-boundary arc fit, dropping outliers hanging below
    from scipy.optimize import least_squares

    def arc(p, x):
        cx_, y0, a, b = p
        u = np.clip((x - cx_) / max(a, 1.0), -0.999, 0.999)
        return y0 + b * np.sqrt(1 - u * u)

    span = cols[-1] - cols[0]
    p0 = [cols.mean(), np.percentile(bot, 15), span / 2, max(4.0, np.ptp(bot) * 0.7)]
    keep = np.ones(len(cols), bool)
    fit = None
    for _ in range(3):
        fit = least_squares(lambda p: arc(p, cols[keep]) - bot[keep], p0,
                            loss="soft_l1", f_scale=3.0,
                            bounds=([cols[0], 0, span * 0.3, 1.0],
                                    [cols[-1], Hs, span, Hs * 0.5]))
        p0 = fit.x
        resid = bot - arc(fit.x, cols)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) + 1e-6
        keep = resid < 3.0 * mad
        if keep.sum() < 30:
            break
    return fit.x


def elevation_from_sweep(sweeps, df, args):
    # elevation from the swept ring: its lower boundary is an ellipse arc
    # with axis ratio sin(elevation)
    if args.eps is not None:
        return args.eps, "given"
    top = np.percentile(df["theta_apparent"], 85)
    keep_w = set((df.loc[df["theta_apparent"] >= top - 1.5, "time"] // args.win).astype(int))
    sweep = None
    for w, m in sweeps.items():
        if w in keep_w:
            sweep = m.copy() if sweep is None else (sweep | m)
    if sweep is None:
        return 0.0, "none"
    ys, xs = np.where(sweep > 0)
    if len(xs) < 500:
        return 0.0, "none"
    cols_all = np.arange(xs.min(), xs.max() + 1)
    bot = np.full(len(cols_all), np.nan)
    for i, x in enumerate(cols_all):
        cy = ys[xs == x]
        if len(cy):
            bot[i] = cy.max()
    ok = ~np.isnan(bot)
    cols, bot = cols_all[ok].astype(float), bot[ok].astype(float)
    if len(cols) < 60:
        return 0.0, "none"
    cx, y0, a, b = fit_lower_arc(cols, bot, sweep.shape[0])
    arc_y = y0 + b * np.sqrt(1 - np.clip((cols - cx) / max(a, 1), -0.999, 0.999) ** 2)
    r_arc = float(np.sqrt(np.mean((bot - arc_y) ** 2)))
    line = np.polyval(np.polyfit(cols, bot, 1), cols)
    r_line = float(np.sqrt(np.mean((bot - line) ** 2)))
    # a straight line fitting as well = bottom hidden, elevation unreliable
    conf = "good" if r_line > 1.3 * r_arc else "low (ring bottom may be hidden)"
    return float(np.degrees(np.arcsin(np.clip(b / max(a, 1), 0, 0.75)))), conf


def finish(df, eps, args):
    # deproject the apparent angles and pull out the steady state
    for c in ("theta_apparent", "lo", "hi"):
        out = "theta" if c == "theta_apparent" else c
        df[out] = [deproject(v, eps) for v in df[c]]
    df["theta_ema"] = accel.smooth(df["theta"].to_numpy(), 2 / (args.smooth + 1))
    top = np.percentile(df["theta"], 90)             # robust: outliers cannot set it
    plateau = df["theta"] >= top - 2.5
    steady = float(df.loc[plateau, "theta"].mean())
    steady_sd = float(df.loc[plateau, "theta"].std())
    return df, steady, steady_sd, plateau


def steady_spans(times, plateau, gap=12.0):
    # merge plateau windows into contiguous shading spans
    tpl = np.asarray(times)[np.asarray(plateau)]
    if not len(tpl):
        return []
    spans = [[tpl[0], tpl[0]]]
    for t in tpl[1:]:
        if t - spans[-1][1] <= gap:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return spans


def plot_angle(df, steady, plateau, eps, video, png, show=False):
    # theta(t) with the wave band - shade where the steady mean comes from
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    a1.fill_between(df["time"], df["lo"], df["hi"], color="#f2c7c0", alpha=0.6,
                    label="per-window spread (the wave)")
    a1.plot(df["time"], df["theta"], ".", color=accel.C_RAW, ms=4, label="per window")
    a1.plot(df["time"], df["theta_ema"], color=accel.C_EMA, lw=2, label="smoothed")
    spans = steady_spans(df["time"].to_numpy(), plateau)
    for k, (s, e) in enumerate(spans):
        a1.axvspan(s, e, color="#8fb3d9", alpha=0.18,
                   label="steady state (angle averaged here)" if k == 0 else None)
    a1.axhline(steady, color=accel.C_FIT, lw=1.2, ls="--",
               label=f"steady-state mean {steady:.1f} deg")
    if spans:
        # name the run phases once, above the plateau
        smid = 0.5 * (spans[0][0] + spans[-1][1])
        a1.annotate("spin-up", (spans[0][0], 4), (spans[0][0] - 2, 4), fontsize=8,
                    ha="right", va="bottom", color="#555555")
        a1.annotate("slow-down", (spans[-1][1], 4), (spans[-1][1] + 2, 4), fontsize=8,
                    ha="left", va="bottom", color="#555555")
    a1.set_ylabel("fly-out angle theta (deg)")
    a1.legend(loc="lower center", fontsize=8, ncols=5)
    accel.style_axis(a1)
    a2.plot(df["time"], G * np.tan(np.radians(df["theta_ema"])), color=accel.C_FIT, lw=1.6)
    a2.set_ylabel(r"$g\,\tan\theta$  (m/s$^2$)")
    a2.set_xlabel("time (s)")
    accel.style_axis(a2)
    fig.suptitle(f"Fly-out angle, automatic: {os.path.basename(video)}   "
                 f"(elevation-corrected {eps:.0f} deg)", fontsize=12, fontweight="bold")
    fig.savefig(png, dpi=150)
    plt.close(fig)


def analyse(video, args):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ntot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=== {os.path.basename(video)}  ({ntot} frames @ {fps:.1f} fps)")
    cal = auto_calibrate(video, cap, fps, ntot, args)
    print(f"Calib:   camera roll {cal['roll_deg']:+.2f} deg "
          f"({cal['n_vert_lines']} vertical edges)")
    png, csv = accel.out_paths_for(args.out, args.outdir
                                   or os.path.join(ROOT, "output", "angles"), ROOT, video,
                                   f"_angle_{camera_tag(video)}")
    annot_path = os.path.splitext(png)[0] + "_annot.mp4" if args.annot else None
    meas, sweeps, sweep_area, activity, a_min = measure(cap, fps, ntot, cal, args, video, annot_path)
    cap.release()
    if len(meas) < 10:
        print("!! not enough rope measurements; skipping")
        return None
    df = aggregate(meas, sweep_area, activity, a_min, args)
    eps, eps_conf = elevation_from_sweep(sweeps, df, args)
    df, steady, steady_sd, plateau = finish(df, eps, args)
    print(f"Ring:    camera elevation {eps:.1f} deg, confidence {eps_conf}")
    print(f"Measured: {len(meas)} ropes over {len(df)} windows")
    print(f"Steady:  theta = {steady:.1f} +/- {steady_sd:.1f} deg   "
          f"g tan = {G*np.tan(np.radians(steady)):.1f} m/s^2")

    plot_angle(df, steady, plateau, eps, video, png, args.show)
    df.to_csv(csv, index=False)
    print(f"Graph:   {png}")
    print(f"Data:    {csv}")
    if annot_path and os.path.exists(annot_path):
        print(f"Check:   {annot_path}  (full-length, plays like the ride)")
    return {"video": video, "df": df, "steady": steady, "steady_sd": steady_sd,
            "eps": eps, "roll": cal["roll_deg"]}


def accel_compare(res, accel_csv, args):
    # overlay the video theta(t) with the phone's arctan(aT/g)
    import matplotlib.pyplot as plt
    df = res["df"]
    a = accel.load(accel_csv)
    tcol = accel.find_time(a)
    at_col = next((c for c in a.columns if c.strip().lower().startswith("at")), None)
    if at_col is None:
        return
    ta = pd.to_numeric(a[tcol], errors="coerce").to_numpy()
    at = pd.to_numeric(a[at_col], errors="coerce").to_numpy()
    at_s = accel.smooth(at, 2 / 301)
    th_a = np.degrees(np.arctan(at_s / G))
    # clocks from vidsync - unknown footage falls back to correlation
    import vidsync
    grid = 0.5
    tv = df["time"].to_numpy()
    thv = np.nan_to_num(df["theta_ema"].to_numpy())
    lag = vidsync.video_lag(res["video"])
    if lag is None:
        lag = vidsync.xcorr_lag(tv, thv, ta, np.nan_to_num(th_a))
        print("Sync:    from curve correlation (video not in the sync table)")
    gv = np.arange(0, tv.max() + grid, grid)
    sv = np.interp(gv, tv, thv, left=0, right=0)
    ga = np.arange(ta.min(), ta.max(), grid)
    sa = np.interp(ga, ta, np.nan_to_num(th_a))
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    ax.plot(gv, sv, color=accel.C_EMA, lw=2, label="video theta(t)")
    ax.plot(ga - lag, sa, color=accel.C_FIT, lw=1.5, label="phone arctan(aT/g)")
    ax.set_xlim(0, gv.max())
    ax.set_xlabel("time (s, video clock)")
    ax.set_ylabel("theta (deg)")
    ax.legend(fontsize=9)
    accel.style_axis(ax)
    base = os.path.splitext(os.path.basename(res["video"]))[0]
    tag = camera_tag(res["video"])
    fig.suptitle(f"Video ({tag}) vs accelerometer: {base}", fontsize=12, fontweight="bold")
    out = os.path.join(args.outdir or os.path.join(ROOT, "output", "accel"),
                       f"{base}_vs_accel_{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    hi = sa >= 0.8 * np.nanmax(sa)     # plateau only, however long the idle wait was
    print(f"Accel:   steady theta from phone ~ {np.nanmean(sa[hi]):.1f} deg   "
          f"video {res['steady']:.1f} deg   ({os.path.basename(accel_csv)})")
    print(f"Overlay: {out}")


ACCEL_MAP = {"trial 1": "data/1st Trial.csv", "trial 2": "data/2nd Trial.csv",
             "trial 3": "data/3rd Trial.csv", "trial 4": "data/4th Trial.csv"}


def find_accel_csv(video):
    name = os.path.basename(video).lower()
    for key, csvf in ACCEL_MAP.items():
        if key in name:
            p = os.path.join(ROOT, csvf)
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Automatic fly-out angle from ride video.")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--all", action="store_true", help="run every video in Videos/")
    ap.add_argument("--annot", action="store_true", help="write a verification video")
    ap.add_argument("--debug", action="store_true", help="dump calibration overlay image")
    ap.add_argument("--recal", action="store_true", help="ignore cached calibration")
    ap.add_argument("--eps", type=float, help="override camera elevation (deg)")
    ap.add_argument("--win", type=float, default=3.0, help="aggregation window (s)")
    ap.add_argument("--step", type=int, help="process every K-th frame")
    ap.add_argument("--start", type=float), ap.add_argument("--end", type=float)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("-o", "--out"), ap.add_argument("--outdir")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.all:
        vids = sorted(glob.glob(os.path.join(ROOT, "Videos", "*", "*.mp4"))
                      + glob.glob(os.path.join(ROOT, "Videos", "*", "*.MOV")))
        if not vids:
            sys.exit("no videos found under Videos/")
        results = []
        for v in vids:
            r = analyse(v, args)
            if r:
                results.append(r)
                acsv = find_accel_csv(v)
                if acsv:
                    accel_compare(r, acsv, args)
        print("\n=== summary")
        print(f"{'video':44s} {'steady':>8s} {'sd':>5s} {'g tan':>7s} {'eps':>5s}")
        for r in results:
            name = os.path.relpath(r["video"], os.path.join(ROOT, "Videos"))
            print(f"{name[:44]:44s} {r['steady']:7.1f} {r['steady_sd']:5.1f} "
                  f"{G*np.tan(np.radians(r['steady'])):7.1f} {r['eps']:5.1f}")
        rows = [{"video": os.path.relpath(r["video"], ROOT), "steady_theta_deg": r["steady"],
                 "sd": r["steady_sd"], "g_tan_theta": G * np.tan(np.radians(r["steady"])),
                 "eps_deg": r["eps"], "roll_deg": r["roll"]} for r in results]
        out = os.path.join(args.outdir or os.path.join(ROOT, "output", "report"), "flyout_summary.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Summary: {out}")
        return

    if not args.video:
        sys.exit("give a video path, or --all")
    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")
    r = analyse(args.video, args)
    if r:
        acsv = find_accel_csv(args.video)
        if acsv:
            accel_compare(r, acsv, args)


if __name__ == "__main__":
    main()
