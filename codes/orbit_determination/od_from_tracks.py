"""Angles-only orbit determination of LEO objects from YOLO + BoT-SORT tracks (detect_track.py).

Steps
  1. Box centre (+1 px, undoing the label padding) -> camera angles at t + 0.5 s: each label box
     spans the object's motion from t to t+1, so its centre is the mid-exposure position.
  2. LEO tracks are selected by the detector's own class (0 = LEO, SMA <= 8413 km).
  3. Each track -> J2000 lines of sight -> IOD + batch least squares (run_od.solve_object).
  4. Evaluation only: every track point is matched to the nearest ground-truth object in pixels,
     each track gets the majority truth ID, and the orbit is scored against Target_States.csv.
     A baseline runs the same OD on ground-truth angles over the same image window.

Usage (from the repository root):
    python codes/orbit_determination/od_from_tracks.py
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dynamics import RE, propagate
from observations import POS, VEL, build_measurements, load_attitude, load_observer, load_tables
from run_od import solve_object

W, F_MM, H_MM = 4418, 35.0, 14.1376      # image size [px], focal length, sensor width [mm]
ARCSEC_PER_PX = np.degrees(H_MM / W / F_MM) * 3600


def pixel_to_angles(x, y):
    """Inverse of YOLODatasetGenerator.image_to_pixel_coordinates (full-image pixels)."""
    a1 = np.degrees(np.arctan((np.asarray(x) / W - 0.5) * H_MM / F_MM))
    a2 = np.degrees(np.arctan((0.5 - np.asarray(y) / W) * H_MM / F_MM))
    return a1, a2


def angles_to_pixel(a1, a2):
    return ((np.tan(np.radians(a1)) * F_MM / H_MM + 0.5) * W,
            (0.5 - np.tan(np.radians(a2)) * F_MM / H_MM) * W)


def truth_midpoints(tracking, t):
    """Truth pixel positions at t + 0.5 s (mean of t and t+1) for objects seen in both frames."""
    a = tracking[tracking['t'] == t].set_index('Object')
    b = tracking[tracking['t'] == t + 1].set_index('Object')
    common = a.index.intersection(b.index)
    x0, y0 = angles_to_pixel(a.loc[common, 'Angle1'].values, a.loc[common, 'Angle2'].values)
    x1, y1 = angles_to_pixel(b.loc[common, 'Angle1'].values, b.loc[common, 'Angle2'].values)
    return common.values, (x0 + x1) / 2, (y0 + y1) / 2


def match_to_truth(points, tracking, max_px):
    """Nearest truth object (within max_px) and pixel error for each detection/track point."""
    points = points.copy()
    points[['truth', 'dx', 'dy']] = np.nan
    for t, g in points.groupby('t'):
        ids, mx, my = truth_midpoints(tracking, t)
        if len(ids) == 0:
            continue
        d = np.hypot(g['xc'].values[:, None] - mx[None], g['yc'].values[:, None] - my[None])
        j = d.argmin(axis=1)
        ok = d[np.arange(len(g)), j] <= max_px
        points.loc[g.index[ok], 'truth'] = ids[j[ok]]
        points.loc[g.index[ok], 'dx'] = (g['xc'].values - mx[j])[ok]
        points.loc[g.index[ok], 'dy'] = (g['yc'].values - my[j])[ok]
    return points


def snap_to_detections(tr, det, max_px=3.0):
    """Attach each track point to the nearest raw detection in the same frame.

    ObjectTracker.track_objects reads the score column as the class (track[5] is the score in
    [x1, y1, x2, y2, id, score, cls, idx]), so its class is always 0; the detection carries the
    real class. The raw (not Kalman-smoothed) box centre is also the actual measurement.
    """
    tr = tr.copy()
    tr[['cls', 'conf', 'x_raw', 'y_raw']] = np.nan
    for t, g in tr.groupby('t'):
        d = det[det['t'] == t]
        if d.empty:
            continue
        dist = np.hypot(g['x'].values[:, None] - d['x'].values[None], g['y'].values[:, None] - d['y'].values[None])
        j = dist.argmin(axis=1)
        ok = dist[np.arange(len(g)), j] <= max_px
        tr.loc[g.index[ok], ['cls', 'conf', 'x_raw', 'y_raw']] = d[['cls', 'conf', 'x', 'y']].values[j[ok]]
    return tr


def truth_at(states, obj, times):
    """Truth J2000 states of `obj` at fractional times (J2 propagation from the previous second)."""
    s = states[states['Object'] == obj].set_index('t')[POS + VEL]
    out = np.empty((len(times), 6))
    for i, t in enumerate(times):
        t0 = np.floor(t)
        out[i] = propagate(s.loc[t0].values, t0, [t])[0] if t > t0 else s.loc[t0].values
    return out


def summarise(res, label):
    ok = res[res['status'].str.startswith('ok')]
    print(f'\n{label}: solved {len(ok)}/{len(res)}')
    if ok.empty:
        return
    cols = ['pos_err_km', 'vel_err_ms', 'da_km', 'di_deg', 'draan_deg', 'du_deg', 'rms_ang_arcsec', 'sigma_pos_km']
    print(ok[cols].abs().describe(percentiles=[0.5, 0.9]).T[['50%', '90%', 'max']]
          .to_string(float_format=lambda v: f'{v:.4g}'))
    for thr in (1.0, 10.0, 100.0):
        print(f'  epoch position error < {thr:g} km: {(ok["pos_err_km"] < thr).mean() * 100:.1f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/SpaceNet-100/images')
    ap.add_argument('--tracks', default='results/orbit_determination/detections/tracks.csv')
    ap.add_argument('--box', choices=['raw', 'track'], default='raw',
                    help='measure angles from the raw detection box or the Kalman-smoothed track box')
    ap.add_argument('--observer', choices=['auto', 'metadata', 'trilateration'], default='auto')
    ap.add_argument('--attitude', choices=['rsw', 'quaternion'], default='rsw')
    ap.add_argument('--min-points', type=int, default=10)
    ap.add_argument('--match-px', type=float, default=5.0, help='truth association radius, px (evaluation only)')
    ap.add_argument('--angle-sigma', type=float, default=None,
                    help='measurement sigma, arcsec (default: measured detector RMS error)')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--out', default='results/orbit_determination')
    a = ap.parse_args()

    tracking, states = load_tables(a.data)
    tracking['t'] = tracking['t'].astype(int)
    states['t'] = states['t'].astype(int)
    observer = load_observer(a.data, tracking, states, a.observer)
    attitude = load_attitude(a.data, a.attitude)
    print(f'Observer: {observer.description} | attitude: {a.attitude}')

    tr = snap_to_detections(pd.read_csv(a.tracks).drop(columns='cls'),
                            pd.read_csv(os.path.join(os.path.dirname(a.tracks), 'detections.csv')))
    print(f'Track points snapped to a raw detection: {tr["cls"].notna().mean() * 100:.1f}%')
    if a.box == 'raw':
        tr = tr.dropna(subset=['x_raw'])
        tr['x'], tr['y'] = tr['x_raw'], tr['y_raw']
    tr['xc'], tr['yc'] = tr['x'] + 1.0, tr['y'] + 1.0          # undo the -1 px label-centre offset
    tr = match_to_truth(tr, tracking, a.match_px)
    err_px = np.hypot(tr['dx'], tr['dy']).dropna()
    sigma = a.angle_sigma or float(np.sqrt(np.mean(err_px**2) / 2) * ARCSEC_PER_PX)
    print(f'Track points: {len(tr)} | matched to truth: {tr["truth"].notna().mean() * 100:.1f}% | '
          f'pixel error median {err_px.median():.2f} px, RMS {np.sqrt(np.mean(err_px**2)):.2f} px '
          f'(1 px = {ARCSEC_PER_PX:.1f}") -> per-axis sigma {sigma:.1f}"')

    # Truth regime of each object (evaluation): LEO = SMA <= 8413 km, the detector's class-0 rule.
    sma = states.sort_values('t').groupby('Object')['SMA'].first()
    rows = []
    for tid, g in tr.groupby('track_id'):
        truth_ids = g['truth'].dropna()
        maj = truth_ids.mode().iloc[0] if len(truth_ids) else np.nan
        cls = g['cls'].dropna()
        rows.append(dict(track_id=tid, n=len(g), cls=int(cls.mode().iloc[0]) if len(cls) else -1, truth=maj,
                         purity=(truth_ids == maj).sum() / len(g) if len(truth_ids) else 0.0,
                         truth_leo=bool(sma.get(maj, np.inf) <= 8413) if pd.notna(maj) else False))
    tracks = pd.DataFrame(rows)
    leo = tracks[tracks['cls'] == 0]
    truth_leo_ids = set(sma.index[sma <= 8413]) & set(tracking.loc[tracking['t'] < tr['t'].max() + 1, 'Object'])
    print(f'Tracks: {len(tracks)} | predicted LEO (class 0): {len(leo)} '
          f'(truly LEO: {leo["truth_leo"].sum()}, purity median {leo["purity"].median():.2f}) | '
          f'distinct truth LEO objects tracked: {leo.loc[leo["truth_leo"], "truth"].nunique()} '
          f'of {len(truth_leo_ids)} LEO objects in view')

    cfg = dict(mode='angles', angle_noise=0.0, range_noise=0.0, seed=0, alt_min=150.0, alt_max=2000.0,
               angle_sigma=sigma)
    jobs, base_jobs = [], []
    for _, row in leo[(leo['n'] >= a.min_points) & leo['truth'].notna()].iterrows():
        g = tr[tr['track_id'] == row['track_id']].sort_values('t').drop_duplicates('t')
        a1, a2 = pixel_to_angles(g['xc'].values, g['yc'].values)
        m = build_measurements(pd.DataFrame({'Object': row['track_id'], 't': g['t'].values + 0.5,
                                             'Angle1': a1, 'Angle2': a2, 'Range': np.nan}), observer, attitude)
        t = m['t'].values
        jobs.append((row['track_id'], t, m[['ox', 'oy', 'oz']].values, m[['lx', 'ly', 'lz']].values,
                     m['Range'].values, truth_at(states, row['truth'], t), cfg))
        # Baseline: ground-truth angles of the same object over the same seconds.
        gt = tracking[(tracking['Object'] == row['truth']) & tracking['t'].between(g['t'].min(), g['t'].max())]
        mb = build_measurements(gt, observer, attitude)
        base_jobs.append((row['track_id'], mb['t'].values.astype(float), mb[['ox', 'oy', 'oz']].values,
                          mb[['lx', 'ly', 'lz']].values, mb['Range'].values,
                          states.set_index(['Object', 't']).loc[[(row['truth'], int(x)) for x in mb['t']], POS + VEL].values,
                          dict(cfg, angle_sigma=None)))
    print(f'Running OD on {len(jobs)} LEO tracks with >= {a.min_points} points')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = pd.DataFrame(list(pool.map(solve_object, jobs)))
        base = pd.DataFrame(list(pool.map(solve_object, base_jobs)))
    res = tracks.merge(res.rename(columns={'Object': 'track_id'}), on='track_id')
    base = tracks.merge(base.rename(columns={'Object': 'track_id'}), on='track_id')

    summarise(res, f'Detector tracks (sigma {sigma:.1f}")')
    summarise(base, 'Baseline: ground-truth angles, same objects and time window')
    os.makedirs(a.out, exist_ok=True)
    tr.to_csv(os.path.join(a.out, 'track_points_matched.csv'), index=False)
    res.to_csv(os.path.join(a.out, f'od_leo_tracks_{a.attitude}.csv'), index=False)
    base.to_csv(os.path.join(a.out, f'od_leo_tracks_truth_baseline_{a.attitude}.csv'), index=False)
    print(f'\nResults written to {a.out}')


if __name__ == '__main__':
    main()
