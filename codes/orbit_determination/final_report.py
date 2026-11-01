"""Final LEO orbit-determination results vs ground truth (SpaceNet-100), one row per object and case.

For each estimated orbit (from run_od.py / od_from_tracks.py result files) this re-predicts the
measurements and compares them with the ground truth:
  * Angle1, Angle2 [deg] and Range [km] predicted from the estimated orbit vs Tracking_Data.csv,
    at the evaluation epoch and as RMS / max over every ground-truth sample in the tracked arc;
  * the six classical elements a, e, i, RAAN, AOP, TA (J2000/GCRS, osculating) vs the elements
    of the Target_States.csv J2000 state at the same epoch, plus position/velocity errors.

Evaluation epoch = first whole second of the tracked arc (where ground truth exists).
Note: Target_States.csv's own SMA..TA columns are in an of-date equatorial frame (i, RAAN, AOP
differ from J2000 by up to ~0.1-0.45 deg of precession), so true elements are recomputed in J2000.

Usage (from the repository root):
    python codes/orbit_determination/final_report.py
"""
import argparse
import os

import numpy as np
import pandas as pd

from dynamics import propagate, rv2coe
from observations import POS, VEL, load_attitude, load_observer, load_tables

CASES = [  # (case label, result file, object id column, measurements used)
    ('angles+range, ground-truth measurements', 'od_leo_angles-range_ang0_rng0_{att}.csv', 'Object', 'Angle1, Angle2, Range'),
    ('angles-only, ground-truth angles', 'od_leo_angles_ang0_rng0_{att}.csv', 'Object', 'Angle1, Angle2'),
    ('angles-only, YOLO detections', 'od_leo_tracks_{att}.csv', 'truth', 'YOLO box centres'),
]
ELEMENTS = [('a', '_km'), ('e', ''), ('i', '_deg'), ('raan', '_deg'), ('aop', '_deg'), ('ta', '_deg')]
ARCSEC = 3600.0


def predict_angles(states, times, observer, attitude):
    """Angle1, Angle2 [deg] and Range [km] of target states seen from the observer camera."""
    o = observer(times)
    rel = states[:, :3] - o[:, :3]
    rng = np.linalg.norm(rel, axis=1)
    u = np.einsum('nji,nj->ni', attitude(times, o), rel / rng[:, None])   # camera frame
    return np.degrees(np.arctan(u[:, 1] / u[:, 2])), np.degrees(np.arctan(u[:, 0] / u[:, 2])), rng


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/SpaceNet-100/images')
    ap.add_argument('--results', default='results/orbit_determination')
    ap.add_argument('--attitude', default='rsw', choices=['rsw', 'quaternion'])
    a = ap.parse_args()

    tracking, states = load_tables(a.data)
    observer = load_observer(a.data, tracking, states, 'auto')
    attitude = load_attitude(a.data, a.attitude)
    truth_states = states.set_index(['Object', 't'])[POS + VEL]

    # Sanity check of the measurement model: true states must reproduce the true angles/range.
    chk = tracking.merge(states[['Object', 't'] + POS + VEL], on=['Object', 't']).sample(5000, random_state=0)
    a1, a2, rg = predict_angles(chk[POS + VEL].values, chk['t'].values, observer, attitude)
    print(f'Model check (truth states -> measurements): Angle RMS '
          f'{np.sqrt(np.mean(((a1 - chk.Angle1) ** 2 + (a2 - chk.Angle2) ** 2) / 2)) * ARCSEC:.2f}", '
          f'Range RMS {np.sqrt(np.mean((rg - chk.Range) ** 2)) * 1e3:.1f} m')

    rows = []
    for case, fname, id_col, used in CASES:
        res = pd.read_csv(os.path.join(a.results, fname.format(att=a.attitude)))
        for _, r in res.iterrows():
            obj = r[id_col]
            row = dict(case=case, measurements_used=used, Object=int(obj) if pd.notna(obj) else None,
                       track_id=int(r['track_id']) if 'track_id' in r and pd.notna(r.get('track_id')) else None,
                       n_obs=r.get('n_obs'), arc_s=r.get('arc_s'), status=r.get('status'),
                       fit_rms_arcsec=r.get('rms_ang_arcsec'))
            if pd.isna(obj) or not str(r.get('status', '')).startswith('ok') or pd.isna(r.get('state_est')):
                rows.append(row)
                continue
            x_ref = np.array(r['state_est'].split(), float)
            gt = tracking[tracking['Object'] == obj]
            if 'track_id' in r and pd.notna(r.get('track_id')):      # detector track: its own seconds
                pts = pd.read_csv(os.path.join(a.results, 'track_points_matched.csv'))
                tt = pts.loc[pts['track_id'] == r['track_id'], 't']
                gt = gt[gt['t'].between(tt.min(), tt.max() + 1)]
            gt = gt[gt.set_index(['Object', 't']).index.isin(truth_states.index)].sort_values('t')
            t = gt['t'].values
            est = propagate(x_ref, r['t_ref'], t)
            tru = truth_states.loc[list(zip(gt['Object'], gt['t']))].values
            p1, p2, prng = predict_angles(est, t, observer, attitude)
            d1, d2, dr = (p1 - gt['Angle1'].values) * ARCSEC, (p2 - gt['Angle2'].values) * ARCSEC, prng - gt['Range'].values
            dpos = np.linalg.norm(est[:, :3] - tru[:, :3], axis=1)

            row.update(epoch=pd.Timestamp('2023-01-01') + pd.Timedelta(seconds=float(t[0])),
                       Angle1_est_deg=p1[0], Angle1_true_deg=gt['Angle1'].iloc[0], Angle1_err_arcsec=d1[0],
                       Angle2_est_deg=p2[0], Angle2_true_deg=gt['Angle2'].iloc[0], Angle2_err_arcsec=d2[0],
                       Range_est_km=prng[0], Range_true_km=gt['Range'].iloc[0], Range_err_km=dr[0],
                       Angle1_rms_err_arcsec=np.sqrt(np.mean(d1**2)), Angle2_rms_err_arcsec=np.sqrt(np.mean(d2**2)),
                       Angle_max_err_arcsec=np.max(np.hypot(d1, d2)),
                       Range_rms_err_km=np.sqrt(np.mean(dr**2)), Range_max_err_km=np.max(np.abs(dr)))
            ce, ct = rv2coe(est[0, :3], est[0, 3:]), rv2coe(tru[0, :3], tru[0, 3:])
            for (name, unit), e_, t_ in zip(ELEMENTS, ce, ct):
                row[f'{name}_est{unit}'], row[f'{name}_true{unit}'] = e_, t_
                row[f'{name}_err{unit}'] = wrap(e_ - t_) if name in ('raan', 'aop', 'ta') else e_ - t_
            row['arg_latitude_err_deg'] = wrap(ce[4] + ce[5] - ct[4] - ct[5])
            row.update(pos_err_km=dpos[0], vel_err_ms=1e3 * np.linalg.norm(est[0, 3:] - tru[0, 3:]),
                       pos_rms_err_over_arc_km=np.sqrt(np.mean(dpos**2)))
            rows.append(row)

    out = pd.DataFrame(rows)
    # Fits whose residual is far above the measurement noise did not converge to the right orbit.
    noise = np.where(out['case'].str.contains('YOLO'), 15.1, 0.15)
    out['converged'] = out['fit_rms_arcsec'] < 10 * noise
    path = os.path.join(a.results, f'final_leo_od_results_{a.attitude}.csv')
    out.to_csv(path, index=False, float_format='%.6g')

    stats = [('pos_err_km', 'Position error at epoch [km]'), ('vel_err_ms', 'Velocity error at epoch [m/s]'),
             ('Angle1_rms_err_arcsec', 'Angle1 RMS error over arc [arcsec]'),
             ('Angle2_rms_err_arcsec', 'Angle2 RMS error over arc [arcsec]'),
             ('Range_rms_err_km', 'Range RMS error over arc [km]'),
             ('a_err_km', 'a error [km]'), ('e_err', 'e error [-]'), ('i_err_deg', 'i error [deg]'),
             ('raan_err_deg', 'RAAN error [deg]'), ('aop_err_deg', 'AOP error [deg]'),
             ('ta_err_deg', 'TA error [deg]'), ('arg_latitude_err_deg', 'AOP+TA (arg. of latitude) error [deg]')]
    summ = []
    for case, g in out[out['converged']].groupby('case', sort=False):
        for col, label in stats:
            v = g[col].abs().dropna()
            summ.append(dict(case=case, metric=label, n_objects=len(v), median=v.median(),
                             p90=v.quantile(0.9), max=v.max()))
    summ = pd.DataFrame(summ)
    spath = os.path.join(a.results, f'final_leo_od_summary_{a.attitude}.csv')
    summ.to_csv(spath, index=False, float_format='%.6g')
    pd.set_option('display.width', 220)
    print('Median |error| of converged fits:')
    print(summ.pivot(index='metric', columns='case', values='median')
          .reindex(index=[l for _, l in stats], columns=[c[0] for c in CASES])
          .to_string(float_format=lambda v: f'{v:.4g}'))
    print(f'\nObjects: ' + ', '.join(f'{c}: {int(n)} ({int(k)} converged)' for c, n, k in
                                     zip(out.groupby('case', sort=False).size().index,
                                         out.groupby('case', sort=False).size(),
                                         out.groupby('case', sort=False)['converged'].sum())))
    print(f'Written {path}\n        {spath}')


if __name__ == '__main__':
    main()
