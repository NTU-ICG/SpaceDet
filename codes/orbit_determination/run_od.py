"""Orbit determination of LEO RSOs in SpaceDet (SpaceNet-100) and evaluation against ground truth.

Pipeline per object: J2000 lines of sight (+ range) -> IOD (Gauss, or Gibbs/Herrick-Gibbs with
range) -> batch least squares with two-body + J2 -> errors vs Target_States.csv.

Usage (from the repository root):
    python codes/orbit_determination/run_od.py --mode angles
    python codes/orbit_determination/run_od.py --mode angles-range --angle-noise 2 --range-noise 0.05
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from batch_ls import ARCSEC, batch_ls
from dynamics import RE, propagate, rv2coe
from iod import admissible_region, gauss, range_iod
from observations import POS, VEL, build_measurements, load_attitude, load_observer, load_tables

ELEMENTS = ['a', 'e', 'i', 'raan', 'aop', 'ta']


def add_noise(L, rho, sigma_ang_arcsec, sigma_rng_km, rng):
    """Perturb unit lines of sight by isotropic angular noise and ranges by Gaussian noise."""
    if sigma_ang_arcsec > 0:
        L = L + rng.normal(scale=sigma_ang_arcsec * ARCSEC, size=L.shape)
        L /= np.linalg.norm(L, axis=1)[:, None]
    if sigma_rng_km > 0:
        rho = rho + rng.normal(scale=sigma_rng_km, size=rho.shape)
    return L, rho


def split_passes(t, gap_s=60.0):
    """Index arrays of contiguous passes (sightings separated by less than gap_s)."""
    return np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > gap_s) + 1)


def iod_candidates(t, O, L, rho, use_range, alt_range):
    """Candidate states as (epoch, state) pairs from one pass of sightings."""
    n = len(t)
    k = [0, n // 2, n - 1]
    if use_range:
        return [(t[k[1]], range_iod(t[k], O[k], L[k], rho[k]))]
    cands = [(t[k[1]], x) for x in gauss(t[k], O[k], L[k])]
    cands += [(t[0], x) for x in admissible_region(t, O, L, alt_range)]
    return cands


def solve_object(args):
    obj, t, O, L, rho, truth, cfg = args
    rng = np.random.default_rng(cfg['seed'] + int(obj))
    L, rho = add_noise(L, rho, cfg['angle_noise'], cfg['range_noise'], rng)
    use_range = cfg['mode'] == 'angles-range'
    # 0.15" = quantisation of the 1e-4 deg angles; detector-derived angles pass their own sigma.
    sig_a = cfg.get('angle_sigma') or max(cfg['angle_noise'], 0.15)
    sig_r = max(cfg['range_noise'], 0.01)
    passes = split_passes(t)
    main = max(passes, key=len)                      # IOD on the longest pass
    t_ref = t[main[0]]
    row = dict(Object=obj, n_obs=len(t), n_passes=len(passes), arc_s=t[-1] - t[0],
               main_pass_s=t[main[-1]] - t[main[0]], t_ref=t_ref, status='ok')
    start = time.time()
    try:
        cands = iod_candidates(t[main], O[main], L[main], rho[main], use_range,
                               (cfg['alt_min'], cfg['alt_max'] + 500.0))
    except Exception as exc:  # degenerate geometry
        cands, row['status'] = [], f'iod_error: {exc}'
    best = None
    for t_iod, x_iod in cands:
        try:
            x0 = propagate(x_iod, t_iod, [t_ref])[0]
            if not np.all(np.isfinite(x0)):
                continue
            x, cov, fit = batch_ls(x0, t_ref, t[main], O[main], L[main], rho[main], use_range, sig_a, sig_r)
        except Exception:
            continue
        if best is None or fit.cost < best[3].cost:
            best = (t_iod, x_iod, x, fit, cov)
    row['n_iod_candidates'] = len(cands)
    if best is None:
        row['status'] = row['status'] if row['status'] != 'ok' else 'no_solution'
        return row
    t_iod, x_iod, x, fit, cov = best
    if len(passes) > 1:                              # extend the fit to all passes
        try:
            x_all, cov_all, fit_all = batch_ls(x, t_ref, t, O, L, rho, use_range, sig_a, sig_r)
            if fit_all.rms_ang_arcsec < max(3 * fit.rms_ang_arcsec, 2 * sig_a):
                x, cov, fit = x_all, cov_all, fit_all
            else:
                row['status'] = 'ok_main_pass_only'
        except Exception:
            row['status'] = 'ok_main_pass_only'
    row['runtime_s'] = time.time() - start
    row['rms_ang_arcsec'] = fit.rms_ang_arcsec
    row['sigma_pos_km'] = np.sqrt(np.trace(cov[:3, :3]))
    truth_ref = truth[main[0]]
    row['iod_pos_err_km'] = np.linalg.norm(x_iod[:3] - truth[np.searchsorted(t, t_iod), :3])
    row['pos_err_km'] = np.linalg.norm(x[:3] - truth_ref[:3])
    row['vel_err_ms'] = 1e3 * np.linalg.norm(x[3:] - truth_ref[3:])
    est_arc = propagate(x, t_ref, t)
    row['arc_rms_pos_err_km'] = np.sqrt(np.mean(np.sum((est_arc[:, :3] - truth[:, :3]) ** 2, axis=1)))
    ce, ct = rv2coe(x[:3], x[3:]), rv2coe(truth_ref[:3], truth_ref[3:])
    for name, e_, t_ in zip(ELEMENTS, ce, ct):
        row[f'{name}_est'], row[f'{name}_true'] = e_, t_
    row['da_km'] = ce[0] - ct[0]
    row['de'] = ce[1] - ct[1]
    row['di_deg'] = ce[2] - ct[2]
    row['draan_deg'] = (ce[3] - ct[3] + 180) % 360 - 180
    row['du_deg'] = (ce[4] + ce[5] - ct[4] - ct[5] + 180) % 360 - 180  # argument of latitude
    row['state_est'] = ' '.join(f'{v:.6f}' for v in x)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/SpaceNet-100/images')
    ap.add_argument('--mode', choices=['angles', 'angles-range'], default='angles')
    ap.add_argument('--angle-noise', type=float, default=0.0, help='added angular noise, arcsec (1-sigma)')
    ap.add_argument('--range-noise', type=float, default=0.0, help='added range noise, km (1-sigma)')
    ap.add_argument('--alt-max', type=float, default=2000.0,
                    help='LEO definition: truth apogee altitude cut for object selection, km; also the '
                         'admissible-region prior (+500 km margin)')
    ap.add_argument('--alt-min', type=float, default=150.0, help='admissible-region minimum perigee altitude, km')
    ap.add_argument('--observer', choices=['auto', 'metadata', 'trilateration'], default='auto',
                    help='observer ephemeris source (auto: Metadata.csv if present)')
    ap.add_argument('--attitude', choices=['rsw', 'quaternion'], default='rsw',
                    help='camera attitude: nominal RSW pointing law or Metadata.csv quaternions')
    ap.add_argument('--min-obs', type=int, default=10)
    ap.add_argument('--max-objects', type=int, default=None)
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/orbit_determination')
    cfg = vars(ap.parse_args())

    tracking, states = load_tables(cfg['data'])
    observer = load_observer(cfg['data'], tracking, states, cfg['observer'])
    print(f'Observer: {observer.description} | attitude: {cfg["attitude"]}')
    meas = build_measurements(tracking, observer, load_attitude(cfg['data'], cfg['attitude']))

    # LEO selection uses the truth orbit (perigee altitude), only to choose which objects to evaluate.
    first = states.sort_values('t').groupby('Object').first()
    apogee_alt = first['SMA'] * (1 + first['ECC']) - RE
    leo = set(first.index[apogee_alt <= cfg['alt_max']])

    truth = states.set_index(['Object', 't'])[POS + VEL]
    jobs, skipped = [], []
    for obj, g in meas[meas['Object'].isin(leo)].groupby('Object'):
        g = g.drop_duplicates('t')
        g = g[g.set_index(['Object', 't']).index.isin(truth.index)]
        if len(g) < cfg['min_obs']:
            skipped.append(obj)
            continue
        jobs.append((obj, g['t'].values, g[['ox', 'oy', 'oz']].values, g[['lx', 'ly', 'lz']].values,
                     g['Range'].values, truth.loc[list(zip(g['Object'], g['t']))].values, cfg))
    jobs = jobs[:cfg['max_objects']]
    print(f'LEO objects: {len(leo)} | solving {len(jobs)} | skipped (<{cfg["min_obs"]} obs): {len(skipped)}')

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        rows = list(pool.map(solve_object, jobs, chunksize=4))
    res = pd.DataFrame(rows)
    print(f'Done in {time.time() - t0:.1f} s')

    os.makedirs(cfg['out'], exist_ok=True)
    tag = f"{cfg['mode']}_ang{cfg['angle_noise']:g}_rng{cfg['range_noise']:g}_{cfg['attitude']}"
    path = os.path.join(cfg['out'], f'od_leo_{tag}.csv')
    res.to_csv(path, index=False)

    ok = res[res['status'] == 'ok']
    print(f'\nSolved {len(ok)}/{len(res)}  ({res["status"].value_counts().to_dict()})')
    cols = ['iod_pos_err_km', 'pos_err_km', 'vel_err_ms', 'arc_rms_pos_err_km', 'da_km', 'de', 'di_deg', 'draan_deg',
            'du_deg', 'rms_ang_arcsec']
    summary = ok[cols].abs().describe(percentiles=[0.5, 0.9]).T[['50%', '90%', 'max']]
    print(summary.to_string(float_format=lambda v: f'{v:.4g}'))
    for thr in (0.1, 1.0, 10.0):
        print(f'  epoch position error < {thr:g} km: {(ok["pos_err_km"] < thr).mean() * 100:.1f}%')
    print(f'\nResults written to {path}')


if __name__ == '__main__':
    main()
