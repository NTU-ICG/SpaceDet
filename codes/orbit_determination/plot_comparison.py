"""Comparison plot: estimated LEO orbits vs ground truth (SpaceNet-100).

Panels
  1. Position error per object: YOLO-detector angles vs ground-truth angles (same 99 s).
  2. Angles-only error vs tracking-arc length, all LEO objects (ground-truth angles), with the
     detector tracks overlaid.
  3. Median position error by measurement type.

Usage (from the repository root, after run_od.py and od_from_tracks.py):
    python codes/orbit_determination/plot_comparison.py
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
DETECTOR, TRUTH = '#2a78d6', '#eb6834'      # categorical slots 1 and 2 (validated pair)


def fmt_km(v):
    return f'{v * 1e3:.3g} m' if v < 1 else f'{v:.3g} km' if v < 100 else f'{v:.0f} km'


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def log_axis(axis, fmt):
    """Log scale with readable tick labels (set after set_*scale, which resets formatters)."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    axis.set_minor_formatter(NullFormatter())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results/orbit_determination')
    ap.add_argument('--data', default='data/SpaceNet-100/images')
    ap.add_argument('--attitude', default='rsw')
    a = ap.parse_args()
    R = lambda name: pd.read_csv(os.path.join(a.results, name))

    det = R(f'od_leo_tracks_{a.attitude}.csv')
    base = R(f'od_leo_tracks_truth_baseline_{a.attitude}.csv')
    full = R(f'od_leo_angles_ang0_rng0_{a.attitude}.csv')
    rng_ = R(f'od_leo_angles-range_ang0_rng0_{a.attitude}.csv')
    full, rng_ = (d[d['status'].str.startswith('ok')] for d in (full, rng_))
    tracking = pd.read_csv(os.path.join(a.data, 'Tracking_Data.csv'), index_col=0)
    mean_range = tracking.groupby('Object')['Range'].mean()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1, 1.25, 1.1], 'wspace': 0.42})

    # 1. Per-object dumbbell: detector vs ground-truth angles, same objects and seconds.
    ax = axes[0]
    style(ax)
    m = det.merge(base[['track_id', 'pos_err_km']], on='track_id', suffixes=('', '_truth'))
    m = m.sort_values('pos_err_km')
    for y, (_, r) in enumerate(m.iterrows()):
        ax.plot([r['pos_err_km_truth'], r['pos_err_km']], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter(r['pos_err_km_truth'], y, s=70, color=TRUTH, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(r['pos_err_km'], y, s=70, color=DETECTOR, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(fmt_km(r['pos_err_km']), (r['pos_err_km'], y), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK_2)
        ax.annotate(fmt_km(r['pos_err_km_truth']), (r['pos_err_km_truth'], y), xytext=(-8, 0),
                    textcoords='offset points', va='center', ha='right', fontsize=9, color=INK_2)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([f'Object {int(o)}\n{mean_range[o]:.0f} km away' for o in m['truth']], fontsize=9)
    ax.set_xscale('log')
    log_axis(ax.xaxis, fmt_km)
    ax.set_xlim(0.3, 3000)              # room for the left-hand value labels
    ax.set_ylim(-0.6, len(m) - 0.4)
    ax.grid(False, axis='y')
    ax.set_xlabel('Position error at track start (log scale)', color=INK_2, fontsize=9.5)
    ax.set_title('Same 4 LEO objects, same 99 s', loc='left', fontsize=11.5, fontweight='bold', pad=22)
    ax.text(0, 1.02, 'Detector noise makes errors 7–50× larger', transform=ax.transAxes,
            fontsize=9.5, color=INK_2)

    # 2. Error vs arc length: all LEO objects (truth angles) + detector tracks.
    ax = axes[1]
    style(ax)
    ax.scatter(full['main_pass_s'], full['pos_err_km'], s=22, color=TRUTH, edgecolor=SURFACE, linewidth=0.8,
               alpha=0.85, zorder=2, label=f'Ground-truth angles, full visibility ({len(full)} LEO objects)')
    ax.scatter(det['main_pass_s'], det['pos_err_km'], s=70, color=DETECTOR, edgecolor=SURFACE, linewidth=2,
               zorder=3, label=f'YOLO detector angles ({len(det)} tracks)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    log_axis(ax.xaxis, lambda v: f'{v:g} s')
    log_axis(ax.yaxis, fmt_km)
    ax.set_ylim(0.005, 2e4)
    ax.axvline(99, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(95, 1.5e4, 'image window (100 frames) ', fontsize=8.5, color=INK_2, va='top', ha='right')
    ax.set_xlabel('Tracking arc length (log scale)', color=INK_2, fontsize=9.5)
    ax.set_ylabel('Position error at track start (log scale)', color=INK_2, fontsize=9.5)
    ax.set_title('Angles-only accuracy vs how long an object is tracked', loc='left', fontsize=11.5,
                 fontweight='bold', pad=22)
    ax.text(0, 1.02, 'Longer arcs → smaller errors; most LEO passes last < 1 min', transform=ax.transAxes,
            fontsize=9.5, color=INK_2)
    ax.legend(loc='lower left', fontsize=8.5, frameon=False, labelcolor=INK_2)

    # 3. Median error by measurement type.
    ax = axes[2]
    style(ax)
    rows = [('Angles + range\n(truth angles, 206 objects)', rng_['pos_err_km'].median(), TRUTH),
            ('Angles only, truth angles\n(full visibility, 206 objects)', full['pos_err_km'].median(), TRUTH),
            ('Angles only, truth angles\n(same 99 s, 4 objects)', base['pos_err_km'].median(), TRUTH),
            ('Angles only, YOLO detections\n(99 s, 4 objects)', det['pos_err_km'].median(), DETECTOR)]
    for y, (label, v, c) in enumerate(rows[::-1]):
        ax.barh(y, v, height=0.55, color=c, left=1e-4, zorder=2)
        ax.annotate(fmt_km(v), (v, y), xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=9.5, color=INK, fontweight='bold')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xscale('log')
    log_axis(ax.xaxis, fmt_km)
    ax.set_xlim(1e-4, 5e3)
    ax.set_xticks([1e-3, 1e-1, 1e1, 1e3])
    ax.grid(False, axis='y')
    ax.set_xlabel('Median position error (log scale)', color=INK_2, fontsize=9.5)
    ax.set_title('Median error by measurement type', loc='left', fontsize=11.5, fontweight='bold', pad=22)
    ax.text(0, 1.02, 'Range is what pins down LEO orbits', transform=ax.transAxes, fontsize=9.5, color=INK_2)

    fig.suptitle('SpaceNet-100 LEO orbit determination vs ground truth (Target_States.csv)',
                 x=0.06, ha='left', y=1.03, fontsize=14, fontweight='bold')
    out = os.path.join(a.results, f'od_comparison_{a.attitude}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
