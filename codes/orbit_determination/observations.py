"""Load SpaceDet tracking data and convert it into J2000 line-of-sight measurements.

Measurement model (verified against Target_States.csv to ~1 arcsec):
  * Angle1/Angle2 are camera-frame bearing angles; the body-frame line of sight is
    u = normalize(tan(Angle2), tan(Angle1), 1).
  * The camera is fixed in the observer's RSW frame (R radial, S along-track, W orbit normal)
    with C_cam2rsw = Rx(-15 deg) @ Ry(15 deg), i.e. boresight ~ (0.259, 0.250, 0.933) in RSW.
  * Range is the observer-to-target distance in km.

Frames: the "Position"/"Velocity" columns are ITRS and the "J2000" columns are GCRS as computed by
astropy (they agree to < 1 mm), so Metadata.csv's geodetic observer track is converted the same way.
Metadata.csv's quaternion (scalar first) rotates GCRS into the satellite body; the camera is the
body frame turned 180 deg about the boresight. It carries ~36" of simulated attitude-knowledge
noise per epoch relative to the RSW pointing law above.
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from dynamics import propagate

EPOCH = pd.Timestamp('2023-01-01 00:00:00')
POS = ['Px-J2000', 'Py-J2000', 'Pz-J2000']
VEL = ['Vx-J2000', 'Vy-J2000', 'Vz-J2000']


def _rx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


C_CAM2RSW = _rx(-15.0) @ _ry(15.0)
C_CAM2BODY = np.diag([-1.0, -1.0, 1.0])


def _metadata_times(md):
    return (pd.to_datetime(md['Timestamp']) - EPOCH).dt.total_seconds().values


def load_tables(data_dir):
    tracking = pd.read_csv(os.path.join(data_dir, 'Tracking_Data.csv'), index_col=0)
    states = pd.read_csv(os.path.join(data_dir, 'Target_States.csv'), index_col=0)
    for df in (tracking, states):
        df['t'] = (pd.to_datetime(df['Timestamp']) - EPOCH).dt.total_seconds()
    return tracking, states


def body_los(angle1_deg, angle2_deg):
    a1, a2 = np.radians(angle1_deg), np.radians(angle2_deg)
    u = np.column_stack((np.tan(a2), np.tan(a1), np.ones_like(a1)))
    return u / np.linalg.norm(u, axis=1)[:, None]


def rsw_basis(r, v):
    """(N,3,3) matrices whose columns are the R, S, W unit vectors in J2000."""
    R = r / np.linalg.norm(r, axis=1)[:, None]
    W = np.cross(r, v); W /= np.linalg.norm(W, axis=1)[:, None]
    S = np.cross(W, R)
    return np.stack((R, S, W), axis=2)


class ObserverEphemeris:
    """Observer (sensor satellite) J2000 ephemeris as a single J2 orbit fit."""

    def __init__(self, x0, t0):
        self.x0, self.t0 = np.asarray(x0, float), float(t0)

    def __call__(self, times):
        return propagate(self.x0, self.t0, times)

    @classmethod
    def from_metadata(cls, path):
        """Fit to the tracker states in Metadata.csv (shipped with SpaceDet-full)."""
        md = pd.read_csv(path)
        t = (pd.to_datetime(md['Timestamp']) - EPOCH).dt.total_seconds().values
        return cls._fit(t, md[POS].values, md[POS + VEL].values[0])

    @classmethod
    def from_ranges(cls, tracking, states, min_targets=4):
        """Reconstruct the observer by range trilateration on the ground-truth target positions.

        Stand-in for the observer metadata that SpaceNet-100 does not ship. It only recovers the
        shared observer orbit; target states are never passed to the OD estimator itself.
        """
        m = tracking.merge(states[['Object', 't'] + POS], on=['Object', 't'])
        ts, pos, guess = [], [], np.array([-6000.0, -1500.0, -3000.0])
        for t, g in m.groupby('t'):
            if len(g) < min_targets:
                continue
            r, rho = g[POS].values, g['Range'].values
            fit = least_squares(lambda o: np.linalg.norm(r - o, axis=1) - rho, guess)
            if np.abs(fit.fun).max() < 0.02:
                ts.append(t); pos.append(fit.x); guess = fit.x
        ts, pos = np.array(ts), np.array(pos)
        k = slice(0, 30)
        v0 = np.array([np.polyfit(ts[k] - ts[0], pos[k, j], 3)[-2] for j in range(3)])
        return cls._fit(ts, pos, np.concatenate((pos[0], v0)))

    @classmethod
    def _fit(cls, t, pos, x_guess):
        t0 = t[0]
        fit = least_squares(lambda x: (propagate(x, t0, t)[:, :3] - pos).ravel(), x_guess,
                            x_scale=[1, 1, 1, 1e-3, 1e-3, 1e-3])
        eph = cls(fit.x, t0)
        eph.fit_rms_km = np.sqrt(np.mean(np.sum((eph(t)[:, :3] - pos) ** 2, axis=1)))
        return eph


class SplineEphemeris:
    """Observer J2000 (GCRS) ephemeris from Metadata.csv's geodetic track (uses no target data).

    Positions are WGS84 latitude/longitude/altitude -> ITRS -> GCRS. They carry metre-level noise
    (GPS-like), and differentiating them directly gives 2-6 m/s velocity errors, which tilt the RSW
    camera frame by up to ~150". So the track is modelled as a J2 orbit fit plus a heavily smoothed
    spline of its residuals (unmodelled forces vary over minutes, the noise over seconds).
    smoothing_lam=3e7 sits in the middle of a broad optimum (1e7-1e8) of line-of-sight consistency
    with Tracking_Data on SpaceNet-100 (0.17" RMS); GCV under-smooths (0.56").
    """

    def __init__(self, path, smoothing_lam=3e7):
        import warnings
        from astropy import units as u
        from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
        from astropy.time import Time

        md = pd.read_csv(path)
        self.t = _metadata_times(md)
        loc = EarthLocation.from_geodetic(md['Longitude'].values * u.deg, md['Latitude'].values * u.deg,
                                          md['Altitude'].values * u.km, ellipsoid='WGS84')
        obstime = Time(md['Timestamp'].values.astype(str), scale='utc')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # astropy IERS / ERFA "dubious year" notices
            gcrs = ITRS(CartesianRepresentation(loc.x, loc.y, loc.z), obstime=obstime).transform_to(
                GCRS(obstime=obstime))
        pos = gcrs.cartesian.xyz.to(u.km).value.T
        k = slice(0, 31)
        v0 = np.array([np.polyfit(self.t[k] - self.t[0], pos[k, j], 3)[-2] for j in range(3)])
        self.orbit = ObserverEphemeris._fit(self.t, pos, np.concatenate((pos[0], v0)))
        res = pos - self.orbit(self.t)[:, :3]
        self.res_splines = [make_smoothing_spline(self.t, res[:, j], lam=smoothing_lam) for j in range(3)]
        smooth = np.column_stack([s(self.t) for s in self.res_splines])
        self.noise_rms_m = 1e3 * np.sqrt(np.mean(np.sum((res - smooth) ** 2, axis=1)))
        self.description = (f'Metadata.csv geodetic track -> GCRS ({len(self.t)} epochs): J2 fit '
                            f'(RMS {self.orbit.fit_rms_km * 1e3:.1f} m) + smoothed residuals '
                            f'(removed noise RMS {self.noise_rms_m:.1f} m)')

    def __call__(self, times):
        times = np.atleast_1d(np.asarray(times, float))
        x = self.orbit(times)
        x[:, :3] += np.column_stack([s(times) for s in self.res_splines])
        x[:, 3:] += np.column_stack([s.derivative()(times) for s in self.res_splines])
        return x


class QuaternionAttitude:
    """Camera-to-GCRS rotations from Metadata.csv quaternions, slerped between epochs."""

    def __init__(self, path):
        md = pd.read_csv(path)
        q = np.array([np.array(s.strip('[]').split(), float) for s in md['Quaternion']])
        gcrs_to_body = Rotation.from_quat(q[:, [1, 2, 3, 0]])          # scalar first -> scipy order
        self.slerp = Slerp(_metadata_times(md), gcrs_to_body.inv())

    def __call__(self, times, observer_states=None):
        return self.slerp(np.atleast_1d(times)).as_matrix() @ C_CAM2BODY


def rsw_attitude(times, observer_states):
    """Camera-to-GCRS rotations from the nominal pointing law (camera fixed in RSW)."""
    return rsw_basis(observer_states[:, :3], observer_states[:, 3:]) @ C_CAM2RSW


def load_observer(data_dir, tracking=None, states=None, source='auto'):
    """Observer ephemeris: Metadata.csv geodetic track (default) or range trilateration."""
    md = os.path.join(data_dir, 'Metadata.csv')
    if source == 'auto':
        source = 'metadata' if os.path.exists(md) else 'trilateration'
    if source == 'metadata':
        has_j2000 = pd.read_csv(md, usecols=['Px-J2000'])['Px-J2000'].notna().any()
        if not has_j2000:
            return SplineEphemeris(md)
        eph = ObserverEphemeris.from_metadata(md)
        eph.description = f'Metadata.csv J2000 states, J2 fit RMS {eph.fit_rms_km * 1e3:.1f} m'
        return eph
    eph = ObserverEphemeris.from_ranges(tracking, states)
    eph.description = f'range trilateration on truth target positions, J2 fit RMS {eph.fit_rms_km * 1e3:.1f} m'
    return eph


def load_attitude(data_dir, source='rsw'):
    return QuaternionAttitude(os.path.join(data_dir, 'Metadata.csv')) if source == 'quaternion' else rsw_attitude


def build_measurements(tracking, observer, attitude=rsw_attitude):
    """Add observer position and J2000 unit line-of-sight columns to the tracking table."""
    obs = tracking.sort_values(['Object', 't']).reset_index(drop=True)
    times = np.unique(obs['t'].values)
    eph = observer(times)
    row = np.searchsorted(times, obs['t'].values)
    o = eph[row]
    cam2gcrs = attitude(times, eph)[row]
    los = np.einsum('nij,nj->ni', cam2gcrs, body_los(obs['Angle1'].values, obs['Angle2'].values))
    obs[['ox', 'oy', 'oz']] = o[:, :3]
    obs[['lx', 'ly', 'lz']] = los
    return obs
