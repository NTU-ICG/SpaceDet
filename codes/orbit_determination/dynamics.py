"""Two-body + J2 dynamics, propagation and orbital-element conversion (km, s, rad)."""
import numpy as np
from scipy.integrate import solve_ivp

MU = 398600.4418        # km^3/s^2
RE = 6378.137           # km
J2 = 1.08262668e-3


def eom(t, y, j2=True):
    r = y[:3]
    n = np.linalg.norm(r)
    a = -MU * r / n**3
    if j2:
        z2 = (r[2] / n) ** 2
        k = 1.5 * J2 * MU * RE**2 / n**5
        a = a + k * r * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])
    return np.concatenate((y[3:], a))


def propagate(x0, t0, times, j2=True, rtol=1e-11, atol=1e-11):
    """Propagate state x0 (6,) at time t0 to each of `times` (any order). Returns (len(times), 6)."""
    times, inverse = np.unique(np.atleast_1d(np.asarray(times, dtype=float)), return_inverse=True)
    out = np.empty((len(times), 6))
    for forward in (True, False):
        idx = np.where(times >= t0)[0] if forward else np.where(times < t0)[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(times[idx])] if forward else idx[np.argsort(-times[idx])]
        t_end = times[idx[-1]]
        if t_end == t0:
            out[idx] = x0
            continue
        sol = solve_ivp(eom, (t0, t_end), x0, method='DOP853', t_eval=times[idx],
                        args=(j2,), rtol=rtol, atol=atol)
        out[idx] = sol.y.T
    return out[inverse.ravel()]


def _stumpff(z):
    pos, neg = z > 1e-8, z < -1e-8
    sp, sn = np.sqrt(np.where(pos, z, 1.0)), np.sqrt(np.where(neg, -z, 1.0))
    C = np.where(pos, (1 - np.cos(sp)) / np.where(pos, z, 1.0),
                 np.where(neg, (np.cosh(sn) - 1) / np.where(neg, -z, 1.0), 0.5))
    S = np.where(pos, (sp - np.sin(sp)) / sp**3, np.where(neg, (np.sinh(sn) - sn) / sn**3, 1.0 / 6.0))
    return C, S


def kepler_positions(r0, v0, dt, iters=30):
    """Vectorised two-body positions (M, K, 3) for M states r0, v0 (M, 3) at K time offsets dt."""
    sqmu = np.sqrt(MU)
    r0n = np.linalg.norm(r0, axis=1)[:, None]
    vr0 = (np.sum(r0 * v0, axis=1))[:, None] / r0n
    alpha = 2.0 / r0n - np.sum(v0 * v0, axis=1)[:, None] / MU
    dt = np.asarray(dt, float)[None, :]
    chi = sqmu * np.abs(alpha) * dt
    for _ in range(iters):
        z = alpha * chi**2
        C, S = _stumpff(z)
        F = r0n * vr0 / sqmu * chi**2 * C + (1 - alpha * r0n) * chi**3 * S + r0n * chi - sqmu * dt
        dF = r0n * vr0 / sqmu * chi * (1 - z * S) + (1 - alpha * r0n) * chi**2 * C + r0n
        chi = chi - F / dF
    C, S = _stumpff(alpha * chi**2)
    f = 1 - chi**2 / r0n * C
    g = dt - chi**3 * S / sqmu
    return f[..., None] * r0[:, None, :] + g[..., None] * v0[:, None, :]


def rv2coe(r, v):
    """Classical elements [a, e, i, RAAN, AOP, TA] (km, -, deg x4) from a J2000 state."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v); hn = np.linalg.norm(h)
    nvec = np.cross([0, 0, 1.0], h); nn = np.linalg.norm(nvec)
    evec = ((vn**2 - MU / rn) * r - np.dot(r, v) * v) / MU
    e = np.linalg.norm(evec)
    a = 1.0 / (2.0 / rn - vn**2 / MU)
    inc = np.arccos(np.clip(h[2] / hn, -1, 1))
    raan = np.arccos(np.clip(nvec[0] / nn, -1, 1))
    if nvec[1] < 0:
        raan = 2 * np.pi - raan
    aop = np.arccos(np.clip(np.dot(nvec, evec) / (nn * e), -1, 1))
    if evec[2] < 0:
        aop = 2 * np.pi - aop
    ta = np.arccos(np.clip(np.dot(evec, r) / (e * rn), -1, 1))
    if np.dot(r, v) < 0:
        ta = 2 * np.pi - ta
    return np.array([a, e, *np.degrees([inc, raan, aop, ta])])
