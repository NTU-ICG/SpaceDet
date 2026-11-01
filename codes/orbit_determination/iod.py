"""Initial orbit determination: Gauss (angles-only) and Gibbs / Herrick-Gibbs (angles + range)."""
import numpy as np

from dynamics import MU, RE, kepler_positions, propagate


def gauss(t, R, L, refine_iters=20):
    """Gauss angles-only IOD from three sightings.

    t: (3,) times [s]; R: (3,3) observer positions; L: (3,3) unit lines of sight (rows).
    Returns a list of candidate (r2, v2) states at t[1], one per admissible root of the
    8th-order range polynomial, each refined with exact two-body f and g coefficients.
    """
    tau1, tau3 = t[0] - t[1], t[2] - t[1]
    tau = tau3 - tau1
    p1, p2, p3 = np.cross(L[1], L[2]), np.cross(L[0], L[2]), np.cross(L[0], L[1])
    D0 = L[0] @ p1
    if abs(D0) < 1e-14:
        return []
    D = np.array([[Ri @ p1, Ri @ p2, Ri @ p3] for Ri in R])
    A = (-D[0, 1] * tau3 / tau + D[1, 1] + D[2, 1] * tau1 / tau) / D0
    B = (D[0, 1] * (tau3**2 - tau**2) * tau3 / tau + D[2, 1] * (tau**2 - tau1**2) * tau1 / tau) / (6 * D0)
    E = R[1] @ L[1]
    a = -(A**2 + 2 * A * E + R[1] @ R[1])
    b = -2 * MU * B * (A + E)
    c = -(MU * B) ** 2
    roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    cands = []
    for r2 in roots[(np.abs(roots.imag) < 1e-6) & (roots.real > 6378.0)].real:
        r23 = r2**3
        rho = np.empty(3)
        rho[0] = ((6 * (D[2, 0] * tau1 / tau3 + D[1, 0] * tau / tau3) * r23
                   + MU * D[2, 0] * (tau**2 - tau1**2) * tau1 / tau3)
                  / (6 * r23 + MU * (tau**2 - tau3**2)) - D[0, 0]) / D0
        rho[1] = A + MU * B / r23
        rho[2] = ((6 * (D[0, 2] * tau3 / tau1 - D[1, 2] * tau / tau1) * r23
                   + MU * D[0, 2] * (tau**2 - tau3**2) * tau3 / tau1)
                  / (6 * r23 + MU * (tau**2 - tau1**2)) - D[2, 2]) / D0
        if np.any(rho <= 0):
            continue
        r = R + rho[:, None] * L
        f1, f3 = 1 - MU * tau1**2 / (2 * r23), 1 - MU * tau3**2 / (2 * r23)
        g1, g3 = tau1 - MU * tau1**3 / (6 * r23), tau3 - MU * tau3**3 / (6 * r23)
        v2 = (-f3 * r[0] + f1 * r[2]) / (f1 * g3 - f3 * g1)
        state = _gauss_refine(t, R, L, D, D0, r[1], v2, rho, refine_iters)
        if state is not None:
            cands.append(state)
    return cands


def _gauss_refine(t, R, L, D, D0, r2, v2, rho, iters):
    """Iteratively replace the series f, g with exact two-body values (Curtis Alg. 5.6)."""
    for _ in range(iters):
        x = np.concatenate((r2, v2))
        if not np.all(np.isfinite(x)) or 1.0 / (2 / np.linalg.norm(r2) - (v2 @ v2) / MU) <= 0:
            return None
        prop = propagate(x, t[1], [t[0], t[2]], j2=False, rtol=1e-10, atol=1e-10)
        # Exact f, g from the propagated positions: r_i = f_i r2 + g_i v2.
        M = np.column_stack((r2, v2))
        (f1, g1), (f3, g3) = (np.linalg.lstsq(M, p[:3], rcond=None)[0] for p in prop)
        den = f1 * g3 - f3 * g1
        c1, c3 = g3 / den, -g1 / den
        new = np.array([(-D[0, 0] + D[1, 0] / c1 - c3 / c1 * D[2, 0]) / D0,
                        (-c1 * D[0, 1] + D[1, 1] - c3 * D[2, 1]) / D0,
                        (-c1 / c3 * D[0, 2] + D[1, 2] / c3 - D[2, 2]) / D0])
        if np.any(new <= 0):
            return None
        r = R + new[:, None] * L
        r2, v2 = r[1], (-f3 * r[0] + f1 * r[2]) / den
        if np.max(np.abs(new - rho)) < 1e-8:
            break
        rho = new
    return np.concatenate((r2, v2))


def admissible_region(t, O, L, alt_range=(150.0, 2500.0), n_rho=120, n_rdot=121, n_eval=25, top_k=5):
    """Angles-only IOD by grid search over (range, range-rate) at t[0].

    The line of sight and its rate at t[0] come from a quadratic fit to the sightings; each grid
    point (rho, rho_dot) defines a full state that is kept only if its perigee/apogee altitudes fall
    inside `alt_range` (the LEO prior). Candidates are scored by the two-body angular residual over
    up to `n_eval` sightings. Returns up to `top_k` distinct states at t[0], best first.
    """
    dt = t - t[0]
    deg = 2 if len(t) >= 4 else 1
    l0, ldot = np.empty(3), np.empty(3)
    o0, odot = np.empty(3), np.empty(3)
    for j in range(3):
        cl, co = np.polyfit(dt, L[:, j], deg), np.polyfit(dt, O[:, j], min(3, len(t) - 1))
        l0[j], ldot[j] = np.polyval(cl, 0), np.polyval(np.polyder(cl), 0)
        o0[j], odot[j] = np.polyval(co, 0), np.polyval(np.polyder(co), 0)
    l0 /= np.linalg.norm(l0)
    ldot -= (ldot @ l0) * l0

    rho, rdot = np.meshgrid(np.geomspace(50.0, 12000.0, n_rho), np.linspace(-8.0, 8.0, n_rdot), indexing='ij')
    rho, rdot = rho.ravel(), rdot.ravel()
    r0 = o0 + rho[:, None] * l0
    v0 = odot + rdot[:, None] * l0 + rho[:, None] * ldot
    rn = np.linalg.norm(r0, axis=1)
    a = 1.0 / (2.0 / rn - np.sum(v0**2, axis=1) / MU)
    h = np.linalg.norm(np.cross(r0, v0), axis=1)
    e = np.sqrt(np.clip(1 - h**2 / (MU * np.where(a > 0, a, np.inf)), 0, None))
    ok = (a > 0) & (a * (1 - e) >= RE + alt_range[0]) & (a * (1 + e) <= RE + alt_range[1])
    if not ok.any():
        return []
    idx = np.unique(np.linspace(0, len(t) - 1, min(n_eval, len(t))).astype(int))
    pos = kepler_positions(r0[ok], v0[ok], dt[idx])
    los = pos - O[idx][None]
    los /= np.linalg.norm(los, axis=2)[..., None]
    cost = np.full(len(rho), np.inf)
    cost[ok] = np.nan_to_num(np.sum((los - L[idx][None]) ** 2, axis=(1, 2)), nan=np.inf)

    # Keep the best grid point of each local minimum in the (rho, rho_dot) grid.
    grid = cost.reshape(n_rho, n_rdot)
    pad = np.pad(grid, 1, constant_values=np.inf)
    neighbours = np.stack([pad[1 + di:1 + di + n_rho, 1 + dj:1 + dj + n_rdot]
                           for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj])
    minima = np.flatnonzero(np.isfinite(grid) & (grid <= neighbours.min(axis=0)))
    minima = minima[np.argsort(cost[minima])][:top_k]
    return [np.concatenate((r0[k], v0[k])) for k in minima]


def gibbs(r1, r2, r3):
    """Gibbs velocity at r2 from three coplanar position vectors."""
    n1, n2, n3 = (np.linalg.norm(x) for x in (r1, r2, r3))
    N = n1 * np.cross(r2, r3) + n2 * np.cross(r3, r1) + n3 * np.cross(r1, r2)
    Dv = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)
    S = r1 * (n2 - n3) + r2 * (n3 - n1) + r3 * (n1 - n2)
    return np.sqrt(MU / (np.linalg.norm(N) * np.linalg.norm(Dv))) * (np.cross(Dv, r2) / n2 + S)


def herrick_gibbs(t, r1, r2, r3):
    """Herrick-Gibbs velocity at r2 (suited to small angular separations)."""
    dt21, dt31, dt32 = t[1] - t[0], t[2] - t[0], t[2] - t[1]
    n1, n2, n3 = (np.linalg.norm(x) for x in (r1, r2, r3))
    return (-dt32 * (1 / (dt21 * dt31) + MU / (12 * n1**3)) * r1
            + (dt32 - dt21) * (1 / (dt21 * dt32) + MU / (12 * n2**3)) * r2
            + dt21 * (1 / (dt32 * dt31) + MU / (12 * n3**3)) * r3)


def range_iod(t, R, L, rho):
    """IOD from three angle+range sightings; returns the state at t[1]."""
    r = R + rho[:, None] * L
    ang = min(np.arccos(np.clip(r[i] @ r[i + 1] / np.linalg.norm(r[i]) / np.linalg.norm(r[i + 1]), -1, 1))
              for i in range(2))
    v2 = gibbs(*r) if np.degrees(ang) > 3.0 else herrick_gibbs(t, *r)
    return np.concatenate((r[1], v2))
