"""Batch weighted least-squares orbit refinement with a two-body + J2 force model."""
import numpy as np
from scipy.optimize import least_squares

from dynamics import propagate

ARCSEC = np.radians(1.0 / 3600.0)


def residuals(x, t_ref, t, O, L, rho, sigma_ang, sigma_rng, use_range):
    r = propagate(x, t_ref, t, rtol=1e-10, atol=1e-10)[:, :3] - O
    n = np.linalg.norm(r, axis=1)
    res = [((r / n[:, None]) - L).ravel() / sigma_ang]
    if use_range:
        res.append((n - rho) / sigma_rng)
    return np.concatenate(res)


def batch_ls(x0, t_ref, t, O, L, rho=None, use_range=False,
             sigma_ang_arcsec=1.0, sigma_rng_km=0.01, loss='soft_l1'):
    """Refine the state x0 at t_ref from line-of-sight (and optionally range) measurements.

    Returns (state, covariance, fit) where `fit` is the scipy result. Residuals are whitened,
    so the covariance is (J^T J)^-1 scaled by the post-fit reduced chi-square.
    """
    sigma_ang = sigma_ang_arcsec * ARCSEC
    fit = least_squares(residuals, x0, args=(t_ref, t, O, L, rho, sigma_ang, sigma_rng_km, use_range),
                        x_scale=[1, 1, 1, 1e-3, 1e-3, 1e-3], loss=loss, f_scale=3.0, max_nfev=200)
    J = fit.jac
    dof = max(len(fit.fun) - 6, 1)
    chi2 = np.sum(fit.fun**2) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * max(chi2, 1.0)
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.nan)
    # |unit-vector difference| ~ angular error, so sum the three components per sighting.
    fit.rms_ang_arcsec = np.sqrt(3 * np.mean(fit.fun[:3 * len(t)] ** 2)) * sigma_ang_arcsec
    return fit.x, cov, fit
