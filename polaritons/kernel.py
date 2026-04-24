"""
Disorder scattering kernel K(q, k) and propagator F(q, Q, eta).

All inputs/outputs are in natural units.  Pass a Params object that has
already been converted via Params.to_natural().
"""

from __future__ import annotations
from functools import lru_cache
import numpy as np
from .parameters import Params


def make_propagator(p: Params):
    """
    Return a vectorised propagator function F(q, Q, eta).

    F(q, Q, eta) = -eta*q / (E_gap - E_bind - hbar^2*q^2/(2M) + Q + i*epsilon)
    """
    E_gap   = p.E_gap
    E_bind  = p.E_bind
    hbar    = p.hbar
    M       = p.M

    def F(q, Q, eta=1.0):
        return -eta * q / (E_gap - E_bind - (hbar**2 * q**2) / (2.0 * M) + Q + 1e-9j)

    return F


def make_kernel_gaussian(p: Params, n_gauss: int = 96):
    """
    Return a cached scalar kernel function K(q, k) using Gauss-Legendre
    quadrature over the angular variable theta for Gaussian-correlated disorder.

    K(q,k) = prefactor * ∫dθ exp(-xi^2/2 * |q-k|^2) * [bracket(|q-k|^2)]^2

    with bracket(p^2) = (p^2+shift_h)^{-3/2}/m_h^2 - (p^2+shift_e)^{-3/2}/m_e^2
    """
    D_0     = p.D_0
    M       = p.M
    m_prime = p.m_prime
    m_rest  = p.m_rest
    m_e     = p.m_e
    m_h     = p.m_h
    a       = p.a
    xi      = p.xi

    prefactor = (
        2 * D_0 * M**6 * m_prime**2 * m_rest**2
        / (np.pi**2 * a**6 * m_e**2 * m_h**2)
    )
    shift_e = 4 * M**2 / (a**2 * m_e**2)
    shift_h = 4 * M**2 / (a**2 * m_h**2)

    # Gauss-Legendre nodes/weights on [0, 2*pi]
    theta_x, theta_w = np.polynomial.legendre.leggauss(n_gauss)
    theta   = np.pi * (theta_x + 1.0)
    theta_w = np.pi * theta_w
    cos_theta = np.cos(theta)

    @lru_cache(maxsize=None)
    def K(q: float, k: float) -> float:
        p2      = k*k + q*q - 2*k*q*cos_theta
        gauss   = np.exp(-0.5 * xi**2 * p2)
        bracket = (p2 + shift_h)**(-1.5) / m_h**2 - (p2 + shift_e)**(-1.5) / m_e**2
        return float(prefactor * np.dot(theta_w, gauss * bracket**2))

    K_vec = np.vectorize(K, otypes=[float])
    return K, K_vec


def make_kernel_nongaussian(p: Params):
    """
    Return a kernel function K(q, k) using exact analytic form for
    white-noise (non-Gaussian correlated) disorder.
    """
    from scipy.integrate import quad

    D_0     = p.D_0
    M       = p.M
    m_prime = p.m_prime
    m_rest  = p.m_rest
    m_e     = p.m_e
    m_h     = p.m_h
    a       = p.a

    prefactor = (
        4 * D_0 * M**6 * m_prime**2 * m_rest**2
        / (np.pi * a**6 * m_e**2 * m_h**2)
    )

    @lru_cache(maxsize=None)
    def term3(q: float, k: float) -> float:
        def integrand(theta):
            p2_e = k**2 + q**2 - 2*k*q*np.cos(theta) + 4*M**2/(a**2 * m_e**2)
            p2_h = k**2 + q**2 - 2*k*q*np.cos(theta) + 4*M**2/(a**2 * m_h**2)
            return (-1/np.pi * m_e**2 * m_h**2) * p2_e**(-1.5) * p2_h**(-1.5)
        val, _ = quad(integrand, 0, 2*np.pi)
        return val

    def K(q: float, k: float) -> float:
        A_e = k**2 + q**2 + 4*M**2/(a**2 * m_e**2)
        A_h = k**2 + q**2 + 4*M**2/(a**2 * m_h**2)
        t1  = (A_e**2 + 2*k**2*q**2) / (m_e**4 * (A_e**2 - 4*k**2*q**2)**(2.5))
        t2  = (A_h**2 + 2*k**2*q**2) / (m_h**4 * (A_h**2 - 4*k**2*q**2)**(2.5))
        t3  = term3(q, k)
        return prefactor * (t1 + t2 + t3)

    K_vec = np.vectorize(K, otypes=[float])
    return K, K_vec
