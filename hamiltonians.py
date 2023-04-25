import qutip
import numpy as np

# reduced Planck constant
hbar = 1.054571817 * 10 ** -34 # J * s

# bare electron mass
m0 = 9.109 * 10 ** -39 # kg

# Spin-3/2 matrices
Sx = 0.5 * hbar * np.array([
    [0, np.sqrt(3), 0, 0],
    [np.sqrt(3), 0, 2, 0],
    [0, 2, 0, np.sqrt(3)],
    [0, 0, np.sqrt(3), 0]
])

Sy = 0.5 * hbar * np.array([
    [0, complex(0, np.sqrt(3)), 0, 0],
    [complex(0, np.sqrt(3)), 0, -2j, 0],
    [0, 2j, 0, complex(0, -np.sqrt(3))],
    [0, 0, complex(0, np.sqrt(3)), 0]
])

Sz = 0.5 * hbar * np.array([
    [3, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -3]
])

# Luttinger parameters
gamma1 = 1
gamma2 = 1
gamma3 = 1

# crystal momentum
kx = 1
ky = 1
kz = 1

K = np.array([kx, ky, kz])

# Luttinger Hamiltonian for hole spins
H = (gamma1 + 5 * gamma2 / 2) * ((hbar ** 2 * K ** 2) / (2 * m0)) \
    + gamma2 * (hbar ** 2 / m0) * (kx ** 2 * Sx ** 2 + ky ** 2 * Sy ** 2 + kz ** 2 * Sz ** 2) \
        + gamma3 * (hbar ** 2 / (2 * m0)) * (((kx * ky + ky * kx) / 2) * ((Sx * Sy + Sy * Sx) / 2)) \
            * (((ky * kz + kz * ky) / 2) * ((Sy * Sz + Sz * Sy) / 2)) * (((kz * kx + kx * kz) / 2) * ((Sz * Sx + Sx * Sz) / 2))

